import { useEffect, useRef, useState } from "react";
import WatchScene3D, { CamMode } from "../components/WatchScene3D";

type Scene = {
  episode_id: string; dist_id: string; seed: number;
  spawn: number[]; goal: number[]; obstacles: number[][];
  extent: number; ts: string;
};
type Frame = {
  episode_id: string; step: number; pos: number[]; quat: number[];
  vel: number[]; reward: number; done: boolean;
};
type EpMetric = {
  ep: string; outcome: "success" | "crash" | "timeout";
  air_s: number; jerk: number; xtrack: number; goal_dist: number;
};

const FPS = 20;
const MAX_EPS = 40;

function yawFromQuat(q: number[]): number {
  const [x, y, z, w] = q;
  return Math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + x * x));
}

// mean perpendicular distance from the straight spawn->goal line (3D)
function crossTrack(spawn: number[], goal: number[], p: number[]): number {
  const d = [goal[0] - spawn[0], goal[1] - spawn[1], goal[2] - spawn[2]];
  const dd = Math.hypot(d[0], d[1], d[2]) || 1;
  const a = [p[0] - spawn[0], p[1] - spawn[1], p[2] - spawn[2]];
  const t = (a[0] * d[0] + a[1] * d[1] + a[2] * d[2]) / (dd * dd);
  const proj = [spawn[0] + t * d[0], spawn[1] + t * d[1], spawn[2] + t * d[2]];
  return Math.hypot(p[0] - proj[0], p[1] - proj[1], p[2] - proj[2]);
}

function episodeMetrics(ep: string, scene: Scene, frames: Frame[],
  outcome: EpMetric["outcome"]): EpMetric | null {
  if (frames.length < 3) return null;
  const dt = 1 / FPS;
  // RMS jerk magnitude from finite-differenced velocities
  let jsq = 0, jn = 0;
  for (let i = 2; i < frames.length; i++) {
    const a1 = frames[i - 1].vel.map((v, k) => (v - frames[i - 2].vel[k]) / dt);
    const a2 = frames[i].vel.map((v, k) => (v - frames[i - 1].vel[k]) / dt);
    const j = Math.hypot(...a2.map((a, k) => (a - a1[k]) / dt));
    jsq += j * j; jn++;
  }
  const jerk = jn ? Math.sqrt(jsq / jn) : 0;
  let xt = 0;
  for (const f of frames) xt += crossTrack(scene.spawn, scene.goal, f.pos);
  const last = frames[frames.length - 1];
  return {
    ep, outcome,
    air_s: frames[frames.length - 1].step / FPS,
    jerk,
    xtrack: xt / frames.length,
    goal_dist: Math.hypot(...scene.goal.map((g, k) => g - last.pos[k])),
  };
}

const OUTCOME_COLOR = { success: "#2ecc71", crash: "#e74c3c", timeout: "#f1c40f" };

function drawSpark(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number,
  opts: { label: string; unit: string; vals: number[]; color: string }, eps: EpMetric[]) {
  const pad = 10, labelH = 14;
  ctx.fillStyle = "#8b95a5";
  ctx.font = "10px system-ui";
  ctx.fillText(opts.label, x + pad, y + labelH - 3);
  const n = Math.min(opts.vals.length, 30);
  if (!n) {
    ctx.fillStyle = "#3a4356";
    ctx.fillText("waiting for episodes", x + pad + 100, y + labelH - 3);
    return;
  }
  const vals = opts.vals.slice(-n);
  const max = Math.max(...vals);
  const min = Math.min(...vals);
  const span = max - min || max || 1e-6;
  const lastV = vals[n - 1];
  const lastM = eps[eps.length - 1];
  ctx.fillStyle = OUTCOME_COLOR[lastM.outcome];
  const shown = opts.unit === "%" ? `${Math.round(lastV)}%` : `${lastV.toFixed(1)}${opts.unit}`;
  ctx.fillText(shown, x + w - 50, y + labelH - 3);
  const cw = w - pad * 2, chh = h - labelH - 12;
  const top = y + labelH + 2;
  const px = (i: number) => x + pad + (n === 1 ? cw / 2 : (i * cw) / (n - 1));
  const py = (v: number) => top + chh * (1 - (v - min) / span);
  ctx.strokeStyle = "#1a2230";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x + pad, top + chh);
  ctx.lineTo(x + pad + cw, top + chh);
  ctx.stroke();
  ctx.beginPath();
  vals.forEach((v, i) => (i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(0), py(v))));
  ctx.strokeStyle = opts.color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.lineTo(px(n - 1), top + chh);
  ctx.lineTo(px(0), top + chh);
  ctx.closePath();
  ctx.fillStyle = opts.color + "1a";
  ctx.fill();
  for (let i = 0; i < n; i++) {
    const m = eps[eps.length - n + i];
    ctx.fillStyle = OUTCOME_COLOR[m.outcome];
    ctx.globalAlpha = 0.75;
    ctx.beginPath();
    ctx.arc(px(i), top + chh, 1.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = opts.color;
  ctx.beginPath();
  ctx.arc(px(n - 1), py(lastV), 2.5, 0, Math.PI * 2);
  ctx.fill();
}


// --- multi-source watch ------------------------------------------------------
// The server keeps one stream channel per experiment source. This page holds a
// single SSE connection, demultiplexes events by their `source` field, and
// renders one pane per selected source.

type SourceEntry = {
  id: string; label?: string; policy?: string; policy_obs?: string;
  dynamics?: string; status?: string;
  eval?: { goto?: number; hover_hold?: number; land?: number };
  updated_at?: string;
};
type ChanState = {
  meta: Record<string, unknown>;
  scene: Scene | null;
  frames: Frame[];
  last_event_at: string | null;
};
type Listener = (event: string, data: any) => void;

function WatchPane({ sourceId, entry, single, subscribe, snapshot }: {
  sourceId: string;
  entry?: SourceEntry;
  single: boolean;
  subscribe: (id: string, fn: Listener) => () => void;
  snapshot: (id: string) => ChanState | undefined;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartsRef = useRef<HTMLCanvasElement>(null);
  const [meta, setMeta] = useState<any>({ status: "connecting" });
  const [scene, setScene] = useState<Scene | null>(null);
  const [last, setLast] = useState<Frame | null>(null);
  const [outcome, setOutcome] = useState<string>("");
  const [epCount, setEpCount] = useState(0);
  const trail = useRef<[number, number][]>([]);
  const trail3d = useRef<[number, number, number][]>([]);
  const [view, setView] = useState<"2d" | "3d">("2d");
  const [camMode, setCamMode] = useState<CamMode>("orbit");
  const [chassisId, setChassisId] = useState<string | null>(null);
  const epFrames = useRef<Frame[]>([]);
  const metrics = useRef<EpMetric[]>([]);
  const sceneRef = useRef<Scene | null>(null);
  const lastRef = useRef<Frame | null>(null);
  sceneRef.current = scene;
  lastRef.current = last;

  useEffect(() => {
    const snap = snapshot(sourceId);
    if (snap) {
      setMeta(snap.meta || {});
      setScene(snap.scene as Scene | null);
      const frames = snap.frames || [];
      epFrames.current = frames;
      trail.current = frames.map((f) => [f.pos[0], f.pos[2]]);
      trail3d.current = frames.map((f) => [f.pos[0], f.pos[1], f.pos[2]]);
      if (frames.length) setLast(frames[frames.length - 1]);
    }
    return subscribe(sourceId, (event, data) => {
      if (event === "__snapshot__") {
        const snap = data as ChanState;
        setMeta(snap.meta || {});
        setScene(snap.scene as Scene | null);
        const frames = snap.frames || [];
        epFrames.current = frames;
        trail.current = frames.map((f) => [f.pos[0], f.pos[2]]);
        trail3d.current = frames.map((f) => [f.pos[0], f.pos[1], f.pos[2]]);
        if (frames.length) setLast(frames[frames.length - 1]);
        return;
      }
      if (event === "status") setMeta(data);
      else if (event === "scene") {
        setScene(data as Scene);
        trail.current = [];
        trail3d.current = [];
        epFrames.current = [];
        setOutcome("");
      } else if (event === "frame") {
        const f = data as Frame;
        setLast(f);
        epFrames.current.push(f);
        trail.current.push([f.pos[0], f.pos[2]]);
        trail3d.current.push([f.pos[0], f.pos[1], f.pos[2]]);
        if (trail3d.current.length > 1200) trail3d.current.splice(0, trail3d.current.length - 1200);
        if (trail.current.length > 1200) trail.current.splice(0, trail.current.length - 1200);
      } else if (event === "episode_end") {
        const oc = data.succeeded ? "success" : data.collided ? "crash" : "timeout";
        setOutcome(oc === "success" ? "SUCCESS" : oc === "crash" ? "CRASH" : "TIMEOUT");
        const sc = sceneRef.current;
        if (sc) {
          const m = episodeMetrics(data.episode_id, sc, epFrames.current, oc as EpMetric["outcome"]);
          if (m) {
            metrics.current.push(m);
            if (metrics.current.length > MAX_EPS) metrics.current.splice(0, metrics.current.length - MAX_EPS);
            setEpCount(metrics.current.length);
          }
        }
      }
    });
  }, [sourceId, subscribe, snapshot]);

  useEffect(() => {
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const cv = canvasRef.current;
      if (cv && (single || view === "2d")) {
        const dpr = window.devicePixelRatio || 1;
        const w = cv.clientWidth, h = cv.clientHeight;
        if (cv.width !== w * dpr || cv.height !== h * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
        const ctx = cv.getContext("2d");
        if (ctx) {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.fillStyle = "#0b0f14";
          ctx.fillRect(0, 0, w, h);
          const sc = sceneRef.current;
          const ext = sc ? Math.max(2, sc.extent) : 10;
          const scale = Math.min(w, h) / (ext * 2.2);
          const toX = (x: number) => w / 2 + x * scale;
          const toY = (z: number) => h / 2 + z * scale;
          ctx.strokeStyle = "#1a2230";
          ctx.lineWidth = 1;
          for (let r = ext / 2; r <= ext; r += ext / 2) {
            ctx.beginPath(); ctx.arc(w / 2, h / 2, r * scale, 0, Math.PI * 2); ctx.stroke();
          }
          if (sc) {
            for (const o of sc.obstacles) {
              ctx.fillStyle = "#3b2b2b";
              ctx.strokeStyle = "#7a3b3b";
              ctx.beginPath(); ctx.arc(toX(o[0]), toY(o[2]), o[3] * scale, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
            }
            ctx.strokeStyle = "#2ecc71";
            ctx.lineWidth = 2;
            const gr = (sc as any).success_radius ? Number((sc as any).success_radius) : 0.45;
            ctx.beginPath(); ctx.arc(toX(sc.goal[0]), toY(sc.goal[2]), Math.max(0.2, gr) * scale, 0, Math.PI * 2); ctx.stroke();
            ctx.fillStyle = "#888";
            ctx.beginPath(); ctx.arc(toX(sc.spawn[0]), toY(sc.spawn[2]), 3, 0, Math.PI * 2); ctx.fill();
          }
          const tr = trail.current;
          if (tr.length > 1) {
            ctx.strokeStyle = "#4aa3ff";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(toX(tr[0][0]), toY(tr[0][1]));
            for (const p of tr) ctx.lineTo(toX(p[0]), toY(p[1]));
            ctx.stroke();
          }
          const f = lastRef.current;
          if (f) {
            const yaw = yawFromQuat(f.quat);
            const x = toX(f.pos[0]), y = toY(f.pos[2]);
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(x, y);
            ctx.lineTo(x + Math.sin(yaw) * 12, y - Math.cos(yaw) * 12); ctx.stroke();
            ctx.fillStyle = "#ffd166";
            ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = "#ffd16655";
            ctx.beginPath(); ctx.arc(x, y, 4 + Math.max(0, f.pos[1]) * scale * 0.1, 0, Math.PI * 2); ctx.stroke();
          }
        }
      }
      const ch = chartsRef.current;
      if (ch) {
        const dpr = window.devicePixelRatio || 1;
        const w = ch.clientWidth, h = ch.clientHeight;
        if (ch.width !== w * dpr || ch.height !== h * dpr) { ch.width = w * dpr; ch.height = h * dpr; }
        const ctx = ch.getContext("2d");
        if (ctx) {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.fillStyle = "#0d1219";
          ctx.fillRect(0, 0, w, h);
          const eps = metrics.current;
          const succ = eps.map((_, i) => {
            const win = eps.slice(Math.max(0, i - 9), i + 1);
            return (win.filter((e) => e.outcome === "success").length / win.length) * 100;
          });
          const cells = [
            { label: "time in air", unit: "s", vals: eps.map((m) => m.air_s), color: "#4aa3ff" },
            { label: "success rate (last 10)", unit: "%", vals: succ, color: "#2ecc71" },
            { label: "jerk (RMS)", unit: " m/s3", vals: eps.map((m) => m.jerk), color: "#b088f9" },
            { label: "off-ideal-path", unit: " m", vals: eps.map((m) => m.xtrack), color: "#2dd4bf" },
          ];
          const cw = w / 2, chh = h / 2;
          cells.forEach((c, i) =>
            drawSpark(ctx, (i % 2) * cw, Math.floor(i / 2) * chh, cw, chh, c, eps));
          ctx.strokeStyle = "#141b26";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, chh); ctx.lineTo(w, chh);
          ctx.moveTo(cw, 0); ctx.lineTo(cw, h);
          ctx.stroke();
        }
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [single, view]);

  const show3d = view === "3d";  // per-tile: every pane gets its own 2D/3D + camera controls
  return (
    <div className={`flex flex-col min-h-0 ${single ? "flex-1" : "border-b border-gray-800"}`}>
      <div className={`relative ${single ? "flex-1 min-h-0" : "h-[46vh] shrink-0"}`}>
        {!show3d ? (
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
        ) : (
          <div className="absolute inset-0">
            <WatchScene3D sceneRef={sceneRef} frameRef={lastRef} trail3dRef={trail3d} mode={camMode} onPick={setChassisId} />
          </div>
        )}
        {(
          <div className="absolute top-2 right-2 flex gap-1 text-xs">
            <button onClick={() => setView("2d")}
              className={`px-2 py-1 rounded ${view === "2d" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-300"}`}>2D</button>
            <button onClick={() => setView("3d")}
              className={`px-2 py-1 rounded ${view === "3d" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-300"}`}>3D</button>
            {view === "3d" && (["orbit", "chase", "fpv"] as CamMode[]).map((m) => (
              <button key={m} onClick={() => setCamMode(m)}
                className={`px-2 py-1 rounded ${camMode === m ? "bg-emerald-600 text-white" : "bg-gray-800 text-gray-300"}`}>
                {m === "fpv" ? "FPV" : m}
              </button>
            ))}
          </div>
        )}
        <div className="absolute top-2 left-2 flex flex-col gap-1 text-[11px] pointer-events-none">
          <span className="px-2 py-1 rounded bg-gray-900/80 border border-gray-700 text-gray-200">
            policy: <span className="font-mono text-amber-300">{String(meta.policy || entry?.policy || "-")}</span>
            {(meta.policy_obs || entry?.policy_obs) ? <span className="text-gray-400"> ({String(meta.policy_obs || entry?.policy_obs)})</span> : null}
          </span>
          {show3d && chassisId && (
            <span className="px-2 py-1 rounded bg-gray-900/80 border border-gray-700 text-gray-200">
              chassis: <span className="font-mono text-sky-300">{chassisId.replace(/^cad-chassis-/, "")}</span>
            </span>
          )}
          {(scene as any)?.scenario && (
            <span className="px-2 py-1 rounded bg-gray-900/80 border border-gray-700 text-gray-200">
              {(scene as any).scenario} r={(scene as any).success_radius}m
            </span>
          )}
        </div>
        {outcome && (
          <div className={`absolute top-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded text-sm font-bold ${
            outcome === "SUCCESS" ? "bg-green-700" : outcome === "CRASH" ? "bg-red-700" : "bg-yellow-700"}`}>
            {outcome}
          </div>
        )}
      </div>
      <div className={`${single ? "h-[150px] md:h-[180px]" : "h-[110px]"} shrink-0 border-t border-gray-800`}>
        <canvas ref={chartsRef} className="w-full h-full" />
      </div>
      <footer className="px-4 py-1.5 border-t border-gray-800 text-xs text-gray-400 flex flex-wrap gap-x-4 gap-y-1 shrink-0">
        <span>dist: {scene?.dist_id || String(meta.dist_id || "-")}</span>
        <span>ep: {scene?.episode_id || "-"}</span>
        <span>step: {last?.step ?? "-"}</span>
        <span>alt: {last ? last.pos[1].toFixed(2) + "m" : "-"}</span>
        <span>speed: {last ? Math.hypot(...last.vel).toFixed(1) + "m/s" : "-"}</span>
        <span>eps scored: {epCount}</span>
      </footer>
    </div>
  );
}

export default function Watch({ embedded = false }: { embedded?: boolean }) {
  const [registry, setRegistry] = useState<SourceEntry[]>([]);
  const [liveIds, setLiveIds] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const chans = useRef<Map<string, ChanState>>(new Map());
  const subs = useRef<Map<string, Set<Listener>>>(new Map());

  const emit = (id: string, event: string, data: any) => {
    const set = subs.current.get(id);
    if (set) for (const fn of set) { try { fn(event, data); } catch { /* pane left */ } }
  };
  const subscribe = (id: string, fn: Listener) => {
    let set = subs.current.get(id);
    if (!set) { set = new Set(); subs.current.set(id, set); }
    set.add(fn);
    return () => { set.delete(fn); };
  };
  const snapshot = (id: string) => chans.current.get(id);

  useEffect(() => {
    const es = new EventSource("/api/stream");
    const note = (id: string, patch: Partial<ChanState>) => {
      const cur = chans.current.get(id) || { meta: {}, scene: null, frames: [], last_event_at: null };
      chans.current.set(id, { ...cur, ...patch });
      setLiveIds([...chans.current.keys()]);
    };
    es.addEventListener("snapshot", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      chans.current = new Map(Object.entries(d.sources || {}));
      setRegistry(d.registry || []);
      setLiveIds([...chans.current.keys()]);
      for (const [id, ch] of chans.current) emit(id, "__snapshot__", ch);
    });
    es.addEventListener("status", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      const { source, ...meta } = d;
      note(source, { meta, last_event_at: new Date().toISOString() });
      emit(source, "status", meta);
    });
    es.addEventListener("scene", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      const { source, ...scene } = d;
      note(source, { scene: scene as Scene, frames: [], last_event_at: new Date().toISOString() });
      emit(source, "scene", scene);
    });
    es.addEventListener("frame", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      const cur = chans.current.get(d.source);
      if (cur) {
        cur.frames.push(d);
        if (cur.frames.length > 400) cur.frames.splice(0, cur.frames.length - 400);
        cur.last_event_at = d.ts;
      } else {
        note(d.source, { frames: [d], last_event_at: d.ts });
      }
      emit(d.source, "frame", d);
    });
    es.addEventListener("episode_end", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      emit(d.source, "episode_end", d);
    });
    es.onerror = () => setConnected(false);
    es.onopen = () => setConnected(true);
    const poll = setInterval(() => {
      fetch("/api/stream/sources").then((r) => r.json()).then((r) => setRegistry(r)).catch(() => {});
    }, 30000);
    fetch("/api/stream/sources").then((r) => r.json()).then((r) => setRegistry(r)).catch(() => {});
    return () => { es.close(); clearInterval(poll); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // known sources: registry entries + any live channel not registered
  const known: SourceEntry[] = [...registry];
  for (const id of liveIds) {
    if (!known.some((k) => k.id === id)) known.push({ id, label: id === "default" ? "Live policy" : id });
  }
  // default selection: first known source
  const sel = selected.length ? selected.filter((s) => known.some((k) => k.id === s)) : known.slice(0, 1).map((k) => k.id);
  const toggle = (id: string) =>
    setSelected((cur) => {
      const base = cur.length ? cur : known.slice(0, 1).map((k) => k.id);
      return base.includes(id) ? base.filter((s) => s !== id) : [...base, id];
    });

  return (
    <div className={`bg-[#0b0f14] text-gray-200 flex flex-col ${embedded ? (sel.length <= 1 ? "h-[calc(100dvh-10rem)] md:h-[calc(100dvh-9.5rem)] overflow-hidden" : "") : (sel.length <= 1 ? "h-screen overflow-hidden" : "min-h-screen")}`}>
      {!embedded && (
      <header className="flex items-center justify-between px-4 py-2 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-semibold">Live sim watch</h1>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className={`inline-block w-2 h-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`} />
          {connected ? "live" : "reconnecting"}
        </div>
      </header>)}
      <div className="flex-1 min-h-0 flex flex-col md:flex-row">
      <div className="flex gap-2 p-2 overflow-x-auto border-b border-gray-800 shrink-0 md:w-64 md:flex-col md:overflow-x-hidden md:overflow-y-auto md:border-b-0 md:border-r">
        <div className="hidden md:block px-1.5 pt-1 pb-0.5 text-[10px] font-semibold uppercase tracking-wider text-gray-500">Policies</div>
        {known.map((k) => {
          const on = sel.includes(k.id);
          const live = chans.current.get(k.id)?.meta?.status === "streaming";
          return (
            <button key={k.id} onClick={() => toggle(k.id)}
              className={`shrink-0 text-left px-3 py-2 rounded-lg border text-xs transition-colors ${
                on ? "border-primary/60 bg-primary/10" : "border-gray-800 bg-gray-900/50 hover:border-gray-600"}`}>
              <div className="flex items-center gap-1.5">
                <span className={`inline-block w-1.5 h-1.5 rounded-full ${live ? "bg-emerald-500" : "bg-gray-600"}`} />
                <span className="font-semibold text-gray-100">{k.label || k.id}</span>
              </div>
              <div className="text-gray-400 font-mono mt-1 truncate">
                {k.policy || "-"}{k.policy_obs ? ` - ${k.policy_obs}` : ""}
              </div>
              {k.eval && (
                <div className="flex gap-1.5 mt-1.5">
                  {(["goto", "hover_hold", "land"] as const).map((kk) => (
                    <span key={kk} className="rounded bg-gray-800/80 px-1.5 py-0.5 text-[10px] text-gray-300 font-mono">
                      {kk === "hover_hold" ? "hov" : kk} {Math.round(((k.eval as any)[kk] ?? 0) * 100)}%
                    </span>
                  ))}
                </div>
              )}
            </button>
          );
        })}
      </div>
      <div className={`flex-1 min-h-0 ${sel.length > 1 ? "grid md:grid-cols-2" : "flex flex-col"}`}>
        {sel.map((id) => (
          <WatchPane key={id} sourceId={id} entry={known.find((k) => k.id === id)}
            single={sel.length === 1} subscribe={subscribe} snapshot={snapshot} />
        ))}
        {sel.length === 0 && (
          <div className="p-6 text-sm text-gray-400">No experiment selected - pick a policy from the list.</div>
        )}
      </div>
      </div>
    </div>
  );
}
