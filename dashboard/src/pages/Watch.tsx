import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
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

export default function Watch() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartsRef = useRef<HTMLCanvasElement>(null);
  const [meta, setMeta] = useState<any>({ status: "connecting" });
  const [scene, setScene] = useState<Scene | null>(null);
  const [last, setLast] = useState<Frame | null>(null);
  const [outcome, setOutcome] = useState<string>("");
  const [live, setLive] = useState(false);
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
    const es = new EventSource("/api/stream");
    es.addEventListener("snapshot", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setMeta(d.meta || {});
      setScene(d.scene);
      trail.current = (d.frames || []).map((f: Frame) => [f.pos[0], f.pos[2]]);
      trail3d.current = (d.frames || []).map((f: Frame) => [f.pos[0], f.pos[1], f.pos[2]]);
      epFrames.current = d.frames || [];
      const frames: Frame[] = d.frames || [];
      if (frames.length) setLast(frames[frames.length - 1]);
      setLive(true);
    });
    es.addEventListener("status", (e) => setMeta(JSON.parse((e as MessageEvent).data)));
    es.addEventListener("scene", (e) => {
      const s = JSON.parse((e as MessageEvent).data);
      setScene(s);
      trail.current = [];
      trail3d.current = [];
      epFrames.current = [];
      setOutcome("");
      setLive(true);
    });
    es.addEventListener("frame", (e) => {
      const f: Frame = JSON.parse((e as MessageEvent).data);
      setLast(f);
      epFrames.current.push(f);
      trail.current.push([f.pos[0], f.pos[2]]);
      trail3d.current.push([f.pos[0], f.pos[1], f.pos[2]]);
      if (trail3d.current.length > 1200) trail3d.current.splice(0, trail3d.current.length - 1200);
      if (trail.current.length > 1200) trail.current.splice(0, trail.current.length - 1200);
    });
    es.addEventListener("episode_end", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      const oc = d.succeeded ? "success" : d.collided ? "crash" : "timeout";
      setOutcome(oc === "success" ? "SUCCESS" : oc === "crash" ? "CRASH" : "TIMEOUT");
      const sc = sceneRef.current;
      if (sc) {
        const m = episodeMetrics(d.episode_id, sc, epFrames.current, oc);
        if (m) {
          metrics.current.push(m);
          if (metrics.current.length > MAX_EPS) metrics.current.splice(0, metrics.current.length - MAX_EPS);
          setEpCount(metrics.current.length);
        }
      }
    });
    es.onerror = () => setLive(false);
    es.onopen = () => setLive(true);
    return () => es.close();
  }, []);

  useEffect(() => {
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const cv = canvasRef.current;
      if (cv) {
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
            ctx.beginPath(); ctx.arc(toX(sc.goal[0]), toY(sc.goal[2]), 0.45 * scale, 0, Math.PI * 2); ctx.stroke();
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
      // charts
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
  }, []);

  const stale = !live;
  return (
    <div className="h-screen bg-[#0b0f14] text-gray-200 flex flex-col overflow-hidden">
      <header className="flex items-center justify-between px-4 py-2 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-3">
          <Link to="/" className="text-sm text-gray-400 hover:text-gray-200">&larr; Dashboard</Link>
          <h1 className="text-sm font-semibold">Live sim watch</h1>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className={`inline-block w-2 h-2 rounded-full ${stale ? "bg-red-500" : "bg-green-500"}`} />
          {stale ? "reconnecting" : "live"}
        </div>
      </header>
      <div className="flex-1 relative min-h-0">
        {view === "2d" ? (
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
        ) : (
          <div className="absolute inset-0">
            <WatchScene3D sceneRef={sceneRef} frameRef={lastRef} trail3dRef={trail3d} mode={camMode} onPick={setChassisId} />
          </div>
        )}
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
        <div className="absolute top-2 left-2 flex flex-col gap-1 text-[11px] pointer-events-none">
          <span className="px-2 py-1 rounded bg-gray-900/80 border border-gray-700 text-gray-200">
            policy: <span className="font-mono text-amber-300">{String(meta.policy || "bc_flat.json")}</span>
            {meta.policy_obs ? <span className="text-gray-400"> ({String(meta.policy_obs)})</span> : null}
          </span>
          {chassisId && (
            <span className="px-2 py-1 rounded bg-gray-900/80 border border-gray-700 text-gray-200">
              chassis: <span className="font-mono text-sky-300">{chassisId.replace(/^cad-chassis-/, "")}</span>
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
      <div className="h-[150px] md:h-[180px] shrink-0 border-t border-gray-800">
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
