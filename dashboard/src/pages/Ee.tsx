import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CircuitBoard, CheckCircle2, XCircle, Star, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import CadViewer from "@/components/CadViewer";

type EeVerify = { kind: string; label?: string; name: string };
type EeVersion = {
  version: number;
  created_at: string;
  candidate_id: string;
  netlist_sha: string;
  gates: { gate: string; pass: boolean }[];
  score: number | null;
  adopted: boolean;
  notes?: string;
  files: Record<string, string>;
  verify?: EeVerify[];
};
type EeBoard = { id: string; name: string; created_at: string; versions: EeVersion[] };

type EeProgress = {
  status: string;            // working | done | idle
  board_id?: string;
  candidate?: string;
  base?: string;
  incumbent?: string;
  incumbent_score?: number;
  bar?: number;
  phase?: string;            // authoring | gates | scoring | publish | done
  gates?: { gate: string; pass: boolean | null; failures?: string[] }[];
  score?: number | null;
  outcome?: string;          // adopted | rejected | gate-fail
  note?: string;
  ts?: string;
};

type Diff = {
  from: number; to: number;
  added_comps: { ref: string; value: string }[];
  removed_comps: { ref: string; value: string }[];
  changed_comps: { ref: string; from?: string; to: string }[];
  added_nets: string[];
  removed_nets: string[];
  fp_diff?: {
    fp_added: { ref: string; value: string; x: number; y: number; side: string }[];
    fp_removed: { ref: string; value: string }[];
    fp_moved: { ref: string; from: { x: number; y: number }; to: { x: number; y: number } }[];
    fp_rotated: { ref: string; from: number; to: number }[];
    fp_flipped: { ref: string; from: string; to: string }[];
  } | null;
  layers?: string[];
};

const fileUrl = (b: string, v: number, kind: string) =>
  `/api/ee/boards/${b}/versions/${v}/file?kind=${kind}`;
const verifyUrl = (b: string, v: number, kind: string) =>
  `/api/ee/boards/${b}/versions/${v}/verify?kind=${kind}`;

const ago = (ts?: string) => {
  if (!ts) return "";
  const s = Math.max(0, (Date.now() - new Date(ts).getTime()) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
};

function PanZoom({ children, wrapClass }: { children: React.ReactNode; wrapClass: string }) {
  // wheel zoom around cursor + pointer-drag pan + pinch + double-click reset.
  // Native wheel listener: React's delegated onWheel is passive in practice,
  // and preventDefault is what keeps the page from scrolling while zooming.
  const [t, setT] = useState({ x: 0, y: 0, k: 1 });
  const ref = useRef<HTMLDivElement>(null);
  const tRef = useRef(t);
  tRef.current = t;
  const pts = useRef(new Map<number, { x: number; y: number }>());
  const last = useRef<{ x: number; y: number; d: number } | null>(null);
  const clampK = (k: number) => Math.min(16, Math.max(0.2, k));

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const cx = e.clientX - r.left, cy = e.clientY - r.top;
      const dk = Math.exp(-e.deltaY * 0.0015);
      const t0 = tRef.current;
      const k = clampK(t0.k * dk);
      const s = k / t0.k;
      setT({ k, x: cx - (cx - t0.x) * s, y: cy - (cy - t0.y) * s });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
    pts.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    last.current = null;
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!pts.current.has(e.pointerId)) return;
    pts.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const arr = [...pts.current.values()];
    const t0 = tRef.current;
    if (arr.length === 1) {
      const p = arr[0];
      if (last.current) setT({ ...t0, x: t0.x + p.x - last.current.x, y: t0.y + p.y - last.current.y });
      last.current = { x: p.x, y: p.y, d: 0 };
    } else if (arr.length === 2) {
      const d = Math.hypot(arr[0].x - arr[1].x, arr[0].y - arr[1].y);
      const cx = (arr[0].x + arr[1].x) / 2, cy = (arr[0].y + arr[1].y) / 2;
      const r = ref.current!.getBoundingClientRect();
      const ox = cx - r.left, oy = cy - r.top;
      if (last.current && last.current.d > 0) {
        const k = clampK(t0.k * (d / last.current.d));
        const s = k / t0.k;
        setT({ k, x: ox - (ox - t0.x) * s + cx - last.current.x, y: oy - (oy - t0.y) * s + cy - last.current.y });
      }
      last.current = { x: cx, y: cy, d };
    }
  };
  const onPointerUp = (e: React.PointerEvent) => { pts.current.delete(e.pointerId); last.current = null; };

  return (
    <div ref={ref}
      style={{ backgroundColor: "#ffffff", backgroundImage: "radial-gradient(circle, rgba(0,0,0,0.13) 1px, transparent 1.5px)", backgroundSize: "22px 22px" }}
      className="rounded-md overflow-hidden max-h-[70vh] relative touch-none select-none cursor-grab active:cursor-grabbing p-2"
      onPointerDown={onPointerDown} onPointerMove={onPointerMove}
      onPointerUp={onPointerUp} onPointerCancel={onPointerUp}
      onDoubleClick={() => setT({ x: 0, y: 0, k: 1 })}>
      <div style={{ transform: `translate(${t.x}px, ${t.y}px) scale(${t.k})`, transformOrigin: "0 0" }} className={wrapClass}>
        {children}
      </div>
      {(t.k !== 1 || t.x !== 0 || t.y !== 0) && (
        <button onClick={() => setT({ x: 0, y: 0, k: 1 })}
          className="absolute top-2 right-2 text-xs bg-black/70 text-white rounded px-2 py-1 z-10">reset</button>
      )}
    </div>
  );
}

const LAYER_LABEL: Record<string, string> = {
  pcb_fcu: "F.Cu top copper", pcb_bcu: "B.Cu bottom copper",
  pcb_fsilk: "F.Silkscreen", pcb_fab: "F.Fab footprints",
};

function ArtworkDiff({ board, from, to, layers }: { board: string; from: number; to: number; layers: string[] }) {
  const [layer, setLayer] = useState(layers[0]);
  // black-on-white layer exports: tint v-from red / v-to green and blend with
  // darken - identical copper lands black, red-only = removed, green-only = added
  return (
    <div className="space-y-2">
      <div className="flex gap-1 flex-wrap items-center">
        <span className="text-xs text-muted-foreground mr-1">Artwork diff (red = only in v{from}, green = only in v{to}):</span>
        {layers.map((l) => (
          <Button key={l} size="sm" variant={layer === l ? "default" : "ghost"} onClick={() => setLayer(l)}>
            {LAYER_LABEL[l] || l}
          </Button>
        ))}
      </div>
      <PanZoom wrapClass="min-w-[600px] w-full">
        <div className="relative">
          <img src={fileUrl(board, from, layer)} alt={`v${from} ${layer}`} className="w-full"
            style={{ filter: "invert(27%) sepia(98%) saturate(2000%) hue-rotate(330deg) brightness(95%)" }} />
          <img src={fileUrl(board, to, layer)} alt={`v${to} ${layer}`} className="w-full absolute inset-0"
            style={{ filter: "invert(55%) sepia(90%) saturate(1500%) hue-rotate(75deg) brightness(90%)", mixBlendMode: "darken" }} />
        </div>
      </PanZoom>
    </div>
  );
}

// Live round card: what the EE loop is doing RIGHT NOW - candidate in flight,
// gates as they land, score against the adoption bar when it lands.
function LiveRound({ prog }: { prog: EeProgress | null }) {
  if (!prog || !prog.candidate) {
    return (
      <Card>
        <CardHeader className="p-3 pb-1"><CardTitle className="text-sm">Research loop</CardTitle></CardHeader>
        <CardContent className="p-3 pt-1 text-xs text-muted-foreground">No round state posted yet.</CardContent>
      </Card>
    );
  }
  const working = prog.status === "working";
  const outcomeCls = prog.outcome === "adopted" ? "text-emerald-400" : prog.outcome ? "text-amber-400" : "";
  return (
    <Card className={working ? "border-primary/50" : ""}>
      <CardHeader className="p-3 pb-1">
        <CardTitle className="text-sm flex items-center gap-2">
          {working && (
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-60" />
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-primary" />
            </span>
          )}
          {working ? "Round in flight" : "Last round"}
          <span className="font-mono font-normal text-xs text-muted-foreground">{prog.candidate}</span>
        </CardTitle>
        <CardDescription className="text-xs">
          {prog.phase ?? prog.status}{prog.ts ? ` - ${ago(prog.ts)}` : ""}
          {prog.outcome && <span className={`ml-1 font-medium ${outcomeCls}`}>{prog.outcome}</span>}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-3 pt-1 space-y-2 text-xs">
        {(prog.incumbent || prog.bar != null) && (
          <div className="text-muted-foreground">
            incumbent <span className="font-mono text-foreground">{prog.incumbent ?? "-"}</span>
            {prog.incumbent_score != null && <span className="font-mono"> {prog.incumbent_score.toFixed(1)}</span>}
            {prog.bar != null && <> - adoption bar <span className="font-mono text-foreground">{prog.bar.toFixed(1)}</span></>}
          </div>
        )}
        {prog.base && prog.base !== prog.incumbent && (
          <div className="text-muted-foreground">authoring base <span className="font-mono text-foreground">{prog.base}</span></div>
        )}
        {!!prog.gates?.length && (
          <ul className="space-y-1">
            {prog.gates.map((g) => (
              <li key={g.gate} className="flex items-start gap-1.5">
                {g.pass === null || g.pass === undefined
                  ? <Loader2 className="h-3.5 w-3.5 mt-px animate-spin text-muted-foreground" />
                  : g.pass ? <CheckCircle2 className="h-3.5 w-3.5 mt-px text-green-500" />
                           : <XCircle className="h-3.5 w-3.5 mt-px text-red-500" />}
                <span>
                  {g.gate}
                  {!!g.failures?.length && <span className="block text-red-400/80 font-mono text-[11px]">{g.failures[0]}</span>}
                </span>
              </li>
            ))}
          </ul>
        )}
        {prog.score != null && (
          <div>
            score <span className={`font-mono ${prog.bar != null && prog.score > prog.bar ? "text-emerald-400" : ""}`}>{prog.score.toFixed(1)}</span>
            {prog.bar != null && <span className="text-muted-foreground font-mono"> / {prog.bar.toFixed(1)}</span>}
          </div>
        )}
        {prog.note && <div className="text-muted-foreground">{prog.note}</div>}
      </CardContent>
    </Card>
  );
}

export default function Ee() {
  const boards = useQuery({
    queryKey: ["ee-boards"],
    queryFn: async () => (await (await fetch("/api/ee/boards", { credentials: "same-origin" })).json()).boards as EeBoard[],
    refetchInterval: 30_000,
  });
  const progQ = useQuery({
    queryKey: ["ee-progress"],
    queryFn: async () => (await (await fetch("/api/ee/progress", { credentials: "same-origin" })).json()) as { current: EeProgress | null; rounds: EeProgress[] },
    refetchInterval: 10_000,
  });
  const [boardId, setBoardId] = useState<string | null>(null);
  const [ver, setVer] = useState<number | null>(null);
  const [tab, setTab] = useState<"sch" | "pcb" | "3d" | "diff" | "tests">("sch");
  const [diffFrom, setDiffFrom] = useState<number | null>(null);

  const board = (boards.data || []).find((b) => b.id === boardId) || (boards.data || [])[0] || null;
  const version = board?.versions.find((v) => v.version === ver)
    || board?.versions[board.versions.length - 1] || null;
  const prev = board && version
    ? [...board.versions].reverse().find((v) => v.version < version.version) || null
    : null;
  const dFrom = diffFrom ?? prev?.version ?? null;

  const diff = useQuery({
    queryKey: ["ee-diff", board?.id, dFrom, version?.version],
    enabled: !!board && !!version && dFrom !== null && dFrom !== version.version,
    queryFn: async () =>
      (await (await fetch(`/api/ee/boards/${board!.id}/diff?from=${dFrom}&to=${version!.version}`,
        { credentials: "same-origin" })).json()) as Diff,
  });

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-xl md:text-2xl font-bold tracking-tight flex items-center gap-2">
          <CircuitBoard className="h-5 w-5 text-primary" /> EE
        </h1>
        <p className="text-xs md:text-sm text-muted-foreground mt-1">versioned board designs, live research rounds, verification tests</p>
      </header>

      {boards.data && boards.data.length === 0 && (
        <Card><CardContent className="p-6 text-sm text-muted-foreground">
          No board designs published yet. The EE researcher loop on box #3 posts
          adopted candidates here (versioned schematic + layout, pinned by netlist).
        </CardContent></Card>
      )}

      <div className="md:grid md:grid-cols-[300px_minmax(0,1fr)] md:gap-4 space-y-4 md:space-y-0">
        {/* sidebar */}
        <div className="space-y-3 md:h-full md:min-h-0 md:flex md:flex-col">
          <LiveRound prog={progQ.data?.current ?? null} />
          {boards.data && boards.data.length > 0 && (
            <Card className="md:flex-1 md:min-h-0 md:flex md:flex-col">
              <CardHeader className="p-3 pb-2 shrink-0">
                <CardTitle className="text-sm">Boards & versions</CardTitle>
              </CardHeader>
              <CardContent className="p-2 pt-0 md:flex-1 md:min-h-0 md:flex md:flex-col">
                <div className="max-h-[280px] md:max-h-none md:flex-1 overflow-y-auto space-y-1.5 pr-1">
                  {boards.data.map((b) => (
                    <div key={b.id}>
                      <button onClick={() => { setBoardId(b.id); setVer(null); setDiffFrom(null); }}
                        className={`w-full text-left rounded-lg border p-2.5 ${b.id === board?.id ? "border-primary bg-primary/5" : "border-border"}`}>
                        <div className="text-xs font-medium">{b.name}</div>
                        <div className="text-[11px] text-muted-foreground">{b.versions.length} version{b.versions.length === 1 ? "" : "s"}</div>
                      </button>
                      {b.id === board?.id && (
                        <div className="flex gap-1 flex-wrap mt-1.5 px-1">
                          {[...b.versions].reverse().map((v) => (
                            <button key={v.version} onClick={() => { setVer(v.version); setDiffFrom(null); }}
                              className={`font-mono text-[11px] px-1.5 py-0.5 rounded border ${
                                v.version === version?.version ? "border-primary text-primary" : "border-border text-muted-foreground hover:text-foreground"}`}>
                              v{v.version}{v.adopted && " *"}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* main board view */}
        {board && version && (
          <Card className="min-w-0">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-3 flex-wrap">
                <span>{board.name} - v{version.version}</span>
                {version.adopted && <Star className="h-4 w-4 text-yellow-400" />}
                {(version.gates ?? []).map((g) => (
                  <span key={g.gate} className="flex items-center gap-1 text-xs font-normal">
                    {g.pass
                      ? <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                      : <XCircle className="h-3.5 w-3.5 text-red-500" />}
                    {g.gate}
                  </span>
                ))}
                {version.score !== null && <span className="text-xs font-normal">score {version.score}</span>}
              </CardTitle>
              <p className="text-xs text-muted-foreground">
                {version.candidate_id} - {new Date(version.created_at).toLocaleString()} -
                netlist {version.netlist_sha.slice(0, 12)}
              </p>
              {version.notes && <p className="text-xs mt-1">{version.notes}</p>}
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-1 items-center flex-wrap">
                {(["sch", "pcb", "3d", "diff", "tests"] as const).map((t) => (
                  <Button key={t} size="sm" variant={tab === t ? "default" : "ghost"}
                    onClick={() => setTab(t)}
                    disabled={(t === "sch" && !version.files.sch_svg) || (t === "pcb" && !version.files.pcb_svg) || (t === "3d" && !version.files.glb) || (t === "diff" && !prev) || (t === "tests" && !version.verify?.length)}>
                    {t === "sch" ? "Schematic" : t === "pcb" ? "Layout" : t === "3d" ? "3D" : t === "diff" ? "Diff" : `Tests${version.verify?.length ? ` (${version.verify.length})` : ""}`}
                  </Button>
                ))}
                {tab === "diff" && prev && (
                  <select className="ml-2 bg-background border rounded px-2 py-1 text-xs"
                    value={dFrom ?? ""} onChange={(e) => setDiffFrom(Number(e.target.value))}>
                    {board.versions.filter((v) => v.version < version.version).map((v) => (
                      <option key={v.version} value={v.version}>from v{v.version}</option>
                    ))}
                  </select>
                )}
              </div>
              {tab === "sch" && version.files.sch_svg && (
                <PanZoom wrapClass="min-w-[800px] w-full">
                  <img src={fileUrl(board.id, version.version, "sch_svg")} alt="schematic" className="w-full" draggable={false} />
                </PanZoom>
              )}
              {tab === "pcb" && version.files.pcb_svg && (
                <PanZoom wrapClass="min-w-[600px] w-full">
                  <img src={fileUrl(board.id, version.version, "pcb_svg")} alt="layout" className="w-full" draggable={false} />
                </PanZoom>
              )}
              {tab === "3d" && version.files.glb && (
                <div className="h-[60vh]"><CadViewer url={fileUrl(board.id, version.version, "glb")} /></div>
              )}
              {tab === "tests" && (
                <div className="space-y-3">
                  {!version.verify?.length && (
                    <p className="text-xs text-muted-foreground">
                      No verification artifacts on this version yet. SI/PI runs (S-parameter plots, impedance,
                      field overlays, power-tree drops) attach here as the verification pipeline posts them.
                    </p>
                  )}
                  <div className="grid gap-3 lg:grid-cols-2">
                    {(version.verify ?? []).map((vf) => (
                      <figure key={vf.kind} className="space-y-1 min-w-0">
                        <figcaption className="text-xs text-muted-foreground">{vf.label || vf.kind}</figcaption>
                        <PanZoom wrapClass="w-full">
                          <img src={verifyUrl(board.id, version.version, vf.kind)} alt={vf.label || vf.kind} className="w-full rounded" draggable={false} />
                        </PanZoom>
                      </figure>
                    ))}
                  </div>
                </div>
              )}
              {tab === "diff" && diff.data && (
                <div className="grid md:grid-cols-2 gap-3 text-xs">
                  <DiffList title={`Components added (${(diff.data.added_comps ?? []).length})`}
                    rows={(diff.data.added_comps ?? []).map((c) => `${c.ref} ${c.value}`)} tone="text-green-500" />
                  <DiffList title={`Components removed (${(diff.data.removed_comps ?? []).length})`}
                    rows={(diff.data.removed_comps ?? []).map((c) => `${c.ref} ${c.value}`)} tone="text-red-500" />
                  <DiffList title={`Components changed (${(diff.data.changed_comps ?? []).length})`}
                    rows={(diff.data.changed_comps ?? []).map((c) => `${c.ref}: ${c.from} -> ${c.to}`)} tone="text-yellow-500" />
                  <DiffList title={`Nets added (${(diff.data.added_nets ?? []).length})`} rows={diff.data.added_nets ?? []} tone="text-green-500" />
                  <DiffList title={`Nets removed (${(diff.data.removed_nets ?? []).length})`} rows={diff.data.removed_nets ?? []} tone="text-red-500" />
                </div>
              )}
              {tab === "diff" && diff.data && (diff.data.layers?.length ?? 0) > 0 && dFrom !== null && (
                <ArtworkDiff board={board.id} from={dFrom} to={version.version} layers={diff.data.layers!} />
              )}
              {tab === "diff" && diff.data?.fp_diff && (() => {
                const fd = diff.data.fp_diff!;
                const added = fd.fp_added ?? [], removed = fd.fp_removed ?? [];
                const moved = fd.fp_moved ?? [], rotated = fd.fp_rotated ?? [], flipped = fd.fp_flipped ?? [];
                return (
                <div className="grid md:grid-cols-2 gap-3 text-xs border-t pt-3">
                  <DiffList title={`Footprints added (${added.length})`}
                    rows={added.map((f) => `${f.ref} ${f.value} @ ${f.x},${f.y} ${f.side}`)} tone="text-green-500" />
                  <DiffList title={`Footprints removed (${removed.length})`}
                    rows={removed.map((f) => `${f.ref} ${f.value}`)} tone="text-red-500" />
                  <DiffList title={`Moved (${moved.length})`}
                    rows={moved.map((f) => `${f.ref}: ${f.from.x},${f.from.y} -> ${f.to.x},${f.to.y}`)} tone="text-yellow-500" />
                  <DiffList title={`Rotated / flipped (${rotated.length + flipped.length})`}
                    rows={[...rotated.map((f) => `${f.ref}: rot ${f.from} -> ${f.to}`),
                           ...flipped.map((f) => `${f.ref}: ${f.from} -> ${f.to} side`)]} tone="text-yellow-500" />
                </div>
                );
              })()}
              {tab === "diff" && diff.data && (diff.data.added_comps ?? []).length + (diff.data.removed_comps ?? []).length + (diff.data.changed_comps ?? []).length + (diff.data.added_nets ?? []).length + (diff.data.removed_nets ?? []).length === 0 && (
                <p className="text-xs text-muted-foreground">No structural netlist changes between these versions.</p>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

function DiffList({ title, rows, tone }: { title: string; rows: string[]; tone: string }) {
  return (
    <div>
      <p className={`font-medium ${tone}`}>{title}</p>
      <ul className="mt-1 space-y-0.5 text-muted-foreground">
        {rows.slice(0, 50).map((r, i) => <li key={i} className="font-mono">{r}</li>)}
        {rows.length > 50 && <li>... {rows.length - 50} more</li>}
        {rows.length === 0 && <li className="italic">none</li>}
      </ul>
    </div>
  );
}
