import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { fetchCadDesigns, fetchState, type CadDesign, type CadSnapshotRecord } from "@/api";
import { ChevronRight } from "lucide-react";
import CadViewer from "@/components/CadViewer";

function lineageOf(d: CadDesign, all: CadDesign[]): CadDesign[] {
  const chain: CadDesign[] = [];
  let cur: CadDesign | undefined = d;
  const seen = new Set<string>();
  while (cur && !seen.has(cur.id)) {
    seen.add(cur.id);
    chain.unshift(cur);
    cur = cur.parent_id ? all.find((x) => x.id === cur!.parent_id) : undefined;
  }
  return chain;
}

const fmt = (v: unknown): string => {
  if (typeof v === "number") return Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.001) ? v.toExponential(3) : String(Math.round(v * 10000) / 10000);
  if (typeof v === "boolean") return String(v);
  if (Array.isArray(v)) return v.length > 4 ? `[${v.slice(0, 4).join(",")} +${v.length - 4}]` : `[${v.join(",")}]`;
  if (v && typeof v === "object") return "{...}";
  const s = String(v);
  return s.length > 42 ? s.slice(0, 42) + "..." : s;
};

function KV({ k, v }: { k: string; v: unknown }) {
  return (
    <div className="flex justify-between gap-2 min-w-0">
      <dt className="text-muted-foreground truncate shrink" title={k}>{k.replace(/_/g, " ")}</dt>
      <dd className="font-mono text-right break-all max-w-[58%]" title={typeof v === "object" ? JSON.stringify(v) : String(v)}>{fmt(v)}</dd>
    </div>
  );
}

// One titled property group. Nested objects render as labeled sub-groups so
// FEA load cases (hover_max, crash, ...) read as structured results, not a
// flattened data dump.
function PropGroup({ title, obj }: { title: string; obj: Record<string, unknown> | undefined }) {
  if (!obj || !Object.keys(obj).length) return null;
  const scalars = Object.entries(obj).filter(([, v]) => !v || typeof v !== "object" || Array.isArray(v));
  const nested = Object.entries(obj).filter(([, v]) => v && typeof v === "object" && !Array.isArray(v));
  return (
    <div className="min-w-0 rounded-lg border border-border p-3">
      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">{title}</div>
      {scalars.length > 0 && (
        <dl className="grid grid-cols-1 gap-y-1 text-xs md:text-sm">
          {scalars.map(([k, v]) => <KV key={k} k={k} v={v} />)}
        </dl>
      )}
      {nested.map(([k, v]) => (
        <div key={k} className="mt-2">
          <div className="text-[11px] font-medium text-foreground/80 mb-1">{k.replace(/_/g, " ")}</div>
          <dl className="grid grid-cols-1 gap-y-1 text-xs md:text-sm pl-2 border-l-2 border-border">
            {Object.entries(v as Record<string, unknown>).map(([sk, sv]) => <KV key={sk} k={sk} v={sv} />)}
          </dl>
        </div>
      ))}
    </div>
  );
}

export default function Cad() {
  const { data, error } = useQuery({ queryKey: ["cad"], queryFn: fetchCadDesigns, refetchInterval: 15_000 });
  const stateQ = useQuery({ queryKey: ["state"], queryFn: fetchState, refetchInterval: 15_000 });
  const progQ = useQuery({
    queryKey: ["cad-progress"],
    queryFn: async () => (await (await fetch("/api/cad/progress", { credentials: "same-origin" })).json()) as any,
    refetchInterval: 10_000,
  });
  const prog = progQ.data as any;
  const progAge = prog?.ts ? (Date.now() - new Date(prog.ts).getTime()) / 1000 : 1e9;
  const progLive = prog && prog.status === "working" && progAge < 120;
  // Merge GLB-bearing designs with geometry-pending snapshot records
  // (kind="cad.chassis.snapshot" via /api/ingest) by id. GLB design wins;
  // snapshot fills missing metrics.
  const designs = (() => {
    const byId = new Map<string, CadDesign & { glb_pending_path?: string }>();
    const snaps = ((stateQ.data?.records ?? []) as any[]).filter((r) => r && r.kind === "cad.chassis.snapshot") as CadSnapshotRecord[];
    for (const s of snaps) {
      byId.set(s.id, {
        id: s.id,
        name: s.name ?? s.id,
        parent_id: s.parent_id ?? null,
        created_at: (s.ts as string) ?? new Date().toISOString(),
        metrics: (s.metrics as any) ?? {},
        notes: s.notes,
        glb_bytes: 0,
        glb_url: "",
        glb_pending_path: s.glb_path,
      });
    }
    for (const d of (data?.designs ?? [])) {
      const prev = byId.get(d.id);
      byId.set(d.id, { ...d, metrics: { ...(prev?.metrics ?? {}), ...d.metrics } });
    }
    return [...byId.values()].sort((a, b) => a.created_at.localeCompare(b.created_at));
  })();
  const [selId, setSelId] = useState<string | null>(null);
  const sel = designs.find((d) => d.id === selId) ?? designs[designs.length - 1] ?? null;

  if (error) return <div className="p-8 text-red-400">Failed to load: {(error as Error).message}</div>;
  const chain = sel ? lineageOf(sel, designs) : [];
  const children = sel ? designs.filter((d) => d.parent_id === sel.id) : [];
  const otherMetrics = sel ? Object.fromEntries(
    Object.entries(sel.metrics).filter(([k]) => !["mass_g", "inertia", "printability", "fea"].includes(k))) : {};

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-xl md:text-2xl font-bold tracking-tight">CAD</h1>
        <p className="text-xs md:text-sm text-muted-foreground mt-1">chassis designs from the CAD researcher - auto-refresh 15s</p>
      </header>

      <div className="md:grid md:grid-cols-[300px_minmax(0,1fr)] md:gap-4 md:items-start space-y-4 md:space-y-0">
        {/* sidebar: live status + designs list */}
        <div className="space-y-3 md:sticky md:top-6">
          {progLive && (
            <Card className="border-primary/50">
              <CardContent className="p-3 flex items-center gap-3">
                <span className="relative flex h-3 w-3 shrink-0">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-60"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-primary"></span>
                </span>
                <div className="text-xs md:text-sm min-w-0">
                  <span className="font-semibold">Working on {String(prog.design_id || "next revision")}</span>
                  {prog.stage && <span className="text-muted-foreground"> - {String(prog.stage)}</span>}
                  {prog.detail && <div className="text-muted-foreground truncate">{String(prog.detail)}</div>}
                </div>
              </CardContent>
            </Card>
          )}
          <Card>
            <CardHeader className="p-3 pb-2">
              <CardTitle className="text-sm">Designs</CardTitle>
              <CardDescription className="text-xs">{designs.length} design{designs.length === 1 ? "" : "s"}, newest last</CardDescription>
            </CardHeader>
            <CardContent className="p-2 pt-0">
              <div className="max-h-[320px] md:max-h-[calc(100dvh-16rem)] overflow-y-auto space-y-1.5 pr-1">
                {!designs.length && (
                  <div className="p-3 text-xs text-muted-foreground">
                    No chassis designs yet. The CAD researcher pushes them to <code className="font-mono">POST /api/cad/designs</code> as they land.
                  </div>
                )}
                {[...designs].reverse().map((d) => (
                  <button key={d.id} onClick={() => setSelId(d.id)}
                    className={`w-full text-left rounded-lg border p-2.5 ${sel?.id === d.id ? "border-primary bg-primary/5" : "border-border"}`}>
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-xs">{d.id}</span>
                      <span className="text-[11px] text-muted-foreground shrink-0">
                        {d.metrics.mass_g != null ? `${Number(d.metrics.mass_g).toFixed(0)} g` : d.glb_url ? `${(d.glb_bytes / 1024).toFixed(0)} KB` : "pending"}
                      </span>
                    </div>
                    <div className="text-[11px] text-muted-foreground mt-0.5">
                      {d.name && d.name !== d.id ? `${d.name} - ` : ""}{d.parent_id ? `from ${d.parent_id}` : "root"} - {new Date(d.created_at).toLocaleDateString()}
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* main: chassis viewer + properties */}
        {sel && (
          <Card className="min-w-0">
            <CardHeader className="p-3 md:p-6">
              <CardTitle className="text-base md:text-lg">{sel.name || sel.id}</CardTitle>
              <CardDescription className="text-xs md:text-sm">
                {sel.id} - {new Date(sel.created_at).toLocaleString()} - {(sel.glb_bytes / 1024).toFixed(0)} KB
                {sel.source ? ` - ${sel.source}` : ""}
              </CardDescription>
            </CardHeader>
            <CardContent className="p-3 md:p-6 pt-0 md:pt-0 space-y-4">
              <div className="h-72 md:h-[52vh]">
                {sel.glb_url ? (
                  <CadViewer key={sel.id} url={sel.glb_url} />
                ) : (
                  <div className="h-full grid place-items-center rounded-lg border border-dashed border-border text-center p-4">
                    <div className="text-sm text-muted-foreground">
                      Geometry pending - snapshot received, GLB not uploaded yet.
                      {(sel as any).glb_pending_path && (
                        <div className="font-mono text-[11px] mt-2 break-all">on box: {(sel as any).glb_pending_path}</div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* lineage */}
              <div>
                <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">lineage</div>
                <div className="flex flex-wrap items-center gap-1">
                  {chain.map((d, i) => (
                    <span key={d.id} className="flex items-center gap-1">
                      {i > 0 && <ChevronRight className="h-3 w-3 text-muted-foreground" />}
                      <button onClick={() => setSelId(d.id)}
                        className={`font-mono text-xs px-2 py-1 rounded border ${d.id === sel.id ? "border-primary text-primary" : "border-border text-muted-foreground hover:text-foreground"}`}>
                        {d.id}
                      </button>
                    </span>
                  ))}
                  {children.length > 0 && (
                    <span className="flex items-center gap-1">
                      <ChevronRight className="h-3 w-3 text-muted-foreground" />
                      {children.map((ch) => (
                        <button key={ch.id} onClick={() => setSelId(ch.id)}
                          className="font-mono text-xs px-2 py-1 rounded border border-emerald-400/40 text-emerald-400 hover:bg-emerald-400/10">
                          {ch.id}
                        </button>
                      ))}
                    </span>
                  )}
                </div>
              </div>

              {/* properties, grouped and labeled */}
              <div className="grid gap-3 md:grid-cols-2">
                {(sel.metrics.mass_g != null || sel.metrics.inertia) && (
                  <PropGroup title="mass properties" obj={{
                    ...(sel.metrics.mass_g != null ? { mass: `${Number(sel.metrics.mass_g).toFixed(1)} g` } : {}),
                    ...(sel.metrics.inertia ? { inertia: sel.metrics.inertia } : {}),
                  }} />
                )}
                <PropGroup title="printability" obj={sel.metrics.printability as Record<string, unknown> | undefined} />
                <PropGroup title="FEA" obj={sel.metrics.fea as Record<string, unknown> | undefined} />
                {Object.keys(otherMetrics).length > 0 && <PropGroup title="other metrics" obj={otherMetrics} />}
              </div>
              {sel.notes && <p className="text-xs md:text-sm text-muted-foreground">{sel.notes}</p>}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
