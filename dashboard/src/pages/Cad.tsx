import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, Link } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { fetchCadDesigns, fetchState, logout, type CadDesign, type CadSnapshotRecord } from "@/api";
import { Box, LogOut, FlaskConical, ChevronRight } from "lucide-react";
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

function MetricGrid({ title, obj }: { title: string; obj: Record<string, unknown> | undefined }) {
  if (!obj || !Object.keys(obj).length) return null;
  // Flatten one level of nested objects (FEA hover_max/crash blocks) and
  // format values compactly - long arrays/strings must never blow out
  // mobile width.
  const rows: [string, unknown][] = [];
  for (const [k, v] of Object.entries(obj)) {
    if (v && typeof v === "object" && !Array.isArray(v)) {
      for (const [sk, sv] of Object.entries(v as Record<string, unknown>)) rows.push([`${k} ${sk}`, sv]);
    } else {
      rows.push([k, v]);
    }
  }
  const fmt = (v: unknown): string => {
    if (typeof v === "number") return Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.001) ? v.toExponential(3) : String(Math.round(v * 10000) / 10000);
    if (typeof v === "boolean") return String(v);
    if (Array.isArray(v)) return v.length > 4 ? `[${v.slice(0, 4).join(",")} +${v.length - 4}]` : `[${v.join(",")}]`;
    if (v && typeof v === "object") return "{...}";
    const s = String(v);
    return s.length > 42 ? s.slice(0, 42) + "…" : s;
  };
  return (
    <div className="min-w-0">
      <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">{title}</div>
      <dl className="grid grid-cols-1 gap-y-1 text-xs md:text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2 min-w-0">
            <dt className="text-muted-foreground truncate shrink" title={k}>{k.replace(/_/g, " ")}</dt>
            <dd className="font-mono text-right break-all max-w-[58%]" title={typeof v === "object" ? JSON.stringify(v) : String(v)}>{fmt(v)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export default function Cad() {
  const nav = useNavigate();
  const qc = useQueryClient();
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

  return (
    <div className="min-h-screen p-3 md:p-8 max-w-6xl mx-auto space-y-4 md:space-y-6">
      <header className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h1 className="text-base md:text-2xl font-bold tracking-tight flex items-center gap-2">
            <Box className="h-5 w-5 md:h-6 md:w-6 text-primary shrink-0" />
            <span>DroneStudio - CAD / Mechanicals</span>
          </h1>
          <p className="text-xs md:text-sm text-muted-foreground mt-1">
            Chassis designs from the CAD researcher - auto-refresh 15s
          </p>
        </div>
        <div className="flex items-center gap-1 shrink-0 mt-1">
          <Link to="/">
            <Button variant="ghost" size="sm"><FlaskConical className="h-4 w-4" /> <span className="hidden sm:inline">Research</span></Button>
          </Link>
          <Button variant="ghost" size="sm"
            onClick={async () => { await logout(); await qc.invalidateQueries({ queryKey: ["me"] }); nav("/login"); }}>
            <LogOut className="h-4 w-4" /> <span className="hidden sm:inline">Sign out</span>
          </Button>
        </div>
      </header>

      {progLive && (
        <Card className="border-primary/50">
          <CardContent className="p-3 md:p-4 flex items-center gap-3">
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

      {!designs.length && (
        <Card><CardContent className="p-6 text-sm text-muted-foreground">
          No chassis designs yet. The CAD researcher pushes them to <code className="font-mono text-xs">POST /api/cad/designs</code> as they land.
        </CardContent></Card>
      )}

      {sel && (
        <Card>
          <CardHeader className="p-3 md:p-6">
            <CardTitle className="text-base md:text-lg">{sel.name || sel.id}</CardTitle>
            <CardDescription className="text-xs md:text-sm">
              {sel.id} - {new Date(sel.created_at).toLocaleString()} - {(sel.glb_bytes / 1024).toFixed(0)} KB
              {sel.source ? ` - ${sel.source}` : ""}
            </CardDescription>
          </CardHeader>
          <CardContent className="p-3 md:p-6 pt-0 md:pt-0 space-y-4">
            <div className="h-72 md:h-96">
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

            {/* metrics */}
            <div className="grid gap-4 md:grid-cols-2">
              {(sel.metrics.mass_g != null || sel.metrics.inertia) && (
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">mass properties</div>
                  <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs md:text-sm">
                    {sel.metrics.mass_g != null && (
                      <div className="flex justify-between gap-2">
                        <dt className="text-muted-foreground">mass</dt>
                        <dd className="font-mono">{Number(sel.metrics.mass_g).toFixed(1)} g</dd>
                      </div>
                    )}
                    {sel.metrics.inertia && Object.entries(sel.metrics.inertia).map(([k, v]) => (
                      <div key={k} className="flex justify-between gap-2">
                        <dt className="text-muted-foreground">{k}</dt>
                        <dd className="font-mono">{typeof v === "number" ? v.toExponential(3) : String(v)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              )}
              <MetricGrid title="printability" obj={sel.metrics.printability as Record<string, unknown> | undefined} />
              <MetricGrid title="FEA" obj={sel.metrics.fea as Record<string, unknown> | undefined} />
              {Object.entries(sel.metrics)
                .filter(([k]) => !["mass_g", "inertia", "printability", "fea"].includes(k))
                .length > 0 && (
                <MetricGrid title="other metrics" obj={Object.fromEntries(
                  Object.entries(sel.metrics).filter(([k]) => !["mass_g", "inertia", "printability", "fea"].includes(k)))} />
              )}
            </div>
            {sel.notes && <p className="text-xs md:text-sm text-muted-foreground">{sel.notes}</p>}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="p-3 md:p-6"><CardTitle className="text-base md:text-lg">All designs</CardTitle>
          <CardDescription className="text-xs md:text-sm">{designs.length} design{designs.length === 1 ? "" : "s"}, newest last</CardDescription></CardHeader>
        <CardContent className="p-3 md:p-6 pt-0 md:pt-0 space-y-2">
          {designs.map((d) => (
            <button key={d.id} onClick={() => setSelId(d.id)}
              className={`w-full text-left rounded-lg border p-3 ${sel?.id === d.id ? "border-primary bg-primary/5" : "border-border"}`}>
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-sm">{d.id}</span>
                <span className="text-xs text-muted-foreground">
                  {d.glb_url ? `${(d.glb_bytes / 1024).toFixed(0)} KB` : "no GLB yet"}
                  {d.metrics.mass_g != null ? ` - ${Number(d.metrics.mass_g).toFixed(0)} g` : ""}
                </span>
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {d.name && d.name !== d.id ? `${d.name} - ` : ""}{d.parent_id ? `derives from ${d.parent_id} - ` : "root design - "}
                {new Date(d.created_at).toLocaleDateString()}
              </div>
            </button>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
