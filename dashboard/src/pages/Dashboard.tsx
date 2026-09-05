import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { fetchState, type ArchiveRecord } from "@/api";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

const MUTATOR_META: Record<string, { label: string; cls: string }> = {
  codex: { label: "ChatGPT", cls: "text-emerald-400" },
  llm: { label: "LLM API", cls: "text-sky-400" },
  heuristic: { label: "heuristic", cls: "text-amber-400" },
  restart: { label: "restart", cls: "text-fuchsia-400" },
  none: { label: "base", cls: "text-muted-foreground" },
};

function MutTag({ m }: { m: string }) {
  const meta = MUTATOR_META[m] ?? { label: m, cls: "text-muted-foreground" };
  return <span className={`text-xs font-mono ${meta.cls}`}>{meta.label}</span>;
}

function bestOf(records: ArchiveRecord[]): ArchiveRecord | null {
  if (!records.length) return null;
  return [...records].sort((a, b) =>
    (b.metrics.success_rate - a.metrics.success_rate) ||
    ((b.metrics.mean_return ?? -1e9) - (a.metrics.mean_return ?? -1e9)))[0];
}


function ScenarioBar({ label, v }: { label: string; v: number | null }) {
  if (v == null) return null;
  const pct = Math.round(v * 100);
  const cls = v >= 0.9 ? "bg-emerald-500" : v >= 0.5 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-24 shrink-0 text-muted-foreground">{label}</span>
      <div className="flex-1 h-2 rounded bg-muted/40 overflow-hidden">
        <div className={`h-2 rounded ${cls}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-10 shrink-0 text-right font-mono">{pct}%</span>
    </div>
  );
}

export default function Dashboard() {
  const { data, error } = useQuery({ queryKey: ["state"], queryFn: fetchState, refetchInterval: 10_000 });
  const currQ = useQuery({
    queryKey: ["curriculum-progress"],
    queryFn: async () => (await (await fetch("/api/curriculum/progress", { credentials: "same-origin" })).json()) as any,
    refetchInterval: 10_000,
  });
  const curr = currQ.data as any;
  const seriesQ = useQuery({
    queryKey: ["series"],
    queryFn: async () => (await (await fetch("/api/series", { credentials: "same-origin" })).json()) as any,
    refetchInterval: 10_000,
  });
  const series = (seriesQ.data ?? {}) as Record<string, { t: string; y: number; label?: string }[]>;
  const tstatQ = useQuery({
    queryKey: ["training-status"],
    queryFn: async () => (await (await fetch("/api/training/status", { credentials: "same-origin" })).json()) as any,
    refetchInterval: 10_000,
  });
  const tstat = (tstatQ.data ?? {}) as any;
  const latestSeries = (name: string): number | null => {
    const arr = series[name];
    return arr && arr.length ? arr[arr.length - 1].y : null;
  };
  const [chartMode, setChartMode] = useState<"generations" | "curriculum" | "loss">("curriculum");
  const [range, setRange] = useState<number | "all">("all");
  const streamQ = useQuery({
    queryKey: ["stream-state"],
    queryFn: async () => (await (await fetch("/api/stream/state", { credentials: "same-origin" })).json()) as any,
    refetchInterval: 5_000,
  });
  const stream = streamQ.data as any;
  const streamAge = stream?.last_event_at ? (Date.now() - new Date(stream.last_event_at).getTime()) / 1000 : 1e9;
  const streamLive = streamAge < 30 && stream?.meta?.status && stream.meta.status !== "offline";

  if (error) return <div className="p-8 text-red-400">Failed to load state: {(error as Error).message}</div>;
  const records = (data?.records ?? []).filter((r: any) =>
    r && r.metrics && typeof r.metrics.success_rate === "number" &&
    (r.kind === undefined || r.kind === null || r.kind === "variant"));
  const run = data?.run ?? null;
  const best = bestOf(records);

  const perGen = new Map<number, number>();
  for (const r of records) {
    perGen.set(r.generation, Math.max(perGen.get(r.generation) ?? -1, r.metrics.success_rate));
  }
  const chart = [...perGen.entries()].sort((a, b) => a[0] - b[0])
    .map(([g, s]) => ({ gen: `g${g}`, success: +(s * 100).toFixed(1) }));

  const toSeries = (name: string, pct: boolean) =>
    (series[name] ?? []).map((p, i) => ({
      x: p.label ?? String(i),
      y: pct ? +(p.y * 100).toFixed(1) : +p.y.toFixed(4),
    }));
  const curriculumChart = toSeries("curriculum_success", true);
  const lossChart = toSeries("training_loss", false);
  const campaignData = (() => {
    const g = series["success_goto_t0"] ?? [];
    const h = series["success_hover_hold_t0"] ?? [];
    const l = series["success_land_t0"] ?? [];
    return g.map((p, i) => ({
      x: p.label ?? String(i),
      goto: +(p.y * 100).toFixed(1),
      hover: h[i] ? +(h[i].y * 100).toFixed(1) : null,
      land: l[i] ? +(l[i].y * 100).toFixed(1) : null,
    }));
  })();
  const MODE_META = {
    generations: { title: "Best success rate per generation", desc: "Selection trajectory across the run" },
    curriculum: { title: "Curriculum hill climbing", desc: "Held-out success as the ladder + DAgger advance (live)" },
    loss: { title: "Training loss", desc: "BC / DAgger regression loss over training (live)" },
  } as const;
  const activeChart =
    chartMode === "generations" ? chart : chartMode === "curriculum" ? curriculumChart : lossChart;
  const activeXKey = chartMode === "generations" ? "gen" : "x";
  const activeYKey = chartMode === "generations" || chartMode === "curriculum" ? "success" : "y";
  const activeData = chartMode === "generations" ? chart.map(c => ({ gen: c.gen, success: c.success }))
    : activeChart.map(c => ({ x: (c as any).x, success: chartMode === "curriculum" ? (c as any).y : undefined, y: chartMode === "loss" ? (c as any).y : undefined }));

  const updated = data?.updated_at ? new Date(data.updated_at).toLocaleTimeString() : "never";
  const recent = [...records].reverse().slice(0, 60);

  return (
    <div className="space-y-4 md:space-y-6">
      <Card>
        <CardHeader className="p-3 md:p-6 pb-2 md:pb-3">
          <div className="flex items-start justify-between gap-2 flex-wrap">
            <div>
              <CardTitle className="text-base md:text-lg">Model status</CardTitle>
              <CardDescription className="text-xs md:text-sm">
                What is flying, what is trained, what is running - {records.length} archive variants over {perGen.size} generation{perGen.size === 1 ? "" : "s"}
              </CardDescription>
            </div>
            {tstat.updated_at && (
              <span className="text-[11px] text-muted-foreground shrink-0">
                status posted {new Date(tstat.updated_at).toLocaleString()}
              </span>
            )}
          </div>
        </CardHeader>
        <CardContent className="p-3 md:p-6 pt-0 md:pt-0 grid gap-4 md:grid-cols-3">
          <div className="space-y-1.5">
            <div className="text-[11px] uppercase tracking-wide text-muted-foreground flex items-center gap-2">
              Live policy
              <span className={`inline-block w-2 h-2 rounded-full ${streamLive ? "bg-emerald-500" : "bg-gray-600"}`} />
              <span className="normal-case">{streamLive ? "flying on /watch" : "watch channel off"}</span>
            </div>
            <div className="font-mono text-base md:text-lg">{tstat.live_policy?.name ?? "bc_flat.json"}</div>
            <div className="text-xs text-muted-foreground">
              {tstat.live_policy?.detail ?? "v1 policy - 15-dim obs, chassis_v1 dynamics"}
            </div>
            {tstat.live_policy?.note && <div className="text-xs text-muted-foreground">{String(tstat.live_policy.note)}</div>}
          </div>
          <div className="space-y-1.5">
            <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Best trained candidate</div>
            <div className="font-mono text-base md:text-lg">{tstat.candidate?.name ?? "-"}</div>
            {tstat.candidate?.detail && <div className="text-xs text-muted-foreground">{String(tstat.candidate.detail)}</div>}
            <div className="space-y-1 pt-1">
              <ScenarioBar label="go-to" v={tstat.candidate?.goto ?? latestSeries("success_goto")} />
              <ScenarioBar label="hover-hold" v={tstat.candidate?.hover_hold ?? latestSeries("success_hover_hold")} />
              <ScenarioBar label="land" v={tstat.candidate?.land ?? latestSeries("success_land")} />
            </div>
            {tstat.candidate?.note && <div className="text-xs text-muted-foreground">{String(tstat.candidate.note)}</div>}
          </div>
          <div className="space-y-1.5">
            <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Training now</div>
            {tstat.training?.status === "running" ? (
              <>
                <div className="font-mono text-base md:text-lg flex items-center gap-2">
                  {String(tstat.training.name ?? "run")}
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
                  </span>
                </div>
                {tstat.training.iter != null && tstat.training.iters != null && (
                  <>
                    <div className="flex-1 h-2 rounded bg-muted/40 overflow-hidden">
                      <div className="h-2 rounded bg-emerald-500" style={{ width: `${Math.round((100 * Number(tstat.training.iter)) / Number(tstat.training.iters))}%` }} />
                    </div>
                    <div className="text-xs text-muted-foreground">iteration {String(tstat.training.iter)} / {String(tstat.training.iters)}</div>
                  </>
                )}
                {tstat.training.note && <div className="text-xs text-muted-foreground">{String(tstat.training.note)}</div>}
              </>
            ) : (
              <div className="text-sm text-muted-foreground">
                idle{tstat.training?.name ? ` - last: ${String(tstat.training.name)}` : ""}
                {tstat.training?.note ? ` - ${String(tstat.training.note)}` : ""}
              </div>
            )}
            {Array.isArray(tstat.queue) && tstat.queue.length > 0 && (
              <div className="pt-1">
                <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Queued</div>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  {tstat.queue.map((q: any, i: number) => <li key={i}>{String(q)}</li>)}
                </ul>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-3 md:gap-4 md:grid-cols-5">
        <Card className="md:col-span-3">
          <CardHeader className="p-3 md:p-6">
            <div className="flex items-start justify-between gap-2 flex-wrap">
              <div>
                <CardTitle className="text-base md:text-lg">{MODE_META[chartMode].title}</CardTitle>
                <CardDescription className="text-xs md:text-sm">{MODE_META[chartMode].desc}</CardDescription>
              </div>
              <div className="flex gap-1 items-center flex-wrap">
                {([["all", "All"], [500, "500"], [200, "200"], [50, "50"]] as const).map(([v, lab]) => (
                  <button key={lab} onClick={() => setRange(v as number | "all")}
                    className={`px-2 py-1 rounded text-[11px] font-medium ${range === v ? "bg-secondary text-foreground border border-border" : "text-muted-foreground hover:text-foreground"}`}>
                    {lab}
                  </button>
                ))}
                <span className="w-px h-4 bg-border mx-1" />
                {(["curriculum", "loss", "generations"] as const).map((m) => (
                  <button key={m} onClick={() => setChartMode(m)}
                    className={`px-2 py-1 rounded text-[11px] font-medium ${chartMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"}`}>
                    {m === "curriculum" ? "Hill climb" : m === "loss" ? "Loss" : "Generations"}
                  </button>
                ))}
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-3 md:p-6 pt-0 md:pt-0 h-64 md:h-80">
            {activeData.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={range === "all" ? activeData : activeData.slice(-range)} margin={{ top: 5, right: 8, bottom: 5, left: -6 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 17%)" />
                  <XAxis dataKey={activeXKey} stroke="hsl(215 20% 65%)" fontSize={11} tickMargin={4} />
                  <YAxis stroke="hsl(215 20% 65%)" fontSize={10}
                         unit={chartMode === "loss" ? "" : "%"}
                         domain={chartMode === "loss" ? [0, "auto"] : [0, 100]} width={46} />
                  <Tooltip contentStyle={{ background: "hsl(222 47% 9%)", border: "1px solid hsl(217 33% 17%)", borderRadius: 8, fontSize: 13 }}
                           labelStyle={{ color: "hsl(210 40% 96%)" }} />
                  <Line type="monotone" dataKey={activeYKey} stroke={chartMode === "loss" ? "hsl(45 93% 58%)" : "hsl(217 91% 60%)"}
                        strokeWidth={2.5} dot={false} activeDot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : <div className="h-full grid place-items-center text-muted-foreground text-sm">
                  {chartMode === "generations" ? "no generations yet" : "waiting for training data"}
                </div>}
          </CardContent>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader className="p-3 md:p-6"><CardTitle className="text-base md:text-lg">Best variant scene params</CardTitle>
            <CardDescription className="text-xs md:text-sm">{best ? `${best.id} - generation ${best.generation}` : "none yet"}</CardDescription></CardHeader>
          <CardContent className="p-3 md:p-6 pt-0 md:pt-0">
            {best ? (
              <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs md:text-sm">
                {Object.entries(best.params).map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-2">
                    <dt className="text-muted-foreground truncate" title={k}>{k.replace(/_/g, " ")}</dt>
                    <dd className="font-mono">{typeof v === "number" ? v.toFixed(3) : String(v)}</dd>
                  </div>
                ))}
              </dl>
            ) : <p className="text-sm text-muted-foreground">waiting for variants</p>}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="p-3 md:p-6">
          <CardTitle className="text-base md:text-lg">t4 campaign rounds</CardTitle>
          <CardDescription className="text-xs md:text-sm">
            Per-round scenario success (t0 tier, held-out) - posts live as each DAgger round lands. Blue go-to, amber hover-hold, green land.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-3 md:p-6 pt-0 md:pt-0 h-56 md:h-64">
          {campaignData.length ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={campaignData} margin={{ top: 5, right: 8, bottom: 5, left: -6 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 17%)" />
                <XAxis dataKey="x" stroke="hsl(215 20% 65%)" fontSize={9} tickMargin={4} interval="preserveStartEnd" />
                <YAxis stroke="hsl(215 20% 65%)" fontSize={10} unit="%" domain={[0, 100]} width={46} />
                <Tooltip contentStyle={{ background: "hsl(222 47% 9%)", border: "1px solid hsl(217 33% 17%)", borderRadius: 8, fontSize: 13 }}
                         labelStyle={{ color: "hsl(210 40% 96%)" }} />
                <Line type="monotone" dataKey="goto" stroke="hsl(217 91% 60%)" strokeWidth={2} dot={{ r: 2.5 }} name="go-to" />
                <Line type="monotone" dataKey="hover" stroke="hsl(45 93% 58%)" strokeWidth={2} dot={{ r: 2.5 }} name="hover-hold" />
                <Line type="monotone" dataKey="land" stroke="hsl(142 71% 45%)" strokeWidth={2} dot={{ r: 2.5 }} name="land" />
              </LineChart>
            </ResponsiveContainer>
          ) : <div className="h-full grid place-items-center text-muted-foreground text-sm">no t4 campaign rounds yet</div>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="p-3 md:p-6">
          <CardTitle className="text-base md:text-lg flex items-center gap-2">
            Curriculum ladder
            {curr?.status === "working" && (
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
              </span>
            )}
          </CardTitle>
          <CardDescription className="text-xs md:text-sm">
            {curr?.status === "working"
              ? `running${curr.current_stage != null ? ` - stage goal ${JSON.stringify(curr.current_stage)}` : ""}${curr.note ? ` - ${curr.note}` : ""}`
              : curr?.stages?.length
                ? `idle - ${curr.stages.length} stage result${curr.stages.length === 1 ? "" : "s"} posted`
                : "no curriculum runs posted yet"}
          </CardDescription>
        </CardHeader>
        {!!curr?.stages?.length && (
          <CardContent className="p-3 md:p-6 pt-0 md:pt-0">
            <div className="max-h-[220px] overflow-y-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Goal</TableHead><TableHead>Trainer</TableHead>
                    <TableHead className="text-right">Success</TableHead>
                    <TableHead className="text-right">Return</TableHead>
                    <TableHead className="text-right">Steps</TableHead>
                    <TableHead className="text-right">Budget</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[...curr.stages].reverse().map((s: any, i: number) => (
                    <TableRow key={i}>
                      <TableCell className="font-mono text-xs">{s.goal_m}m</TableCell>
                      <TableCell className="text-xs text-muted-foreground">{String(s.trainer ?? "cem")}</TableCell>
                      <TableCell className={`text-right font-mono ${s.success_rate > 0 ? "text-emerald-400" : ""}`}>
                        {(s.success_rate * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-right font-mono">{(s.mean_return ?? 0).toFixed(1)}</TableCell>
                      <TableCell className="text-right font-mono">{s.mean_steps != null ? s.mean_steps.toFixed(0) : "-"}</TableCell>
                      <TableCell className="text-right text-xs text-muted-foreground">{String(s.budget ?? "-")}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        )}
      </Card>

      <Card>
        <CardHeader className="p-3 md:p-6"><CardTitle className="text-base md:text-lg">Generations</CardTitle>
          <CardDescription className="text-xs md:text-sm">Every evaluated scene-distribution variant, newest first</CardDescription></CardHeader>
        <CardContent className="p-3 md:p-6 pt-0 md:pt-0">
          {/* phone: card list, no horizontal scroll */}
          <div className="md:hidden space-y-2 max-h-[420px] overflow-y-auto pr-1">
            {recent.map((r) => (
              <div key={r.id} className={`rounded-lg border border-border p-3 ${best?.id === r.id ? "bg-amber-400/5 border-amber-400/30" : ""}`}>
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-sm">{r.id}{best?.id === r.id ? " *" : ""}</span>
                  <MutTag m={r.mutator} />
                </div>
                <div className="mt-2 grid grid-cols-3 gap-2 text-center">
                  <div>
                    <div className="text-[10px] uppercase tracking-wide text-muted-foreground">success</div>
                    <div className={`font-mono text-base ${r.metrics.success_rate > 0 ? "text-emerald-400" : ""}`}>{(r.metrics.success_rate * 100).toFixed(0)}%</div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wide text-muted-foreground">return</div>
                    <div className="font-mono text-base">{(r.metrics.mean_return ?? 0).toFixed(1)}</div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wide text-muted-foreground">novelty</div>
                    <div className="font-mono text-base">{r.novelty != null ? r.novelty.toFixed(2) : "-"}</div>
                  </div>
                </div>
                <div className="mt-1 text-[11px] text-muted-foreground">gen {r.generation} - {String(r.metrics.trainer ?? "cem")}</div>
              </div>
            ))}
            {!records.length && (
              <div className="text-center text-muted-foreground py-8 text-sm">
                No data yet - the poster on the research box has not checked in.
              </div>
            )}
          </div>
          {/* desktop: full table */}
          <div className="hidden md:block max-h-[520px] overflow-y-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Variant</TableHead><TableHead>Gen</TableHead><TableHead>Mutator</TableHead>
                  <TableHead className="text-right">Success</TableHead><TableHead className="text-right">Return</TableHead>
                  <TableHead className="text-right">Novelty</TableHead><TableHead className="text-right">Trainer</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recent.map((r) => (
                  <TableRow key={r.id} className={best?.id === r.id ? "bg-amber-400/5" : ""}>
                    <TableCell className="font-mono text-xs">{r.id}{best?.id === r.id ? " *" : ""}</TableCell>
                    <TableCell>{r.generation}</TableCell>
                    <TableCell><MutTag m={r.mutator} /></TableCell>
                    <TableCell className="text-right font-mono">{(r.metrics.success_rate * 100).toFixed(0)}%</TableCell>
                    <TableCell className="text-right font-mono">{(r.metrics.mean_return ?? 0).toFixed(1)}</TableCell>
                    <TableCell className="text-right font-mono">{r.novelty != null ? r.novelty.toFixed(2) : "-"}</TableCell>
                    <TableCell className="text-right text-xs text-muted-foreground">{String(r.metrics.trainer ?? "cem")}</TableCell>
                  </TableRow>
                ))}
                {!records.length && (
                  <TableRow><TableCell colSpan={7} className="text-center text-muted-foreground py-8">
                    No data yet - the poster on the research box has not checked in.
                  </TableCell></TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
