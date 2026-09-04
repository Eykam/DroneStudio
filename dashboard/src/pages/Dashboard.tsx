import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { fetchState, logout, type ArchiveRecord } from "@/api";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { Activity, FlaskConical, Trophy, LogOut, Bot, Dices } from "lucide-react";

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

export default function Dashboard() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const { data, error } = useQuery({ queryKey: ["state"], queryFn: fetchState, refetchInterval: 10_000 });

  if (error) return <div className="p-8 text-red-400">Failed to load state: {(error as Error).message}</div>;
  const records = data?.records ?? [];
  const run = data?.run ?? null;
  const best = bestOf(records);

  const perGen = new Map<number, number>();
  for (const r of records) {
    perGen.set(r.generation, Math.max(perGen.get(r.generation) ?? -1, r.metrics.success_rate));
  }
  const chart = [...perGen.entries()].sort((a, b) => a[0] - b[0])
    .map(([g, s]) => ({ gen: `g${g}`, success: +(s * 100).toFixed(1) }));

  const mutators = new Set(records.map((r) => r.mutator));
  const running = run?.status === "running";
  const updated = data?.updated_at ? new Date(data.updated_at).toLocaleTimeString() : "never";
  const recent = [...records].reverse().slice(0, 60);

  return (
    <div className="min-h-screen p-3 md:p-8 max-w-6xl mx-auto space-y-4 md:space-y-6">
      <header className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h1 className="text-base md:text-2xl font-bold tracking-tight flex items-center gap-2">
            <FlaskConical className="h-5 w-5 md:h-6 md:w-6 text-primary shrink-0" />
            <span>DroneStudio Auto-Researcher</span>
          </h1>
          <p className="text-xs md:text-sm text-muted-foreground mt-1">
            Live loop state - auto-refresh 10s - updated {updated}
          </p>
        </div>
        <Button variant="ghost" size="sm" className="shrink-0 mt-1"
          onClick={async () => { await logout(); await qc.invalidateQueries({ queryKey: ["me"] }); nav("/login"); }}>
          <LogOut className="h-4 w-4" /> <span className="hidden sm:inline">Sign out</span>
        </Button>
      </header>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 md:gap-4">
        <Card>
          <CardHeader className="p-3 md:p-6 pb-1 md:pb-2"><CardDescription className="text-xs">Run status</CardDescription>
            <CardTitle className="flex items-center gap-1.5 text-base md:text-lg">
              <Activity className={`h-4 w-4 md:h-5 md:w-5 shrink-0 ${running ? "text-emerald-400 animate-pulse" : "text-muted-foreground"}`} />
              <span className="leading-tight">{run ? (running ? `Running - gen ${run.generations ?? "?"}` : (run.status ?? "idle")) : "No signal"}</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-3 md:p-6 pt-0 md:pt-0 text-xs md:text-sm text-muted-foreground">
            <span className="line-clamp-2 break-all">{run?.detail ? String(run.detail).split("/").pop() : "waiting for heartbeat"}</span>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="p-3 md:p-6 pb-1 md:pb-2"><CardDescription className="text-xs">Variants evaluated</CardDescription>
            <CardTitle className="text-base md:text-lg">{records.length}</CardTitle></CardHeader>
          <CardContent className="p-3 md:p-6 pt-0 md:pt-0 text-xs md:text-sm text-muted-foreground">
            {perGen.size} generation{perGen.size === 1 ? "" : "s"}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="p-3 md:p-6 pb-1 md:pb-2"><CardDescription className="text-xs">Best success rate</CardDescription>
            <CardTitle className="text-base md:text-lg flex items-center gap-1.5">
              <Trophy className="h-4 w-4 md:h-5 md:w-5 text-amber-400 shrink-0" />
              {best ? `${(best.metrics.success_rate * 100).toFixed(0)}%` : "-"}
            </CardTitle></CardHeader>
          <CardContent className="p-3 md:p-6 pt-0 md:pt-0 text-xs md:text-sm text-muted-foreground">
            {best ? `${best.id} via ` : "no variants yet"}{best && <MutTag m={best.mutator} />}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="p-3 md:p-6 pb-1 md:pb-2"><CardDescription className="text-xs">Outer-loop brain</CardDescription>
            <CardTitle className="text-base md:text-lg flex items-center gap-1.5">
              {mutators.has("codex")
                ? <><Bot className="h-4 w-4 md:h-5 md:w-5 text-emerald-400 shrink-0" /> ChatGPT</>
                : <><Dices className="h-4 w-4 md:h-5 md:w-5 text-amber-400 shrink-0" /> heuristic</>}
            </CardTitle></CardHeader>
          <CardContent className="p-3 md:p-6 pt-0 md:pt-0 text-xs md:text-sm text-muted-foreground">
            {[...mutators].filter((m) => m !== "none").join(", ") || "-"}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-3 md:gap-4 md:grid-cols-5">
        <Card className="md:col-span-3">
          <CardHeader className="p-3 md:p-6"><CardTitle className="text-base md:text-lg">Best success rate per generation</CardTitle>
            <CardDescription className="text-xs md:text-sm">Selection trajectory across the run</CardDescription></CardHeader>
          <CardContent className="p-3 md:p-6 pt-0 md:pt-0 h-52 md:h-64">
            {chart.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chart} margin={{ top: 5, right: 8, bottom: 5, left: -6 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 17%)" />
                  <XAxis dataKey="gen" stroke="hsl(215 20% 65%)" fontSize={11} tickMargin={4} />
                  <YAxis stroke="hsl(215 20% 65%)" fontSize={10} unit="%" domain={[0, 100]} width={46} />
                  <Tooltip contentStyle={{ background: "hsl(222 47% 9%)", border: "1px solid hsl(217 33% 17%)", borderRadius: 8, fontSize: 13 }}
                           labelStyle={{ color: "hsl(210 40% 96%)" }} />
                  <Line type="monotone" dataKey="success" stroke="hsl(217 91% 60%)" strokeWidth={2.5} dot={{ r: 3.5 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : <div className="h-full grid place-items-center text-muted-foreground text-sm">no generations yet</div>}
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
        <CardHeader className="p-3 md:p-6"><CardTitle className="text-base md:text-lg">Generations</CardTitle>
          <CardDescription className="text-xs md:text-sm">Every evaluated scene-distribution variant, newest first</CardDescription></CardHeader>
        <CardContent className="p-3 md:p-6 pt-0 md:pt-0">
          {/* phone: card list, no horizontal scroll */}
          <div className="md:hidden space-y-2">
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
          <div className="hidden md:block">
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
