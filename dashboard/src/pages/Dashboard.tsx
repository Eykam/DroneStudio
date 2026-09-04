import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { fetchState, logout, type ArchiveRecord } from "@/api";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { Activity, FlaskConical, Trophy, Waves, LogOut, Bot, Dices } from "lucide-react";

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

  return (
    <div className="min-h-screen p-4 md:p-8 max-w-6xl mx-auto space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <FlaskConical className="h-6 w-6 text-primary" /> DroneStudio Auto-Researcher
          </h1>
          <p className="text-sm text-muted-foreground">Live research loop state - refreshes every 10s - last update {updated}</p>
        </div>
        <Button variant="ghost" size="sm" onClick={async () => { await logout(); await qc.invalidateQueries({ queryKey: ["me"] }); nav("/login"); }}>
          <LogOut className="h-4 w-4" /> Sign out
        </Button>
      </header>

      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2"><CardDescription>Run status</CardDescription>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Activity className={`h-5 w-5 ${running ? "text-emerald-400 animate-pulse" : "text-muted-foreground"}`} />
              {run ? (running ? `Running - gen batch ${run.generations ?? "?"}` : (run.status ?? "idle")) : "No signal yet"}
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            {run?.detail ? String(run.detail) : run?.started_at ? `started ${String(run.started_at)}` : "waiting for first heartbeat"}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2"><CardDescription>Variants evaluated</CardDescription>
            <CardTitle className="text-lg">{records.length}</CardTitle></CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            {perGen.size} generation{perGen.size === 1 ? "" : "s"}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2"><CardDescription>Best success rate</CardDescription>
            <CardTitle className="text-lg flex items-center gap-2">
              <Trophy className="h-5 w-5 text-amber-400" />
              {best ? `${(best.metrics.success_rate * 100).toFixed(0)}%` : "-"}
            </CardTitle></CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            {best ? `${best.id} via ` : "no variants yet"}{best && <MutTag m={best.mutator} />}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2"><CardDescription>Outer-loop brain</CardDescription>
            <CardTitle className="text-lg flex items-center gap-2">
              {mutators.has("codex")
                ? <><Bot className="h-5 w-5 text-emerald-400" /> ChatGPT</>
                : <><Dices className="h-5 w-5 text-amber-400" /> heuristic</>}
            </CardTitle></CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            {[...mutators].filter((m) => m !== "none").join(", ") || "-"}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-5">
        <Card className="md:col-span-3">
          <CardHeader><CardTitle>Best success rate per generation</CardTitle>
            <CardDescription>Selection trajectory across the run</CardDescription></CardHeader>
          <CardContent className="h-64">
            {chart.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chart} margin={{ top: 5, right: 10, bottom: 5, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 17%)" />
                  <XAxis dataKey="gen" stroke="hsl(215 20% 65%)" fontSize={12} />
                  <YAxis stroke="hsl(215 20% 65%)" fontSize={12} unit="%" domain={[0, 100]} />
                  <Tooltip contentStyle={{ background: "hsl(222 47% 9%)", border: "1px solid hsl(217 33% 17%)", borderRadius: 8 }}
                           labelStyle={{ color: "hsl(210 40% 96%)" }} />
                  <Line type="monotone" dataKey="success" stroke="hsl(217 91% 60%)" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : <div className="h-full grid place-items-center text-muted-foreground text-sm">no generations yet</div>}
          </CardContent>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader><CardTitle>Best variant scene params</CardTitle>
            <CardDescription>{best ? `${best.id} - generation ${best.generation}` : "none yet"}</CardDescription></CardHeader>
          <CardContent>
            {best ? (
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
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
        <CardHeader><CardTitle>Generations</CardTitle>
          <CardDescription>Every evaluated scene-distribution variant, newest first</CardDescription></CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Variant</TableHead><TableHead>Gen</TableHead><TableHead>Mutator</TableHead>
                <TableHead className="text-right">Success</TableHead><TableHead className="text-right">Return</TableHead>
                <TableHead className="text-right">Novelty</TableHead><TableHead className="text-right">Trainer</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {[...records].reverse().slice(0, 60).map((r) => (
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
        </CardContent>
      </Card>
    </div>
  );
}
