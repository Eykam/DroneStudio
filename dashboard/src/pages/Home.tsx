import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CircuitBoard, Box, Radar, Star, ArrowRight, Bird } from "lucide-react";
import { fetchCadDesigns, fetchState } from "@/api";
import { BRAND, BRAND_TAGLINE } from "@/brand";

const j = (path: string) => async () => (await (await fetch(path, { credentials: "same-origin" })).json()) as any;
const ago = (ts?: string) => {
  if (!ts) return "never";
  const s = Math.max(0, (Date.now() - new Date(ts).getTime()) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
};

export default function Home() {
  const tstatQ = useQuery({ queryKey: ["training-status"], queryFn: j("/api/training/status"), refetchInterval: 10_000 });
  const streamQ = useQuery({ queryKey: ["stream-state"], queryFn: j("/api/stream/state"), refetchInterval: 5_000 });
  const eeQ = useQuery({ queryKey: ["ee-boards"], queryFn: j("/api/ee/boards"), refetchInterval: 30_000 });
  const eeProgQ = useQuery({ queryKey: ["ee-progress"], queryFn: j("/api/ee/progress"), refetchInterval: 10_000 });
  const cadQ = useQuery({ queryKey: ["cad"], queryFn: fetchCadDesigns, refetchInterval: 30_000 });
  const stateQ = useQuery({ queryKey: ["state"], queryFn: fetchState, refetchInterval: 15_000 });

  const tstat = (tstatQ.data ?? {}) as any;
  const stream = streamQ.data as any;
  const streamAge = stream?.last_event_at ? (Date.now() - new Date(stream.last_event_at).getTime()) / 1000 : 1e9;
  const streamLive = streamAge < 30 && stream?.meta?.status && stream.meta.status !== "offline";

  const boards = (eeQ.data?.boards ?? []) as any[];
  const board = boards[0];
  const latest = board?.versions?.[board.versions.length - 1];
  const eeProg = eeProgQ.data?.current as any;
  const eeLive = eeProg?.status === "working";

  const designs = cadQ.data?.designs ?? [];
  const latestCad = designs[designs.length - 1];
  const records = ((stateQ.data?.records ?? []) as any[]).filter((r) => r && r.metrics && typeof r.metrics.success_rate === "number" && (r.kind == null || r.kind === "variant"));
  const best = records.length ? [...records].sort((a, b) => b.metrics.success_rate - a.metrics.success_rate)[0] : null;

  const nav = useNavigate();

  const cardCls =
    "group relative overflow-hidden rounded-xl border-border/60 bg-card/70 backdrop-blur transition-all duration-200 hover:border-primary/50 hover:shadow-lg hover:shadow-primary/5 hover:-translate-y-0.5 cursor-pointer flex flex-col";
  const iconChip = "inline-flex items-center justify-center h-8 w-8 rounded-lg bg-primary/10 border border-primary/20 shrink-0";
  const openCls = "mt-auto inline-flex items-center gap-1 text-xs text-primary pt-3 group-hover:gap-2 transition-all";

  return (
    <div className="min-h-[calc(100dvh-7rem)] flex items-center justify-center">
      <div className="w-full max-w-5xl space-y-8 md:space-y-10 py-6">
      <header className="text-center space-y-3">
        <h1 className="text-3xl md:text-4xl font-bold tracking-tight flex items-center justify-center gap-3">
          <Bird className="h-8 w-8 md:h-9 md:w-9 text-primary" />{BRAND}
        </h1>
        <p className="text-sm md:text-base text-muted-foreground max-w-xl mx-auto leading-relaxed">
          {BRAND_TAGLINE} - one live view across the sim, EE, and CAD research loops.
        </p>
      </header>

      <div className="grid gap-4 md:gap-5 md:grid-cols-3 text-left">
        {/* SIM */}
        <Card className={cardCls} onClick={() => nav("/sim")}>
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent" />
          <CardHeader className="p-4 pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <span className={iconChip}><Radar className="h-4 w-4 text-primary" /></span> SIM
            </CardTitle>
            <CardDescription className="text-xs">navigation policy research</CardDescription>
          </CardHeader>
          <CardContent className="p-4 pt-1 flex flex-col flex-1 text-sm space-y-2">
            <div className="flex items-center gap-2 text-xs">
              <span className={`inline-block w-2 h-2 rounded-full ${streamLive ? "bg-emerald-500" : "bg-gray-600"}`} />
              <span className="text-muted-foreground">{streamLive ? "live policy streaming" : "watch channel off"}</span>
            </div>
            {tstat.training?.status === "running" && (
              <div className="flex items-center gap-2 text-xs">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute h-full w-full rounded-full bg-emerald-400 opacity-60" />
                  <span className="relative rounded-full h-2 w-2 bg-emerald-400" />
                </span>
                <span className="font-mono text-xs">{String(tstat.training.name ?? "run")} training</span>
              </div>
            )}
            {best && (
              <div className="text-xs text-muted-foreground">
                best <span className="font-mono text-foreground">{best.id}</span> - {(best.metrics.success_rate * 100).toFixed(0)}%
              </div>
            )}
            <span className={openCls}>Open SIM <ArrowRight className="h-3 w-3" /></span>
          </CardContent>
        </Card>

        {/* EE */}
        <Card className={cardCls} onClick={() => nav("/ee")}>
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent" />
          <CardHeader className="p-4 pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <span className={iconChip}><CircuitBoard className="h-4 w-4 text-primary" /></span> EE
            </CardTitle>
            <CardDescription className="text-xs">board designs + verification</CardDescription>
          </CardHeader>
          <CardContent className="p-4 pt-1 flex flex-col flex-1 text-sm space-y-2">
            <div className="flex items-center gap-2 text-xs">
              <span className={`inline-block w-2 h-2 rounded-full ${eeLive ? "bg-emerald-500" : "bg-gray-600"}`} />
              <span className="text-muted-foreground">
                {eeLive ? `round in flight: ${eeProg.candidate ?? ""}` : eeProg?.candidate ? `last round ${eeProg.candidate} - ${eeProg.outcome ?? eeProg.status}` : "loop idle"}
              </span>
            </div>
            {board && latest && (
              <div className="text-xs text-muted-foreground">
                {board.name} <span className="font-mono text-foreground">v{latest.version}</span>
                {latest.adopted && <Star className="inline h-3 w-3 text-yellow-400 ml-1 -mt-0.5" />}
                {latest.score != null && <span> - score {latest.score}</span>}
              </div>
            )}
            <span className={openCls}>Open EE <ArrowRight className="h-3 w-3" /></span>
          </CardContent>
        </Card>

        {/* CAD */}
        <Card className={cardCls} onClick={() => nav("/cad")}>
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent" />
          <CardHeader className="p-4 pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <span className={iconChip}><Box className="h-4 w-4 text-primary" /></span> CAD
            </CardTitle>
            <CardDescription className="text-xs">chassis + mechanicals</CardDescription>
          </CardHeader>
          <CardContent className="p-4 pt-1 flex flex-col flex-1 text-sm space-y-2">
            {latestCad ? (
              <div className="text-xs text-muted-foreground">
                latest <span className="font-mono text-foreground">{latestCad.id}</span>
                {latestCad.metrics?.mass_g != null && <span> - {Number(latestCad.metrics.mass_g).toFixed(0)} g</span>}
              </div>
            ) : <div className="text-muted-foreground text-xs">no designs yet</div>}
            <span className={openCls}>Open CAD <ArrowRight className="h-3 w-3" /></span>
          </CardContent>
        </Card>
      </div>
      </div>
    </div>
  );
}
