import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CircuitBoard, Box, Radar, CheckCircle2, XCircle, Star, ArrowRight } from "lucide-react";
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

  return (
    <div className="space-y-4 md:space-y-6">
      <header>
        <h1 className="text-xl md:text-2xl font-bold tracking-tight">{BRAND}</h1>
        <p className="text-xs md:text-sm text-muted-foreground mt-1">{BRAND_TAGLINE} - live across sim, EE, and CAD - auto-refresh</p>
      </header>

      <div className="grid gap-3 md:gap-4 md:grid-cols-3">
        {/* SIM */}
        <Card>
          <CardHeader className="p-4 pb-2">
            <CardTitle className="text-base flex items-center gap-2"><Radar className="h-4 w-4 text-primary" /> SIM</CardTitle>
            <CardDescription className="text-xs">navigation policy research</CardDescription>
          </CardHeader>
          <CardContent className="p-4 pt-1 space-y-2 text-sm">
            <div className="flex items-center gap-2 text-xs">
              <span className={`inline-block w-2 h-2 rounded-full ${streamLive ? "bg-emerald-500" : "bg-gray-600"}`} />
              <span className="text-muted-foreground">{streamLive ? "live policy streaming" : "watch channel off"}</span>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Live policy</div>
              <div className="font-mono">{tstat.live_policy?.name ?? "-"}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Training</div>
              {tstat.training?.status === "running" ? (
                <div className="font-mono flex items-center gap-2">
                  {String(tstat.training.name ?? "run")}
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute h-full w-full rounded-full bg-emerald-400 opacity-60" />
                    <span className="relative rounded-full h-2 w-2 bg-emerald-400" />
                  </span>
                </div>
              ) : <div className="text-muted-foreground">idle</div>}
            </div>
            {best && (
              <div className="text-xs text-muted-foreground">
                best variant <span className="font-mono text-foreground">{best.id}</span> - {(best.metrics.success_rate * 100).toFixed(0)}% success
              </div>
            )}
            <Link to="/sim" className="inline-flex items-center gap-1 text-xs text-primary hover:underline pt-1">
              Open SIM <ArrowRight className="h-3 w-3" />
            </Link>
          </CardContent>
        </Card>

        {/* EE */}
        <Card>
          <CardHeader className="p-4 pb-2">
            <CardTitle className="text-base flex items-center gap-2"><CircuitBoard className="h-4 w-4 text-primary" /> EE</CardTitle>
            <CardDescription className="text-xs">board designs + verification</CardDescription>
          </CardHeader>
          <CardContent className="p-4 pt-1 space-y-2 text-sm">
            <div className="flex items-center gap-2 text-xs">
              <span className={`inline-block w-2 h-2 rounded-full ${eeLive ? "bg-emerald-500" : "bg-gray-600"}`} />
              <span className="text-muted-foreground">
                {eeLive ? `round in flight: ${eeProg.candidate ?? ""} (${eeProg.phase ?? "working"})`
                        : eeProg?.candidate ? `last round ${eeProg.candidate} - ${eeProg.outcome ?? eeProg.status}` : "loop idle"}
              </span>
            </div>
            {board && latest ? (
              <>
                <div>
                  <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{board.name} - latest</div>
                  <div className="font-mono flex items-center gap-2">
                    v{latest.version} {latest.adopted && <Star className="h-3.5 w-3.5 text-yellow-400" />}
                    {latest.score != null && <span className="text-xs text-muted-foreground">score {latest.score}</span>}
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    {latest.gates.map((g: any) => (
                      <span key={g.gate} className="flex items-center gap-1 text-xs text-muted-foreground">
                        {g.pass ? <CheckCircle2 className="h-3 w-3 text-green-500" /> : <XCircle className="h-3 w-3 text-red-500" />}
                        {g.gate}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="text-xs text-muted-foreground">{board.versions.length} version{board.versions.length === 1 ? "" : "s"} published</div>
              </>
            ) : <div className="text-muted-foreground text-xs">no boards published yet</div>}
            <Link to="/ee" className="inline-flex items-center gap-1 text-xs text-primary hover:underline pt-1">
              Open EE <ArrowRight className="h-3 w-3" />
            </Link>
          </CardContent>
        </Card>

        {/* CAD */}
        <Card>
          <CardHeader className="p-4 pb-2">
            <CardTitle className="text-base flex items-center gap-2"><Box className="h-4 w-4 text-primary" /> CAD</CardTitle>
            <CardDescription className="text-xs">chassis + mechanicals</CardDescription>
          </CardHeader>
          <CardContent className="p-4 pt-1 space-y-2 text-sm">
            {latestCad ? (
              <>
                <div>
                  <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Latest design</div>
                  <div className="font-mono">{latestCad.id}</div>
                  <div className="text-xs text-muted-foreground">
                    {latestCad.metrics?.mass_g != null ? `${Number(latestCad.metrics.mass_g).toFixed(0)} g - ` : ""}
                    {ago(latestCad.created_at)}
                  </div>
                </div>
                <div className="text-xs text-muted-foreground">{designs.length} design{designs.length === 1 ? "" : "s"} total</div>
              </>
            ) : <div className="text-muted-foreground text-xs">no designs yet</div>}
            <Link to="/cad" className="inline-flex items-center gap-1 text-xs text-primary hover:underline pt-1">
              Open CAD <ArrowRight className="h-3 w-3" />
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
