import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ArrowLeft, CircuitBoard, CheckCircle2, XCircle, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import CadViewer from "@/components/CadViewer";

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
};
type EeBoard = { id: string; name: string; created_at: string; versions: EeVersion[] };

type Diff = {
  from: number; to: number;
  added_comps: { ref: string; value: string }[];
  removed_comps: { ref: string; value: string }[];
  changed_comps: { ref: string; from?: string; to: string }[];
  added_nets: string[];
  removed_nets: string[];
};

const fileUrl = (b: string, v: number, kind: string) =>
  `/api/ee/boards/${b}/versions/${v}/file?kind=${kind}`;

export default function Ee() {
  const boards = useQuery({
    queryKey: ["ee-boards"],
    queryFn: async () => (await (await fetch("/api/ee/boards", { credentials: "same-origin" })).json()).boards as EeBoard[],
    refetchInterval: 30_000,
  });
  const [boardId, setBoardId] = useState<string | null>(null);
  const [ver, setVer] = useState<number | null>(null);
  const [tab, setTab] = useState<"sch" | "pcb" | "3d" | "diff">("sch");
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
    <div className="min-h-screen p-4 md:p-6 space-y-4 max-w-6xl mx-auto">
      <header className="flex items-center justify-between gap-2">
        <h1 className="text-lg md:text-xl font-semibold flex items-center gap-2">
          <CircuitBoard className="h-5 w-5" /> EE - versioned boards
        </h1>
        <Link to="/"><Button variant="ghost" size="sm"><ArrowLeft className="h-4 w-4" /> Back</Button></Link>
      </header>

      {boards.data && boards.data.length === 0 && (
        <Card><CardContent className="p-6 text-sm text-muted-foreground">
          No board designs published yet. The EE researcher loop on box #3 posts
          adopted candidates here (versioned schematic + layout, pinned by netlist).
        </CardContent></Card>
      )}

      {boards.data && boards.data.length > 0 && (
        <div className="flex gap-2 flex-wrap">
          {boards.data.map((b) => (
            <Button key={b.id} size="sm" variant={b.id === board?.id ? "default" : "outline"}
              onClick={() => { setBoardId(b.id); setVer(null); setDiffFrom(null); }}>
              {b.name} <span className="text-xs opacity-70">v{b.versions.length}</span>
            </Button>
          ))}
        </div>
      )}

      {board && version && (
        <>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-3 flex-wrap">
                <span>{board.name} - v{version.version}</span>
                {version.adopted && <Star className="h-4 w-4 text-yellow-400" />}
                {version.gates.map((g) => (
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
                {(["sch", "pcb", "3d", "diff"] as const).map((t) => (
                  <Button key={t} size="sm" variant={tab === t ? "default" : "ghost"}
                    onClick={() => setTab(t)}
                    disabled={(t === "sch" && !version.files.sch_svg) || (t === "pcb" && !version.files.pcb_svg) || (t === "3d" && !version.files.glb) || (t === "diff" && !prev)}>
                    {t === "sch" ? "Schematic" : t === "pcb" ? "Layout" : t === "3d" ? "3D" : "Diff"}
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
                <div className="bg-white rounded-md p-2 overflow-auto max-h-[70vh]">
                  <img src={fileUrl(board.id, version.version, "sch_svg")} alt="schematic" className="min-w-[800px] w-full" />
                </div>
              )}
              {tab === "pcb" && version.files.pcb_svg && (
                <div className="bg-white rounded-md p-2 overflow-auto max-h-[70vh]">
                  <img src={fileUrl(board.id, version.version, "pcb_svg")} alt="layout" className="min-w-[600px] w-full" />
                </div>
              )}
              {tab === "3d" && version.files.glb && (
                <div className="h-[60vh]"><CadViewer url={fileUrl(board.id, version.version, "glb")} /></div>
              )}
              {tab === "diff" && diff.data && (
                <div className="grid md:grid-cols-2 gap-3 text-xs">
                  <DiffList title={`Components added (${diff.data.added_comps.length})`}
                    rows={diff.data.added_comps.map((c) => `${c.ref} ${c.value}`)} tone="text-green-500" />
                  <DiffList title={`Components removed (${diff.data.removed_comps.length})`}
                    rows={diff.data.removed_comps.map((c) => `${c.ref} ${c.value}`)} tone="text-red-500" />
                  <DiffList title={`Components changed (${diff.data.changed_comps.length})`}
                    rows={diff.data.changed_comps.map((c) => `${c.ref}: ${c.from} -> ${c.to}`)} tone="text-yellow-500" />
                  <DiffList title={`Nets added (${diff.data.added_nets.length})`} rows={diff.data.added_nets} tone="text-green-500" />
                  <DiffList title={`Nets removed (${diff.data.removed_nets.length})`} rows={diff.data.removed_nets} tone="text-red-500" />
                </div>
              )}
              {tab === "diff" && diff.data && diff.data.added_comps.length + diff.data.removed_comps.length + diff.data.changed_comps.length + diff.data.added_nets.length + diff.data.removed_nets.length === 0 && (
                <p className="text-xs text-muted-foreground">No structural netlist changes between these versions.</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">Versions</CardTitle></CardHeader>
            <CardContent className="flex gap-2 flex-wrap">
              {[...board.versions].reverse().map((v) => (
                <Button key={v.version} size="sm" variant={v.version === version.version ? "default" : "outline"}
                  onClick={() => { setVer(v.version); setDiffFrom(null); }}>
                  v{v.version} {v.adopted && "*"}
                </Button>
              ))}
            </CardContent>
          </Card>
        </>
      )}
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
