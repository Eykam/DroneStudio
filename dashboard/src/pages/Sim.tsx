import { useSearchParams } from "react-router-dom";
import { lazy, Suspense } from "react";
import Dashboard from "./Dashboard";

const Watch = lazy(() => import("./Watch"));

export default function Sim() {
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") === "research" ? "research" : "live";
  return (
    <div className="space-y-4">
      <div className="flex gap-1 border-b border-border">
        {([["live", "Simulation"], ["research", "Research loop"]] as const).map(([t, label]) => (
          <button key={t} onClick={() => setParams(t === "live" ? {} : { tab: t }, { replace: true })}
            className={`px-3 py-2 text-sm font-medium border-b-2 -mb-px ${
              tab === t ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"}`}>
            {label}
          </button>
        ))}
      </div>
      {tab === "live" ? (
        <Suspense fallback={<div className="h-96 grid place-items-center text-muted-foreground">Loading viewer...</div>}>
          <div className="-m-3 md:-m-6"><Watch embedded /></div>
        </Suspense>
      ) : (
        <Dashboard />
      )}
    </div>
  );
}
