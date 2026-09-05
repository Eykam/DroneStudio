import { Routes, Route, Navigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { lazy, Suspense } from "react";
import Login from "./pages/Login";
import Home from "./pages/Home";
import Shell from "./components/Shell";

const Sim = lazy(() => import("./pages/Sim"));
const Cad = lazy(() => import("./pages/Cad"));
const Ee = lazy(() => import("./pages/Ee"));

export default function App() {
  const me = useQuery({
    queryKey: ["me"],
    queryFn: async () => {
      const r = await fetch("/api/me", { credentials: "same-origin" });
      return (await r.json()).authed as boolean;
    },
    refetchInterval: 60_000,
  });
  if (me.isLoading) return <div className="min-h-screen grid place-items-center text-muted-foreground">Loading...</div>;
  const authed = (el: React.ReactNode, fallback?: string) =>
    me.data ? <Shell><Suspense fallback={fallback ? <div className="min-h-[40vh] grid place-items-center text-muted-foreground">{fallback}</div> : null}>{el}</Suspense></Shell>
            : <Navigate to="/login" />;
  return (
    <Routes>
      <Route path="/login" element={me.data ? <Navigate to="/" /> : <Login />} />
      <Route path="/" element={authed(<Home />)} />
      <Route path="/sim" element={authed(<Sim />, "Loading SIM...")} />
      <Route path="/watch" element={<Navigate to="/sim" replace />} />
      <Route path="/cad" element={authed(<Cad />, "Loading CAD viewer...")} />
      <Route path="/ee" element={authed(<Ee />, "Loading EE viewer...")} />
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}
