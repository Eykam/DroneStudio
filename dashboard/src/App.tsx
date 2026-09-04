import { Routes, Route, Navigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";

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
  return (
    <Routes>
      <Route path="/login" element={me.data ? <Navigate to="/" /> : <Login />} />
      <Route path="/" element={me.data ? <Dashboard /> : <Navigate to="/login" />} />
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}
