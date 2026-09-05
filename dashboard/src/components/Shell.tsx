import { NavLink, useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Home, CircuitBoard, Box, Radar, LogOut, FlaskConical } from "lucide-react";
import { logout } from "@/api";
import { BRAND } from "@/brand";

const NAV = [
  { to: "/", label: "Home", icon: Home, end: true },
  { to: "/ee", label: "EE", icon: CircuitBoard, end: false },
  { to: "/cad", label: "CAD", icon: Box, end: false },
  { to: "/sim", label: "SIM", icon: Radar, end: false },
];

export default function Shell({ children }: { children: React.ReactNode }) {
  const nav = useNavigate();
  const qc = useQueryClient();
  const signOut = async () => { await logout(); await qc.invalidateQueries({ queryKey: ["me"] }); nav("/login"); };
  const itemCls = (active: boolean) =>
    `flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
      active ? "bg-primary/15 text-primary" : "text-muted-foreground hover:text-foreground hover:bg-muted/40"}`;

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* desktop sidebar */}
      <aside className="hidden md:flex fixed inset-y-0 left-0 w-60 flex-col border-r border-border bg-background z-40">
        <div className="flex items-center gap-2.5 px-4 h-16 border-b border-border">
          <FlaskConical className="h-5 w-5 text-primary shrink-0" />
          <span className="font-bold tracking-tight text-lg">{BRAND}</span>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => itemCls(isActive)}>
              <Icon className="h-4 w-4 shrink-0" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="p-3 border-t border-border">
          <button onClick={signOut}
            className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-muted/40 w-full">
            <LogOut className="h-4 w-4" /> Sign out
          </button>
        </div>
      </aside>

      {/* mobile top bar - stacking layout preserved */}
      <div className="md:hidden sticky top-0 z-40 bg-background/95 backdrop-blur border-b border-border">
        <div className="flex items-center gap-1 px-2 h-12">
          <FlaskConical className="h-4 w-4 text-primary shrink-0 mx-1" />
          <span className="font-bold text-sm mr-2">{BRAND}</span>
          <nav className="flex items-center gap-0.5 overflow-x-auto flex-1">
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink key={to} to={to} end={end}
                className={({ isActive }) =>
                  `flex items-center gap-1 rounded px-2 py-1.5 text-xs font-medium whitespace-nowrap ${
                    isActive ? "bg-primary/15 text-primary" : "text-muted-foreground"}`}>
                <Icon className="h-3.5 w-3.5" /> {label}
              </NavLink>
            ))}
          </nav>
          <button onClick={signOut} className="p-2 text-muted-foreground shrink-0"><LogOut className="h-4 w-4" /></button>
        </div>
      </div>

      <div className="md:pl-60">
        <main className="p-3 md:p-6 max-w-7xl mx-auto">{children}</main>
      </div>
    </div>
  );
}
