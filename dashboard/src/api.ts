export type ArchiveRecord = {
  id: string; parent: string | null; generation: number;
  params: Record<string, number>;
  metrics: { success_rate: number; mean_return: number; mean_steps?: number; trainer?: string; backend?: string } & Record<string, unknown>;
  mutator: string; ts: string; novelty?: number;
};
export type DashState = {
  run: { status?: string; generations?: number; children?: number; started_at?: string; finished_at?: string; detail?: string } | null;
  records: ArchiveRecord[];
  reports: Record<string, unknown>[];
  updated_at: string | null;
};

export async function fetchState(): Promise<DashState> {
  const r = await fetch("/api/state", { credentials: "same-origin" });
  if (r.status === 401) throw new Error("unauthorized");
  if (!r.ok) throw new Error(`state fetch failed: ${r.status}`);
  return r.json();
}
export async function login(password: string): Promise<void> {
  const r = await fetch("/api/login", {
    method: "POST", credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    throw new Error(j.error || `login failed (${r.status})`);
  }
}
export async function logout(): Promise<void> {
  await fetch("/api/logout", { method: "POST", credentials: "same-origin" });
}
