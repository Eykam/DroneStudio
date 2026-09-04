// DroneStudio auto-researcher dashboard - Hono on Bun.
// Single-user auth: password hash + HMAC session cookie, HTTPS-only.
// Ingest: bearer-token POSTs from the research box; state persisted to disk.
import { Hono } from "hono";
import { serveStatic } from "hono/bun";
import { getCookie, setCookie } from "hono/cookie";

const PASSWORD_SHA256 = process.env.DASHBOARD_PASSWORD_SHA256 || "";
const SESSION_SECRET = process.env.SESSION_SECRET || "";
const INGEST_TOKEN = process.env.INGEST_TOKEN || "";
const DATA_DIR = process.env.DATA_DIR || "/app/data";
const PORT = Number(process.env.PORT || 8080);
const SESSION_MAX_AGE_S = 7 * 24 * 3600;

if (!PASSWORD_SHA256 || !SESSION_SECRET || !INGEST_TOKEN) {
  console.error("FATAL: DASHBOARD_PASSWORD_SHA256, SESSION_SECRET, INGEST_TOKEN required");
  process.exit(1);
}

const STATE_FILE = `${DATA_DIR}/state.json`;

type ArchiveRecord = {
  id: string; parent: string | null; generation: number;
  params: Record<string, number>;
  metrics: Record<string, number | string>;
  mutator: string; ts: string; novelty?: number;
};
type State = {
  run: Record<string, unknown> | null;
  records: ArchiveRecord[];
  reports: Record<string, unknown>[];
  updated_at: string | null;
};

async function loadState(): Promise<State> {
  try {
    const f = Bun.file(STATE_FILE);
    if (await f.exists()) return await f.json();
  } catch {}
  return { run: null, records: [], reports: [], updated_at: null };
}

async function saveState(s: State) {
  const fs = await import("node:fs/promises");
  await fs.mkdir(DATA_DIR, { recursive: true });
  const tmp = STATE_FILE + ".tmp";
  await Bun.write(tmp, JSON.stringify(s));
  await fs.rename(tmp, STATE_FILE);
}

// --- crypto helpers (WebCrypto, timing-safe-ish) ---------------------------
async function sha256hex(s: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
async function hmac(value: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(SESSION_SECRET),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(value));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
function eqHex(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
}

async function makeSession(): Promise<string> {
  const ts = Math.floor(Date.now() / 1000).toString();
  return `${ts}.${await hmac("ds-dash:" + ts)}`;
}
async function checkSession(token: string | undefined): Promise<boolean> {
  if (!token) return false;
  const i = token.indexOf(".");
  if (i < 1) return false;
  const ts = token.slice(0, i), sig = token.slice(i + 1);
  if (!/^\d+$/.test(ts)) return false;
  const age = Math.floor(Date.now() / 1000) - Number(ts);
  if (age < 0 || age > SESSION_MAX_AGE_S) return false;
  return eqHex(sig, await hmac("ds-dash:" + ts));
}

// --- login rate limiting (in-memory, per IP) --------------------------------
const attempts = new Map<string, { n: number; reset: number }>();
function rateLimited(ip: string): boolean {
  const now = Date.now();
  const a = attempts.get(ip);
  if (!a || now > a.reset) { attempts.set(ip, { n: 1, reset: now + 300_000 }); return false; }
  a.n += 1;
  return a.n > 8;
}

const app = new Hono();

// --- auth endpoints ----------------------------------------------------------
app.post("/api/login", async (c) => {
  const ip = c.req.header("cf-connecting-ip") || c.req.header("x-forwarded-for") || "unknown";
  if (rateLimited(ip)) return c.json({ error: "too many attempts, wait 5 minutes" }, 429);
  let password = "";
  try { password = (await c.req.json()).password || ""; } catch { return c.json({ error: "bad request" }, 400); }
  const ok = eqHex(await sha256hex(password), PASSWORD_SHA256);
  if (!ok) return c.json({ error: "invalid password" }, 401);
  const token = await makeSession();
  setCookie(c, "ds_session", token, {
    httpOnly: true, secure: true, sameSite: "Lax", path: "/",
    maxAge: SESSION_MAX_AGE_S,
  });
  return c.json({ ok: true });
});

app.post("/api/logout", (c) => {
  setCookie(c, "ds_session", "", { httpOnly: true, secure: true, sameSite: "Lax", path: "/", maxAge: 0 });
  return c.json({ ok: true });
});

app.get("/api/me", async (c) => {
  return c.json({ authed: await checkSession(getCookie(c, "ds_session")) });
});

// --- ingest (bearer token, from the research box) -----------------------------
app.post("/api/ingest", async (c) => {
  const auth = c.req.header("authorization") || "";
  if (!eqHex(await sha256hex(auth.replace(/^Bearer /, "")), await sha256hex(INGEST_TOKEN))) {
    return c.json({ error: "unauthorized" }, 401);
  }
  let body: any;
  try { body = await c.req.json(); } catch { return c.json({ error: "bad request" }, 400); }
  const state = await loadState();
  if (body.run) state.run = body.run;
  if (Array.isArray(body.records)) {
    const have = new Set(state.records.map((r) => r.id));
    for (const r of body.records) if (r && r.id && !have.has(r.id)) state.records.push(r);
  }
  if (Array.isArray(body.reports)) {
    const seen = new Set(state.reports.map((r: any) => r.finished_at));
    for (const r of body.reports) if (r && r.finished_at && !seen.has(r.finished_at)) state.reports.push(r);
  }
  state.updated_at = new Date().toISOString();
  await saveState(state);
  return c.json({ ok: true, records: state.records.length });
});

// --- authed reads ---------------------------------------------------------------
app.use("/api/state", async (c, next) => {
  if (!(await checkSession(getCookie(c, "ds_session")))) return c.json({ error: "unauthorized" }, 401);
  await next();
});
app.get("/api/state", async (c) => c.json(await loadState()));

// --- static frontend ----------------------------------------------------------
app.use("/*", serveStatic({ root: "./dist" }));
app.get("*", serveStatic({ root: "./dist", path: "index.html" })); // SPA fallback

console.log(`dashboard listening on :${PORT}`);
export default { port: PORT, fetch: app.fetch };
