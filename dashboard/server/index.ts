// DroneStudio auto-researcher dashboard - Hono on Bun.
// Single-user auth: password hash + HMAC session cookie, HTTPS-only.
// Ingest: bearer-token POSTs from the research box; state persisted to disk.
import { Hono } from "hono";
import { serveStatic } from "hono/bun";
import { streamSSE } from "hono/streaming";
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


// --- CAD / Mechanicals ------------------------------------------------------
// Contract: dashboard/CAD_INGEST_API.md (read by the CAD researcher agent).
type CadMetrics = {
  mass_g?: number;
  inertia?: { ixx?: number; iyy?: number; izz?: number };
  printability?: Record<string, unknown>;
  fea?: Record<string, unknown>;
  [k: string]: unknown;
};
type CadDesign = {
  id: string;
  name?: string;
  parent_id: string | null;
  created_at: string;
  source?: string;
  metrics: CadMetrics;
  notes?: string;
  glb_bytes: number;
};
const CAD_DIR = `${DATA_DIR}/cad`;
const CAD_FILE = `${CAD_DIR}/designs.json`;

async function loadDesigns(): Promise<CadDesign[]> {
  try {
    const f = Bun.file(CAD_FILE);
    if (await f.exists()) return await f.json();
  } catch {}
  return [];
}

async function saveDesigns(d: CadDesign[]) {
  const fs = await import("node:fs/promises");
  await fs.mkdir(CAD_DIR, { recursive: true });
  const tmp = CAD_FILE + ".tmp";
  await Bun.write(tmp, JSON.stringify(d));
  await fs.rename(tmp, CAD_FILE);
}

function safeId(id: string): string | null {
  return /^[A-Za-z0-9._-]{1,80}$/.test(id) ? id : null;
}

app.post("/api/cad/designs", async (c) => {
  const auth = c.req.header("authorization") || "";
  if (!eqHex(await sha256hex(auth.replace(/^Bearer /, "")), await sha256hex(INGEST_TOKEN))) {
    return c.json({ error: "unauthorized" }, 401);
  }
  let form: any;
  try { form = await c.req.parseBody(); } catch { return c.json({ error: "expected multipart/form-data" }, 400); }
  const file = form["file"];
  if (!(file instanceof File)) return c.json({ error: "missing GLB in 'file' field" }, 400);
  if (file.size > 64 * 1024 * 1024) return c.json({ error: "GLB over 64MB limit" }, 400);
  let meta: any;
  try { meta = JSON.parse(String(form["meta"] || "{}")); } catch { return c.json({ error: "bad JSON in 'meta' field" }, 400); }
  const id = safeId(String(meta.id || ""));
  if (!id) return c.json({ error: "meta.id required (A-Za-z0-9._-, max 80)" }, 400);
  const designs = await loadDesigns();
  const existing = designs.findIndex((d) => d.id === id);
  const rec: CadDesign = {
    id,
    name: typeof meta.name === "string" ? meta.name : id,
    parent_id: typeof meta.parent_id === "string" ? meta.parent_id : null,
    created_at: typeof meta.created_at === "string" ? meta.created_at : new Date().toISOString(),
    source: typeof meta.source === "string" ? meta.source : undefined,
    metrics: (meta.metrics && typeof meta.metrics === "object") ? meta.metrics : {},
    notes: typeof meta.notes === "string" ? meta.notes : undefined,
    glb_bytes: file.size,
  };
  const fs = await import("node:fs/promises");
  await fs.mkdir(CAD_DIR, { recursive: true });
  await Bun.write(`${CAD_DIR}/${id}.glb`, file);
  if (existing >= 0) {
    rec.created_at = designs[existing].created_at; // keep original timestamp on update
    designs[existing] = rec;
  } else {
    designs.push(rec);
  }
  await saveDesigns(designs);
  return c.json({ ok: true, id, glb_bytes: file.size });
});

const cadAuthed = async (c: any, next: any) => {
  if (!(await checkSession(getCookie(c, "ds_session")))) return c.json({ error: "unauthorized" }, 401);
  await next();
};
app.use("/api/cad/designs", cadAuthed);
app.use("/api/cad/designs/:id/glb", cadAuthed);

app.get("/api/cad/designs", async (c) => {
  const designs = await loadDesigns();
  return c.json({
    designs: designs.map((d) => ({ ...d, glb_url: `/api/cad/designs/${d.id}/glb` })),
    updated_at: new Date().toISOString(),
  });
});

app.get("/api/cad/designs/:id/glb", async (c) => {
  const id = safeId(c.req.param("id"));
  if (!id) return c.json({ error: "bad id" }, 400);
  const f = Bun.file(`${CAD_DIR}/${id}.glb`);
  if (!(await f.exists())) return c.json({ error: "not found" }, 404);
  return new Response(f, { headers: { "content-type": "model/gltf-binary", "cache-control": "no-store" } });
});

// --- authed reads ---------------------------------------------------------------

app.use("/api/state", async (c, next) => {
  if (!(await checkSession(getCookie(c, "ds_session")))) return c.json({ error: "unauthorized" }, 401);
  await next();
});
app.get("/api/state", async (c) => c.json(await loadState()));

// --- static frontend ----------------------------------------------------------

// --- live sim stream (watch channel) ---------------------------------------
// The streamer on the research box POSTs scene/frame/episode_end events with
// the ingest bearer token; viewers subscribe over SSE. In-memory only: the
// stream is live, not history.
const FRAME_CAP = 400;
const streamState: {
  meta: Record<string, unknown>;
  scene: Record<string, unknown> | null;
  frames: Record<string, unknown>[];
  last_event_at: string | null;
} = { meta: { status: "offline" }, scene: null, frames: [], last_event_at: null };

type SseClient = { send: (event: string, data: unknown) => void };
const sseClients = new Set<SseClient>();
function broadcast(event: string, data: unknown) {
  for (const cl of sseClients) {
    try { cl.send(event, data); } catch { sseClients.delete(cl); }
  }
}

app.post("/api/stream/ingest", async (c) => {
  const auth = c.req.header("authorization") || "";
  if (!eqHex(await sha256hex(auth.replace(/^Bearer /, "")), await sha256hex(INGEST_TOKEN))) {
    return c.json({ error: "unauthorized" }, 401);
  }
  let body: any;
  try { body = await c.req.json(); } catch { return c.json({ error: "bad request" }, 400); }
  const now = new Date().toISOString();
  streamState.last_event_at = now;
  const t = body.type;
  if (t === "status") {
    const { type, ...rest } = body;
    streamState.meta = { ...rest, ts: now };
    broadcast("status", streamState.meta);
  } else if (t === "scene") {
    const { type, ...rest } = body;
    streamState.scene = { ...rest, ts: now };
    streamState.frames = [];
    broadcast("scene", streamState.scene);
  } else if (t === "frame") {
    const f = { ...body, ts: now };
    streamState.frames.push(f);
    if (streamState.frames.length > FRAME_CAP) streamState.frames.splice(0, streamState.frames.length - FRAME_CAP);
    broadcast("frame", f);
  } else if (t === "episode_end") {
    broadcast("episode_end", body);
  } else {
    return c.json({ error: "unknown type" }, 400);
  }
  return c.json({ ok: true });
});


// --- CAD work-in-progress signal --------------------------------------------
// The CAD research loop POSTs its current stage here; /cad renders an
// in-progress banner. In-memory only, staleness handled client-side.
// Contract documented in CAD_INGEST_API.md.
let cadProgress: Record<string, unknown> = { status: "idle" };

app.post("/api/cad/progress", async (c) => {
  const auth = c.req.header("authorization") || "";
  if (!eqHex(await sha256hex(auth.replace(/^Bearer /, "")), await sha256hex(INGEST_TOKEN))) {
    return c.json({ error: "unauthorized" }, 401);
  }
  let body: any;
  try { body = await c.req.json(); } catch { return c.json({ error: "bad request" }, 400); }
  // Ignore empty/malformed bodies so probes cannot wipe the live banner state.
  if (!body || typeof body !== "object" || Array.isArray(body) || typeof body.status !== "string") {
    return c.json({ error: "bad request: body must be an object with a status field" }, 400);
  }
  cadProgress = { ...body, ts: new Date().toISOString() };
  return c.json({ ok: true });
});

app.get("/api/cad/progress", (c) => c.json(cadProgress));

// Curriculum ladder state: the manual curriculum experiments (and later the
// auto-ratchet runner) POST stage results here so the Research page reflects
// the actual current training state, not just outer-loop generations.
type CurriculumStage = {
  goal_m: number; trainer: string; success_rate: number;
  mean_return?: number; mean_steps?: number; wall_s?: number;
  eval_episodes?: number; budget?: string; ts?: string;
};
let curriculum: { status: string; current_stage?: unknown; note?: string; stages: CurriculumStage[]; ts: string } =
  { status: "idle", stages: [], ts: new Date().toISOString() };

app.post("/api/curriculum/progress", async (c) => {
  const auth = c.req.header("authorization") || "";
  if (!eqHex(await sha256hex(auth.replace(/^Bearer /, "")), await sha256hex(INGEST_TOKEN))) {
    return c.json({ error: "unauthorized" }, 401);
  }
  let body: any;
  try { body = await c.req.json(); } catch { return c.json({ error: "bad request" }, 400); }
  if (!body || typeof body !== "object" || Array.isArray(body) || typeof body.status !== "string") {
    return c.json({ error: "bad request: body must be an object with a status field" }, 400);
  }
  if (body.reset === true) curriculum.stages = [];
  const sr = body.stage_result;
  if (sr && typeof sr === "object" && typeof sr.goal_m === "number" && typeof sr.success_rate === "number") {
    curriculum.stages.push({ ...sr, ts: new Date().toISOString() });
  }
  curriculum = {
    ...curriculum,
    status: body.status,
    current_stage: body.current_stage ?? curriculum.current_stage,
    note: typeof body.note === "string" ? body.note : curriculum.note,
    ts: new Date().toISOString(),
  };
  return c.json({ ok: true, stages: curriculum.stages.length });
});

app.get("/api/curriculum/progress", (c) => c.json(curriculum));

app.get("/api/stream/state", (c) => c.json({
  meta: streamState.meta,
  scene: streamState.scene,
  frames: streamState.frames,
  last_event_at: streamState.last_event_at,
}));

app.get("/api/stream", (c) =>
  streamSSE(c, async (stream) => {
    const client: SseClient = {
      send: (event, data) => {
        void stream.writeSSE({ event, data: JSON.stringify(data) }).catch(() => sseClients.delete(client));
      },
    };
    sseClients.add(client);
    await stream.writeSSE({
      event: "snapshot",
      data: JSON.stringify({ meta: streamState.meta, scene: streamState.scene, frames: streamState.frames }),
    });
    const keepalive = setInterval(() => {
      void stream.writeSSE({ event: "ping", data: "{}" }).catch(() => {});
    }, 15000);
    stream.onAbort(() => { clearInterval(keepalive); sseClients.delete(client); });
    await new Promise<void>(() => {});
  })
);

app.use("/*", serveStatic({ root: "./dist" }));
app.get("*", serveStatic({ root: "./dist", path: "index.html" })); // SPA fallback

console.log(`dashboard listening on :${PORT}`);
export default { port: PORT, fetch: app.fetch };
