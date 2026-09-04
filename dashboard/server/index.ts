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
  // Gate auto-publish to adopted-design ids; test/debug runs must not land here.
  if (!/^cad-chassis-v\d+-g\d+[a-c]?$/.test(id)) {
    return c.json({ error: "id must match cad-chassis-v<N>-g<M>[a-c] (test/debug artifacts are not publishable)" }, 400);
  }
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


app.delete("/api/cad/designs/:id", async (c) => {
  const auth = c.req.header("authorization") || "";
  if (!eqHex(await sha256hex(auth.replace(/^Bearer /, "")), await sha256hex(INGEST_TOKEN))) {
    return c.json({ error: "unauthorized" }, 401);
  }
  const id = safeId(c.req.param("id"));
  if (!id) return c.json({ error: "bad id" }, 400);
  const designs = await loadDesigns();
  const next = designs.filter((d) => d.id !== id);
  if (next.length === designs.length) return c.json({ error: "not found" }, 404);
  await saveDesigns(next);
  try {
    const fs = await import("node:fs/promises");
    await fs.unlink(`${CAD_DIR}/${id}.glb`);
  } catch {}
  return c.json({ ok: true, id });
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
  // Design ids are unique per adopted design, so responses are cacheable.
  const raw = new Uint8Array(await f.arrayBuffer());
  const headers: Record<string, string> = {
    "content-type": "model/gltf-binary",
    "cache-control": "public, max-age=86400, immutable",
    vary: "accept-encoding",
  };
  const ae = c.req.header("accept-encoding") || "";
  const zlib = await import("node:zlib");
  if (/\bbr\b/.test(ae)) {
    headers["content-encoding"] = "br";
    return new Response(zlib.brotliCompressSync(raw, { params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 6 } }), { headers });
  }
  if (/\bgzip\b/.test(ae)) {
    headers["content-encoding"] = "gzip";
    return new Response(zlib.gzipSync(raw, { level: 6 }), { headers });
  }
  return new Response(raw, { headers });
});


// --- training status (live policy / candidate / training now / queue) ------
// The research box POSTs a free-form status doc; the Research page renders it
// as the top "model status" panel. Persisted so redeploys do not lose it.
type TrainingStatus = Record<string, unknown>;
const TRAINING_FILE = `${DATA_DIR}/training_status.json`;

async function loadTrainingStatus(): Promise<TrainingStatus> {
  try {
    const f = Bun.file(TRAINING_FILE);
    if (await f.exists()) return await f.json();
  } catch {}
  return { status: "idle" };
}

async function saveTrainingStatus(s: TrainingStatus) {
  const fs = await import("node:fs/promises");
  await fs.mkdir(DATA_DIR, { recursive: true });
  const tmp = TRAINING_FILE + ".tmp";
  await Bun.write(tmp, JSON.stringify(s));
  await fs.rename(tmp, TRAINING_FILE);
}

app.post("/api/training/status", async (c) => {
  const auth = c.req.header("authorization") || "";
  if (!eqHex(await sha256hex(auth.replace(/^Bearer /, "")), await sha256hex(INGEST_TOKEN))) {
    return c.json({ error: "unauthorized" }, 401);
  }
  let body: any;
  try { body = await c.req.json(); } catch { return c.json({ error: "bad request" }, 400); }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return c.json({ error: "bad request: body must be a JSON object" }, 400);
  }
  await saveTrainingStatus({ ...body, updated_at: new Date().toISOString() });
  return c.json({ ok: true });
});

app.get("/api/training/status", async (c) => {
  if (!(await checkSession(getCookie(c, "ds_session")))) return c.json({ error: "unauthorized" }, 401);
  return c.json(await loadTrainingStatus());
});

// --- authed reads ---------------------------------------------------------------

app.use("/api/state", async (c, next) => {
  if (!(await checkSession(getCookie(c, "ds_session")))) return c.json({ error: "unauthorized" }, 401);
  await next();
});
app.get("/api/state", async (c) => c.json(await loadState()));

// --- static frontend ----------------------------------------------------------

// --- live sim stream (watch channel) ---------------------------------------
// Streamers on the research box POST scene/frame/episode_end events with the
// ingest bearer token; viewers subscribe over SSE. Multi-source: every payload
// may carry "source" (default "default"); state is kept per source so /watch
// can show several concurrent experiments side by side. The sources registry
// (POST /api/stream/sources) lists streamable experiments and is persisted so
// redeploys keep the picker. Streams are live, not history.
const FRAME_CAP = 400;
type StreamChan = {
  meta: Record<string, unknown>;
  scene: Record<string, unknown> | null;
  frames: Record<string, unknown>[];
  last_event_at: string | null;
};
const streams = new Map<string, StreamChan>();
const SRC_ID = /^[a-z0-9][a-z0-9-]{0,31}$/;
function chan(id: string): StreamChan {
  let ch = streams.get(id);
  if (!ch) {
    if (streams.size >= 8) throw new Error("too many stream sources");
    ch = { meta: { status: "offline" }, scene: null, frames: [], last_event_at: null };
    streams.set(id, ch);
  }
  return ch;
}
function streamsSnapshot() {
  return Object.fromEntries([...streams].map(([k, v]) => [k, v]));
}

// Registry of streamable experiments (the /watch picker). The research box
// upserts one entry per experiment/hypothesis it can stream.
const STREAM_SOURCES_FILE = `${DATA_DIR}/stream_sources.json`;
let streamSources: Record<string, unknown>[] = await (async () => {
  try {
    const f = Bun.file(STREAM_SOURCES_FILE);
    if (await f.exists()) return await f.json();
  } catch {}
  return [];
})();

async function saveStreamSources() {
  const fs = await import("node:fs/promises");
  await fs.mkdir(DATA_DIR, { recursive: true });
  const tmp = STREAM_SOURCES_FILE + ".tmp";
  await Bun.write(tmp, JSON.stringify(streamSources));
  await fs.rename(tmp, STREAM_SOURCES_FILE);
}

app.post("/api/stream/sources", async (c) => {
  const auth = c.req.header("authorization") || "";
  if (!eqHex(await sha256hex(auth.replace(/^Bearer /, "")), await sha256hex(INGEST_TOKEN))) {
    return c.json({ error: "unauthorized" }, 401);
  }
  let body: any;
  try { body = await c.req.json(); } catch { return c.json({ error: "bad request" }, 400); }
  if (!body || typeof body !== "object" || Array.isArray(body)
      || typeof body.id !== "string" || !SRC_ID.test(body.id)) {
    return c.json({ error: "bad request: {id: [a-z0-9-], label?, policy?, policy_obs?, eval?, status?, note?}" }, 400);
  }
  const { id, remove, ...fields } = body;
  const i = streamSources.findIndex((s) => s.id === id);
  if (remove === true) {
    if (i >= 0) streamSources.splice(i, 1);
  } else {
    const entry = { ...(i >= 0 ? streamSources[i] : {}), ...fields, id, updated_at: new Date().toISOString() };
    if (i >= 0) streamSources[i] = entry; else streamSources.push(entry);
  }
  await saveStreamSources();
  return c.json({ ok: true, n: streamSources.length });
});

app.get("/api/stream/sources", (c) => c.json(streamSources));

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
  const src = typeof body.source === "string" && SRC_ID.test(body.source) ? body.source : "default";
  const st = chan(src);
  const now = new Date().toISOString();
  st.last_event_at = now;
  const t = body.type;
  if (t === "status") {
    const { type, source, ...rest } = body;
    st.meta = { ...rest, ts: now };
    broadcast("status", { source: src, ...st.meta });
  } else if (t === "scene") {
    const { type, source, ...rest } = body;
    st.scene = { ...rest, ts: now };
    st.frames = [];
    broadcast("scene", { source: src, ...st.scene });
  } else if (t === "frame") {
    const f = { ...body, source: src, ts: now };
    st.frames.push(f);
    if (st.frames.length > FRAME_CAP) st.frames.splice(0, st.frames.length - FRAME_CAP);
    broadcast("frame", f);
  } else if (t === "episode_end") {
    broadcast("episode_end", { ...body, source: src });
  } else {
    return c.json({ error: "unknown type" }, 400);
  }
  return c.json({ ok: true });
});

app.get("/api/stream/state", (c) => c.json({
  sources: streamsSnapshot(),
  registry: streamSources,
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
      data: JSON.stringify({ sources: streamsSnapshot(), registry: streamSources }),
    });
    const keepalive = setInterval(() => {
      void stream.writeSSE({ event: "ping", data: "{}" }).catch(() => {});
    }, 15000);
    stream.onAbort(() => { clearInterval(keepalive); sseClients.delete(client); });
    await new Promise<void>(() => {});
  })
);

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

// Named time series for training curves (curriculum success over time,
// BC/DAgger loss over time, ...). Producers POST one point at a time;
// the Research page main chart toggles between these and generations.
type SeriesPoint = { t: string; y: number; label?: string };
const SERIES_FILE = `${DATA_DIR}/series.json`;
const seriesStore: Record<string, SeriesPoint[]> = await (async () => {
  try {
    const f = Bun.file(SERIES_FILE);
    if (await f.exists()) return await f.json();
  } catch {}
  return {};
})();
const SERIES_CAP = 1000;

async function saveSeries() {
  const fs = await import("node:fs/promises");
  await fs.mkdir(DATA_DIR, { recursive: true });
  const tmp = SERIES_FILE + ".tmp";
  await Bun.write(tmp, JSON.stringify(seriesStore));
  await fs.rename(tmp, SERIES_FILE);
}

app.post("/api/series", async (c) => {
  const auth = c.req.header("authorization") || "";
  if (!eqHex(await sha256hex(auth.replace(/^Bearer /, "")), await sha256hex(INGEST_TOKEN))) {
    return c.json({ error: "unauthorized" }, 401);
  }
  let body: any;
  try { body = await c.req.json(); } catch { return c.json({ error: "bad request" }, 400); }
  if (!body || typeof body !== "object" || typeof body.series !== "string"
      || !body.point || typeof body.point.y !== "number") {
    return c.json({ error: "bad request: {series, point:{y, label?}, reset?}" }, 400);
  }
  if (body.reset === true) seriesStore[body.series] = [];
  const arr = (seriesStore[body.series] ||= []);
  arr.push({ t: typeof body.point.t === "string" ? body.point.t : new Date().toISOString(),
             y: body.point.y,
             label: typeof body.point.label === "string" ? body.point.label : undefined });
  if (arr.length > SERIES_CAP) arr.splice(0, arr.length - SERIES_CAP);
  await saveSeries();
  return c.json({ ok: true, n: arr.length });
});

app.get("/api/series", (c) => c.json(seriesStore));

app.use("/*", serveStatic({ root: "./dist" }));
app.get("*", serveStatic({ root: "./dist", path: "index.html" })); // SPA fallback

console.log(`dashboard listening on :${PORT}`);
export default { port: PORT, fetch: app.fetch };
