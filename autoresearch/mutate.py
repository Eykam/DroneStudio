"""Mutation operators over scene distributions + the LLM mutator interface.

LLM access is env-var configured and OPTIONAL. With no key and no Codex auth,
the heuristic mutator runs the loop; otherwise the outer loop asks the model
to propose variants grounded in the archive. Keys never touch the repo.

Backend selection (AUTORESEARCH_LLM_BACKEND):
  auto      - codex if the Codex CLI is authenticated, else OpenAI-compatible
              API if a key is set, else heuristic (default)
  codex     - Codex CLI only (ChatGPT subscription; no per-call billing)
  openai    - OpenAI-compatible HTTP API only (per-token billing, capped)
  heuristic - no LLM

  AUTORESEARCH_LLM_API_KEY   - enables the OpenAI-compatible mutator when set
  AUTORESEARCH_LLM_BASE_URL  - OpenAI-compatible endpoint (default api.openai.com)
  AUTORESEARCH_LLM_MODEL     - model name (default gpt-4o-mini)

Codex backend (runs against a ChatGPT subscription, not metered API):
  AUTORESEARCH_CODEX_BIN            - codex binary path (default /workspace/bin/codex)
  CODEX_HOME                        - codex config/auth dir (default /workspace/codex-home)
  AUTORESEARCH_CODEX_MODEL          - optional model override (codex -m)
  AUTORESEARCH_CODEX_MIN_INTERVAL_S - min seconds between calls (default 45)
  AUTORESEARCH_CODEX_MAX_CALLS      - per-state-file call budget (default 100)
Throttling exists because subscription rate limits are tuned for interactive
use; when the throttle blocks a call the loop falls back, never queues.
"""
import os, json, time, subprocess, urllib.request
import numpy as np
from scene_schema import SceneDistribution

def heuristic_mutants(base: SceneDistribution, k: int, seed: int, sigma=0.15):
    rng = np.random.default_rng(seed)
    names = SceneDistribution.names()
    out = []
    for _ in range(k):
        vec = np.array(base.to_vector())
        mask = rng.random(len(vec)) < 0.5
        vec = vec + mask * rng.normal(0, sigma, len(vec)) * np.abs(vec + 0.1)
        if rng.random() < 0.3:  # full resample of one dimension
            i = int(rng.integers(len(vec)))
            lo, hi = SceneDistribution.BOUNDS[names[i]]
            vec[i] = rng.uniform(lo, hi)
        out.append(SceneDistribution.from_vector(vec))
    return out

# --- API spend guard -------------------------------------------------------
# Hard cap on outer-loop LLM spend, default $10. Persists across runs in a
# state file next to the archive. Over cap -> heuristic mutator, always.
SPEND_CAP_USD = float(os.environ.get("AUTORESEARCH_LLM_SPEND_CAP_USD", "10"))
SPEND_FILE = os.environ.get("AUTORESEARCH_SPEND_FILE",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".llm_spend.json"))

# USD per 1M tokens (input, output); extend as providers/models are added.
PRICE_TABLE = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
}

def _spend_state():
    try:
        with open(SPEND_FILE) as f:
            return json.load(f)
    except Exception:
        return {"usd": 0.0, "calls": 0}

def _record_spend(model, usage):
    pin, pout = PRICE_TABLE.get(model, (2.50, 10.00))  # assume pricey when unknown
    cost = (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6
    st = _spend_state()
    st["usd"] = round(st["usd"] + cost, 6)
    st["calls"] += 1
    with open(SPEND_FILE, "w") as f:
        json.dump(st, f)
    return st

def spend_status():
    st = _spend_state()
    return {"spent_usd": st["usd"], "cap_usd": SPEND_CAP_USD, "calls": st["calls"],
            "over_cap": st["usd"] >= SPEND_CAP_USD}

def llm_mutants(base: SceneDistribution, k: int, seed: int, archive_summary: str):
    key = os.environ.get("AUTORESEARCH_LLM_API_KEY")
    if not key:
        raise RuntimeError("AUTORESEARCH_LLM_API_KEY not set")
    base_url = os.environ.get("AUTORESEARCH_LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("AUTORESEARCH_LLM_MODEL", "gpt-4o-mini")
    prompt = (
        "You are the outer loop of an auto-researcher that optimizes a drone "
        "simulator's procedural scene distribution to train better navigation "
        "policies. Propose %d mutated scene distributions as a JSON array of "
        "objects with exactly these keys: %s. Base distribution: %s. "
        "Archive so far: %s. Reply with JSON only."
        % (k, ", ".join(SceneDistribution.names()), base.to_json(), archive_summary))
    if _spend_state()["usd"] >= SPEND_CAP_USD:
        raise RuntimeError(f"LLM spend cap reached (${SPEND_CAP_USD}); staying heuristic")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9,
        }).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read())
    _record_spend(model, body.get("usage", {}))
    text = body["choices"][0]["message"]["content"]
    start, end = text.find("["), text.rfind("]")
    variants = json.loads(text[start:end + 1])
    return [SceneDistribution.from_json(json.dumps(v)) for v in variants[:k]]

# --- Codex CLI backend (ChatGPT subscription) ------------------------------
CODEX_BIN = os.environ.get("AUTORESEARCH_CODEX_BIN", "/workspace/bin/codex")
CODEX_HOME_DIR = os.environ.get("CODEX_HOME", "/workspace/codex-home")
CODEX_MODEL = os.environ.get("AUTORESEARCH_CODEX_MODEL", "")
CODEX_MIN_INTERVAL_S = float(os.environ.get("AUTORESEARCH_CODEX_MIN_INTERVAL_S", "45"))
CODEX_MAX_CALLS = int(os.environ.get("AUTORESEARCH_CODEX_MAX_CALLS", "100"))
CODEX_STATE_FILE = os.environ.get(
    "AUTORESEARCH_CODEX_STATE_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".codex_usage.json"))

def codex_available():
    return (os.path.exists(CODEX_BIN)
            and os.path.exists(os.path.join(CODEX_HOME_DIR, "auth.json")))

def _codex_state():
    try:
        with open(CODEX_STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"calls": 0, "last_ts": 0.0}

def _record_codex_call():
    st = _codex_state()
    st["calls"] += 1
    st["last_ts"] = time.time()
    with open(CODEX_STATE_FILE, "w") as f:
        json.dump(st, f)

def codex_status():
    st = _codex_state()
    return {"available": codex_available(), "calls": st["calls"],
            "max_calls": CODEX_MAX_CALLS,
            "min_interval_s": CODEX_MIN_INTERVAL_S,
            "seconds_since_last": round(time.time() - st["last_ts"], 1) if st["last_ts"] else None}

def codex_mutants(base: SceneDistribution, k: int, seed: int, archive_summary: str):
    if not codex_available():
        raise RuntimeError("codex CLI not authenticated (no auth.json)")
    st = _codex_state()
    if st["calls"] >= CODEX_MAX_CALLS:
        raise RuntimeError(f"codex call budget exhausted ({CODEX_MAX_CALLS})")
    since = time.time() - st["last_ts"] if st["last_ts"] else 1e18
    if since < CODEX_MIN_INTERVAL_S:
        raise RuntimeError(f"codex throttle: {CODEX_MIN_INTERVAL_S - since:.0f}s until next allowed call")
    prompt = (
        "You are the outer loop of an auto-researcher that optimizes a drone "
        "simulator's procedural scene distribution to train better navigation "
        "policies. Do not use tools, do not read or write files, do not run "
        "commands - just answer with JSON. Propose %d mutated scene "
        "distributions as a JSON array of objects with exactly these keys: %s. "
        "Respect these numeric bounds: %s. Base distribution: %s. "
        "Archive so far: %s. Your final message must be the JSON array only, "
        "no markdown fences, no commentary."
        % (k, ", ".join(SceneDistribution.names()),
           json.dumps(SceneDistribution.BOUNDS), base.to_json(), archive_summary))
    out_file = CODEX_STATE_FILE + ".last_message"
    cmd = [CODEX_BIN, "exec", "--sandbox", "read-only", "--skip-git-repo-check",
           "--ephemeral", "--color", "never", "--ignore-user-config",
           "-o", out_file, "-"]
    if CODEX_MODEL:
        cmd[2:2] = ["-m", CODEX_MODEL]
    env = dict(os.environ, CODEX_HOME=CODEX_HOME_DIR)
    proc = subprocess.run(cmd, input=prompt.encode(), capture_output=True,
                          timeout=600, env=env)
    _record_codex_call()
    if proc.returncode != 0:
        raise RuntimeError("codex exec failed: " + proc.stderr.decode()[-400:])
    with open(out_file) as f:
        text = f.read().strip()
    start, end = text.find("["), text.rfind("]")
    variants = json.loads(text[start:end + 1])
    return [SceneDistribution.from_json(json.dumps(v)) for v in variants[:k]]

def mutate(base: SceneDistribution, k: int, seed: int, archive_summary: str = ""):
    """Outer-loop mutation entry point: configured backend, heuristic fallback."""
    backend = os.environ.get("AUTORESEARCH_LLM_BACKEND", "auto").lower()
    order = {"auto": ["codex", "openai"], "codex": ["codex"],
             "openai": ["openai"], "heuristic": []}.get(backend, ["codex", "openai"])
    for b in order:
        if b == "codex" and codex_available():
            try:
                return codex_mutants(base, k, seed, archive_summary), "codex"
            except Exception as e:
                print(f"codex mutator failed ({e}); trying next backend")
        if b == "openai" and os.environ.get("AUTORESEARCH_LLM_API_KEY"):
            try:
                return llm_mutants(base, k, seed, archive_summary), "llm"
            except Exception as e:
                print(f"llm mutator failed ({e}); falling back to heuristic")
    return heuristic_mutants(base, k, seed), "heuristic"
