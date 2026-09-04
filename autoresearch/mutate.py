"""Mutation operators over scene distributions + the LLM mutator interface.

LLM access is env-var configured and OPTIONAL. With no key, the heuristic
mutator runs the loop; with a key, the outer loop asks the model to propose
variants grounded in the archive. Keys never touch the repo - environment only.

  AUTORESEARCH_LLM_API_KEY   - enables the LLM mutator when set
  AUTORESEARCH_LLM_BASE_URL  - OpenAI-compatible endpoint (default api.openai.com)
  AUTORESEARCH_LLM_MODEL     - model name (default gpt-4o-mini)
"""
import os, json, urllib.request
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

def mutate(base: SceneDistribution, k: int, seed: int, archive_summary: str = ""):
    """Outer-loop mutation entry point: LLM when configured, heuristic otherwise."""
    if os.environ.get("AUTORESEARCH_LLM_API_KEY"):
        try:
            return llm_mutants(base, k, seed, archive_summary), "llm"
        except Exception as e:
            print(f"llm mutator failed ({e}); falling back to heuristic")
    return heuristic_mutants(base, k, seed), "heuristic"
