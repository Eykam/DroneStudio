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
