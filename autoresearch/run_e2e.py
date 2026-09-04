#!/usr/bin/env python3
"""Single end-to-end cycle of the auto-researcher skeleton.

Runs ONE generation: base scene distribution + 2 mutants, each scored by
training a tiny MLP nav policy under it (CEM inner loop) and measuring
held-out success. Verifies: mutation -> train -> evaluate -> archive ->
selection all execute and produce sane numbers.

WHAT THIS PROVES: the loop machinery closes and runs to budget on CPU.
WHAT IT DOES NOT PROVE: sim fidelity (stub dynamics, not DroneStudio physics/
rendering), RL algorithm quality (CEM, not PPO), LLM outer loop (heuristic
unless AUTORESEARCH_LLM_API_KEY is set), VIO/fast-loop integration.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outer_loop import run

def main():
    quick = "--quick" in sys.argv
    budget = dict(cem_iters=3, cem_pop=8, train_episodes=4, eval_episodes=4, max_steps=150) if quick \
        else dict(cem_iters=3, cem_pop=8, train_episodes=4, eval_episodes=6, max_steps=200)
    best = run(generations=1, children=2, seed=42, budget=budget,
               archive_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive.jsonl"))
    assert best is not None, "archive empty after one generation"
    m = best["metrics"]
    assert 0.0 <= m["success_rate"] <= 1.0, "success rate out of range"
    assert all(v == v for v in m.values()), "NaN in metrics"  # NaN check
    print("\nE2E_CYCLE_OK")
    print(json.dumps({"elite": best["id"], "metrics": m,
                      "elite_params": best["params"]}, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
