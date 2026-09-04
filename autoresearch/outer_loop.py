"""Outer loop: mutate scene distribution -> inner loop trains a policy under
it -> evaluator scores on held-out episodes -> archive -> select elite -> repeat."""
import json
import numpy as np
from scene_schema import SceneDistribution
from mutate import mutate
from evaluator import evaluate_distribution
from archive import Archive

def run(generations=1, children=2, seed=0, budget=None, verbose=True,
        archive_path="archive.jsonl", base=None):
    budget = budget or {}
    archive = Archive(archive_path)
    current = base or SceneDistribution()
    current_id = None

    for gen in range(generations):
        kids, mutator = mutate(current, children, seed + gen, archive.summary())
        cands = [("base", current)] + [(f"mut{i}", d) for i, d in enumerate(kids)]
        scored = []
        for tag, dist in cands:
            metrics = evaluate_distribution(
                dist,
                train_seed=seed + gen * 100,
                cem_iters=budget.get("cem_iters", 3),
                cem_pop=budget.get("cem_pop", 8),
                train_episodes=budget.get("train_episodes", 4),
                eval_episodes=budget.get("eval_episodes", 6),
                max_steps=budget.get("max_steps", 200),
                verbose=False)
            rec = archive.add(gen, current_id, dist, metrics, mutator)
            scored.append((rec, metrics))
            if verbose:
                print(f"gen {gen} {rec['id']} ({tag}, {mutator}): "
                      f"succ={metrics['success_rate']:.2f} ret={metrics['mean_return']:.1f}")
        best_rec, best_metrics = max(scored, key=lambda x: (x[1]["success_rate"], x[1]["mean_return"]))
        current = SceneDistribution.from_json(json.dumps(best_rec["params"]))
        current_id = best_rec["id"]
        if verbose:
            print(f"gen {gen} elite: {current_id} succ={best_metrics['success_rate']:.2f}")
    return archive.best()
