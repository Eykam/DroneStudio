"""Outer loop: mutate scene distribution -> inner loop trains a policy under
it -> evaluator scores on held-out episodes -> archive -> select -> repeat.

Selection strategy (v2):
  - Multi-elite parents: the top-2 distinct archived variants are mutated
    each generation instead of single-lineage hill climbing.
  - Novelty bonus: a candidate's selection score pays a small bonus for
    distance from everything already archived (normalized parameter
    space), countering premature convergence.
  - Stagnation restart: if the elite success rate has not improved for
    `stagnation_limit` generations, one child slot is a full random
    resample inside bounds (exploration reset).
All three behaviors are documented in the night report and configurable
via run() kwargs.
"""
import json
import numpy as np
from scene_schema import SceneDistribution
from mutate import mutate
from evaluator import evaluate_distribution

def _default_backend():
    """Mandate: the inner loop runs on the actual Zig sim binary when it is
    present; QuadNavEnv is the debug/fallback env. Override with backend=."""
    import os
    try:
        from env_sim import BINARY
        if os.path.exists(BINARY):
            return "sim"
    except Exception:
        pass
    return "quad"
from archive import Archive

def run(generations=1, children=2, seed=0, budget=None, verbose=True,
        archive_path="archive.jsonl", base=None, elite_k=2,
        novelty_weight=0.05, stagnation_limit=3, backend=None,
        dynamics=None,
        trainer="cem", ppo_config=None, n_jobs=None,
        plateau_min_improvement=0.02):
    budget = budget or {}
    rng = np.random.default_rng(seed)
    archive = Archive(archive_path)
    elites = [base] if base is not None else [SceneDistribution()]
    elite_ids = [None]
    best_succ = -1.0
    stagnant = 0
    pre_restart_best = None  # adaptive stopping: best success at last restart

    for gen in range(generations):
        # one mutation batch per elite parent
        cands = []
        for ei, parent in enumerate(elites):
            k = max(1, children // len(elites))
            kids, mutator = mutate(parent, k, seed + gen * 31 + ei, archive.summary())
            for d in kids:
                cands.append((f"elite{ei}", parent, elite_ids[ei], d, mutator))
            cands.append(("base", parent, elite_ids[ei], parent, "none"))

        # stagnation restart: replace one child with a full random resample.
        # Adaptive stop: if a restart already happened and the best success
        # since has not beaten the pre-restart best by
        # plateau_min_improvement, the search has plateaued - stop instead
        # of restarting again.  stays the upper bound.
        if stagnant >= stagnation_limit and cands:
            if (pre_restart_best is not None
                    and best_succ < pre_restart_best + plateau_min_improvement):
                if verbose:
                    print(f"gen {gen}: plateau stop - post-restart best "
                          f"{best_succ:.3f} < pre-restart {pre_restart_best:.3f} "
                          f"+ {plateau_min_improvement:.3f}", flush=True)
                break
            pre_restart_best = best_succ
            names = SceneDistribution.names()
            vec = [rng.uniform(*SceneDistribution.BOUNDS[n]) for n in names]
            cands[-1] = ("restart", elites[0], elite_ids[0],
                         SceneDistribution.from_vector(vec), "restart")
            stagnant = 0

        scored = []
        for tag, parent, parent_id, dist, mut in cands:
            metrics = evaluate_distribution(
                dist,
                train_seed=seed + gen * 100,
                cem_iters=budget.get("cem_iters", 3),
                cem_pop=budget.get("cem_pop", 8),
                train_episodes=budget.get("train_episodes", 4),
                eval_episodes=budget.get("eval_episodes", 6),
                max_steps=budget.get("max_steps", 200),
                verbose=False, backend=backend or _default_backend(),
                trainer=trainer, ppo_config=ppo_config, dynamics=dynamics, n_jobs=n_jobs)
            nov = archive.novelty(dist)
            rec = archive.add(gen, parent_id, dist, metrics, mut, novelty=nov)
            sel_score = metrics["success_rate"] + novelty_weight * nov
            scored.append((rec, metrics, sel_score))
            if verbose:
                print(f"gen {gen} {rec['id']} ({tag}, {mut}): "
                      f"succ={metrics['success_rate']:.2f} ret={metrics['mean_return']:.1f} "
                      f"nov={nov:.3f} sel={sel_score:.3f}", flush=True)

        # multi-elite update: dedup top-k by params
        scored.sort(key=lambda x: (x[2], x[1]["mean_return"]), reverse=True)
        elites, elite_ids = [], []
        seen = set()
        for rec, m, s in scored:
            key = tuple(round(v, 2) for v in Archive._norm_vec(rec["params"]))
            if key in seen:
                continue
            seen.add(key)
            elites.append(SceneDistribution.from_json(json.dumps(rec["params"])))
            elite_ids.append(rec["id"])
            if len(elites) >= elite_k:
                break

        top_succ = scored[0][1]["success_rate"]
        if top_succ > best_succ:
            best_succ = top_succ
            stagnant = 0
        else:
            stagnant += 1
        if verbose:
            print(f"gen {gen} elites: {elite_ids} best_succ={best_succ:.2f} "
                  f"diversity={archive.diversity():.3f} stagnant={stagnant}", flush=True)
    return archive.best()
