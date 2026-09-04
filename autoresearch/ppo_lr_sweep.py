"""LR sweep for PPO stabilization on the run-2 best dist (g13v140).
Same protocol as compare_trainers.py: backend=sim, eval seed 20000, 6 eps, 250 steps.
Motivation: ppo_vs_cem.json showed monotonic eval divergence at lr 3e-4."""
import json, time, sys
import numpy as np
from scene_schema import SceneDistribution
from evaluator import _backend
from outer_loop import _default_backend
from ppo import train_ppo, PPOConfig

dist = SceneDistribution.from_json(open("/tmp/best_dist2.json").read())
EnvCls, make_factory = _backend(_default_backend())
out = {}
for lr in [3e-5, 1e-4]:
    factory = make_factory(dist, max_steps=250)
    cfg = PPOConfig(rollout_episodes=8, max_updates=80, hidden=128,
                    eval_every=20, eval_episodes=6, seed=42, lr=lr)
    t0 = time.time()
    res = train_ppo(factory, EnvCls.OBS_DIM, EnvCls.ACT_DIM, config=cfg,
                    progress_cb=lambda u, em, b: print(f"  lr={lr} update {u}: eval={em:.2f}", flush=True))
    out[str(lr)] = {"wall_s": round(time.time()-t0,1),
                    "success_rate": res.success_rate if hasattr(res,"success_rate") else None,
                    "eval_history": res.eval_history if hasattr(res,"eval_history") else None,
                    "best_eval_mean": getattr(res, "best_eval_mean_during_train", None)}
    print(f"lr={lr} done in {out[str(lr)][chr(119)+chr(97)+chr(108)+chr(108)+chr(95)+chr(115)]}s", flush=True)
json.dump(out, open("/workspace/DroneStudio/autoresearch/ppo_lr_sweep.json","w"), indent=1)
print("SWEEP_COMPLETE")
