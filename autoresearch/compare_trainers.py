#!/usr/bin/env python3
"""PPO vs CEM on identical footing: same distribution, same held-out eval
episodes, wall-clock accounted. Answers the question the night queue cares
about: does first-order training beat zeroth-order on QuadNavEnv at box
scale?

Usage: python compare_trainers.py [--quick] [--report PATH]
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scene_schema import SceneDistribution
from env_quad import QuadNavEnv, make_quad_factory
from policy import cem_train
from ppo import train_ppo, PPOConfig, MLP

def eval_policy_act(act, dist, eval_seed, episodes, max_steps):
    rets, succs = [], []
    for i in range(episodes):
        env = QuadNavEnv(dist, seed=eval_seed + i, max_steps=max_steps)
        obs = env.reset()
        total = 0.0
        for _ in range(max_steps):
            obs, r, done = env.step(act(obs))
            total += r
            if done:
                break
        rets.append(total)
        succs.append(1.0 if env.succeeded else 0.0)
    return {"success_rate": float(np.mean(succs)), "mean_return": float(np.mean(rets))}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_vs_cem.json"))
    a = p.parse_args()

    dist = SceneDistribution()
    max_steps = 200 if a.quick else 250
    eval_eps = 4 if a.quick else 6
    factory = make_quad_factory(dist, max_steps=max_steps)

    # --- CEM
    t0 = time.time()
    cem_iters, cem_pop = (2, 8) if a.quick else (6, 12)
    cem_policy, cem_train_ret = cem_train(factory, QuadNavEnv.OBS_DIM, QuadNavEnv.ACT_DIM,
                                          iters=cem_iters, pop=cem_pop, episodes_per_eval=3,
                                          seed=a.seed)
    cem_wall = time.time() - t0
    cem_eval = eval_policy_act(cem_policy.act, dist, 20_000, eval_eps, max_steps)

    # --- PPO
    t0 = time.time()
    cfg = PPOConfig(rollout_episodes=4 if a.quick else 8,
                    max_updates=10 if a.quick else 80,
                    hidden=64 if a.quick else 128,
                    eval_every=5 if a.quick else 20,
                    eval_episodes=eval_eps, seed=a.seed)
    res = train_ppo(factory, QuadNavEnv.OBS_DIM, QuadNavEnv.ACT_DIM, config=cfg,
                    progress_cb=lambda u, em, b: print(f"  ppo update {u}: eval={em:.2f} best={b:.2f}", flush=True))
    ppo_wall = time.time() - t0
    bp = res.best_params
    actor = MLP(np.random.default_rng(0), QuadNavEnv.OBS_DIM, cfg.hidden, QuadNavEnv.ACT_DIM)
    actor.load(bp["actor"])
    mean, var = bp["norm_mean"], bp["norm_var"]
    def ppo_act(obs):
        nobs = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10)
        mu, _ = actor.forward(nobs[None, :])
        return np.tanh(mu[0])
    ppo_eval = eval_policy_act(ppo_act, dist, 20_000, eval_eps, max_steps)

    report = {
        "distribution": json.loads(dist.to_json()),
        "eval_protocol": {"eval_seed": 20_000, "episodes": eval_eps, "max_steps": max_steps},
        "cem": {"iters": cem_iters, "pop": cem_pop, "wall_s": round(cem_wall, 1),
                "train_best_return": float(cem_train_ret), **cem_eval},
        "ppo": {"config": {k: v for k, v in res.config.items()},
                "wall_s": round(ppo_wall, 1),
                "best_eval_mean_during_train": res.best_eval_mean,
                "env_steps": res.total_env_steps,
                "eval_history": res.eval_history, **ppo_eval},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(a.report, "w") as f:
        json.dump(report, f, indent=2)
    print("COMPARE_COMPLETE", a.report)
    print(json.dumps({"cem": report["cem"]["success_rate"], "ppo": report["ppo"]["success_rate"]}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
