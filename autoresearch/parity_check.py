#!/usr/bin/env python3
"""Parity check: identical concrete scene + identical action sequence through
QuadNavEnv (numpy) and SimBinaryEnv (Zig/Bullet). Trajectories should track
within integration tolerance; termination behavior should match.

Writes parity_report.json. Exit 0 always - the report is the artifact.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scene_schema import SceneDistribution
from env_quad import QuadNavEnv
from env_sim import SimBinaryEnv

def run_episode(env, actions):
    obs = env.reset()
    traj = [obs[:3].copy()]  # rel_goal as position proxy? no - keep raw obs
    obses = [obs.copy()]
    total = 0.0
    for a in actions:
        obs, r, done = env.step(a)
        obses.append(obs.copy())
        total += r
        if done:
            break
    return obses, total, env.succeeded

def main():
    dist = SceneDistribution()
    dist.dynamics_noise = 0.0  # deterministic for parity
    seed = 123
    rng = np.random.default_rng(7)
    n_steps = 60
    # hover-ish then climb: thrust around hover (14.715N -> a=(14.715/40)*2-1)
    hover_a = (1.5 * 9.81 / 40.0) * 2 - 1
    actions = []
    for i in range(n_steps):
        actions.append([0.0, 0.0, 0.0, hover_a + 0.05 * np.sin(i / 8.0)])

    qenv = QuadNavEnv(dist, seed=seed, max_steps=200)
    senv = SimBinaryEnv(dist, seed=seed, max_steps=200)

    t0 = time.time(); q_obs, q_ret, q_succ = run_episode(qenv, actions); q_wall = time.time() - t0
    t0 = time.time(); s_obs, s_ret, s_succ = run_episode(senv, actions); s_wall = time.time() - t0
    senv.close()

    n = min(len(q_obs), len(s_obs))
    # obs[3:6] is velocity/10 - derive a divergence metric from goal-relative
    # position proxy: rel_goal obs[0:3]*extent tracks -(pos) up to goal const
    q_arr = np.array(q_obs[:n]); s_arr = np.array(s_obs[:n])
    rel_div = np.linalg.norm((q_arr[:, :3] - s_arr[:, :3]) * dist.scene_extent, axis=1)
    vel_div = np.linalg.norm((q_arr[:, 3:6] - s_arr[:, 3:6]) * 10.0, axis=1)
    report = {
        "steps_compared": n,
        "quad": {"return": q_ret, "succeeded": bool(q_succ), "wall_s": round(q_wall, 3)},
        "sim": {"return": s_ret, "succeeded": bool(s_succ), "wall_s": round(s_wall, 3)},
        "position_divergence_m": {"final": float(rel_div[-1]), "max": float(rel_div.max()),
                                   "mean": float(rel_div.mean())},
        "velocity_divergence_mps": {"final": float(vel_div[-1]), "max": float(vel_div.max())},
        "verdict": "track" if rel_div[-1] < 1.0 else ("diverge-soft" if rel_div[-1] < 5.0 else "diverge-hard"),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
