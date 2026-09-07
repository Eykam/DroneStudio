"""Hypothesis (a) gate measurement: pilot_act3 (full-authority scripted pilot)
driven by ESTIMATED v3 obs (first 19 dims = v2 layout) instead of GT.
If teacher-on-est succeeds at hover/land, estimation noise does not block a
closed-loop controller -> hybrid BC + scripted terminal layer can ship.
If teacher-on-est ALSO fails, the wall is estimation quality itself.
Publishes est_teacher_* series.
"""
import os, json
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
from diverse_bc import pilot_act3
from scenario_sampler import sample_spec, heldout_cells
from eval_scenarios import cell_dist, post_series
from parallel_rollout import parallel_episodes
from ppo_v2 import MANIFEST, post_status
from eval_estimated import EstEnv

def teacher_est(sc, seed):
    spec = sample_spec(seed, force_scenario=sc)
    dist = cell_dist(seed)
    ms = 400 if sc == "goto" else 2100
    env = EstEnv(dist, seed=seed, max_steps=ms, dynamics=MANIFEST,
                 scenario_spec=spec, estimated=True, obs_v2=True, obs_v3=True,
                 vo_aided=True, est_seed=int(seed) + 777)
    obs = env.reset()
    hold_speeds, pos_errs = [], []
    for _ in range(ms):
        a = pilot_act3(obs[:19], 1.75, dist.scene_extent)
        obs, r, done = env.step(a)
        info = env.last_info
        if info.get("hold_steps", 0) > 0:
            v = info.get("vel", [0, 0, 0])
            hold_speeds.append(float(np.linalg.norm(v)))
        if done:
            break
    ok = bool(env.succeeded)
    extra = float(np.mean(hold_speeds)) if hold_speeds else None
    env.close()
    return ok, extra

def main():
    cells = heldout_cells()
    out = {}
    for sc, seeds in cells.items():
        res = parallel_episodes(teacher_est, [(sc, s) for s in seeds])
        succ = [r[0] for r in res]
        extras = [r[1] for r in res if r[1] is not None]
        out[sc] = float(np.mean(succ))
        print(f"EST-TEACHER {sc}: {out[sc]:.3f}"
              + (f" extra={np.mean(extras):.3f}" if extras else ""), flush=True)
        post_series(f"est_teacher_{sc}", out[sc], label="pilot_act3 on est v3 obs")
    post_status({"candidate": {"name": "est_teacher_eval"},
                 "note": "pilot_act3 on ESTIMATED v3 obs: " +
                         " ".join(f"{k} {v:.2f}" for k, v in out.items())})
    print("EST_TEACHER_DONE " + json.dumps({k: round(v, 3) for k, v in out.items()}), flush=True)

if __name__ == "__main__":
    main()
