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
                 vo_aided=True, est_seed=int(seed) + 777,
                 fusion_gated=os.environ.get("FUSION_GATED") == "1")
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
    diag = dict(pos_err=float(np.mean(env.pos_errs)) if env.pos_errs else None,
                att_err=float(np.mean(env.att_errs)) if env.att_errs else None,
                vo_acc=env.vo_accepts, vo_rej=env.vo_rejects)
    env.close()
    return ok, extra, diag

def main():
    cells = heldout_cells()
    out = {}
    for sc, seeds in cells.items():
        res = parallel_episodes(teacher_est, [(sc, s) for s in seeds])
        succ = [r[0] for r in res]
        extras = [r[1] for r in res if r[1] is not None]
        diags = [r[2] for r in res]
        out[sc] = float(np.mean(succ))
        pe = np.mean([d["pos_err"] for d in diags if d["pos_err"] is not None])
        ae = np.mean([d["att_err"] for d in diags if d["att_err"] is not None])
        va = sum(d["vo_acc"] for d in diags); vr = sum(d["vo_rej"] for d in diags)
        print(f"EST-TEACHER {sc}: {out[sc]:.3f}"
              + (f" extra={np.mean(extras):.3f}" if extras else "")
              + f" pos_err={pe:.3f} att_err={ae:.2f} vo_acc={va} vo_rej={vr}", flush=True)
        post_series(f"est_teacher_{sc}", out[sc], label="pilot_act3 on est v3 obs FUSION_GATED=1")
    post_status({"candidate": {"name": "est_teacher_eval"},
                 "note": "pilot_act3 on ESTIMATED v3 obs: " +
                         " ".join(f"{k} {v:.2f}" for k, v in out.items())})
    print("EST_TEACHER_DONE " + json.dumps({k: round(v, 3) for k, v in out.items()}), flush=True)

if __name__ == "__main__":
    main()
