"""Observability-gap diagnostic: estimator errors on hover/land vs goto,
est-obs (policy sees noise) vs passthrough (policy sees GT, estimator passive)."""
import json, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import os
os.environ["AUTORESEARCH_OBS_V2"] = "1"
from scenario_sampler import sample_spec
from eval_scenarios import cell_dist
from eval_estimated import EstEnv, load_policy, MANIFEST

def run(policy_path, scenario, passthrough, n=12, base_seed=30_000):
    net, od = load_policy(policy_path)
    out = []
    for k in range(n):
        seed = base_seed + k
        env = EstEnv(cell_dist(seed), seed=seed, max_steps=700, dynamics=MANIFEST,
                     scenario_spec=sample_spec(seed, force_scenario=scenario),
                     estimated=True, obs_v2=True, vo_aided=True,
                     passthrough=passthrough, est_seed=seed + 555)
        obs = env.reset()
        while True:
            obs, r, done = env.step(net.act(obs))
            if done:
                break
        def q(a, p):
            return float(np.percentile(a, p)) if len(a) else None
        out.append(dict(ok=bool(env.succeeded), collided=env.collided, steps=env.steps,
                        pos=q(env.pos_errs, 50), pos90=q(env.pos_errs, 90),
                        vel=q(env.vel_errs, 50), vel90=q(env.vel_errs, 90),
                        att=q(env.att_errs, 50), att90=q(env.att_errs, 90),
                        rate=q(env.rate_errs, 50)))
        env.close()
    keys = ["pos", "pos90", "vel", "vel90", "att", "att90", "rate"]
    agg = dict(scenario=scenario, passthrough=passthrough, n=n,
               success=float(np.mean([r["ok"] for r in out])),
               collision=float(np.mean([r["collided"] for r in out])),
               steps=float(np.mean([r["steps"] for r in out])))
    for k_ in keys:
        agg[k_] = float(np.mean([r[k_] for r in out if r[k_] is not None]))
    return agg

if __name__ == "__main__":
    pol = "/workspace/bc_ppo_v2_best.json"
    res = []
    for sc in ("goto", "hover_hold", "land"):
        for pt in (False, True):
            agg = run(pol, sc, pt)
            res.append(agg)
            print(json.dumps(agg), flush=True)
    json.dump(res, open("/workspace/vision_model/diag_observability_champion.json", "w"), indent=1)
