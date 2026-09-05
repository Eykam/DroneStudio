import sys, os, json
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
import t4_common as P
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
from t4_pilot import teacher_act, MANIFEST

TIER = int(os.environ.get("TIER", "2"))
def run_one(seed):
    dist = tier_dist(seed, TIER)
    spec = sample_spec(seed, force_scenario="land")
    dist.n_waypoints = 0.0
    env = make_sim_factory(dist, max_steps=700, dynamics=MANIFEST, scenario_spec=spec)(seed)
    obs = env.reset(); ext = float(dist.scene_extent)
    min_d = 1e9; last = None
    for _ in range(700):
        a = teacher_act(obs, ext, "land", float(spec["success_radius"]))
        obs, r, done = env.step(a)
        rel = (obs[0:3] + obs[19:22]) * ext
        d = float(np.linalg.norm(obs[12:15] * ext))
        if 1e-3 < d < min_d: min_d = d
        last = (float(rel[0]), float(-rel[1]), float(rel[2]), float(np.hypot(rel[0], rel[2])))
        if done: break
    succ = bool(env.succeeded); coll = env.collided
    env.close()
    alt, xz = last[1], last[3]
    if succ: kind = "success"
    elif coll and alt < 0.3: kind = "ground_miss"      # touched down off-pad or too fast
    elif coll: kind = "obstacle_hit"
    else: kind = "timeout"
    return {"seed": seed, "kind": kind, "alt": round(alt,2), "xz": round(xz,2),
            "min_d": round(min_d,2), "steps": env.steps}

cells = P.heldout_cells_tiered()
seeds = cells[("land", TIER)]
res = [run_one(s) for s in seeds]
from collections import Counter
print(json.dumps({"counts": Counter(r["kind"] for r in res), "detail": res}, default=str)[:3000])
