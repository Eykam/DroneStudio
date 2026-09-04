import sys, os, json
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
from t4_pilot import teacher_act, MANIFEST
dist = tier_dist(80000, 0)
spec = sample_spec(80000, force_scenario="goto")
env = make_sim_factory(dist, max_steps=400, dynamics=MANIFEST, scenario_spec=spec)(80000)
obs = env.reset(); ext = float(dist.scene_extent)
bs = []
for _ in range(400):
    a = teacher_act(obs, ext, "goto", float(spec["success_radius"]))
    rel = (obs[0:3] + obs[19:22]) * ext
    if np.hypot(rel[0], rel[2]) > 0.5:
        bs.append(abs(float(np.arctan2(rel[2], rel[0]))))
    obs, r, done = env.step(a)
    if done: break
env.close()
b = np.array(bs)
print(json.dumps({"succ_steps": len(b), "mean_abs_bearing_first20": round(float(b[:20].mean()), 3),
                  "mean_abs_bearing_last20": round(float(b[-20:].mean()), 3)}))
