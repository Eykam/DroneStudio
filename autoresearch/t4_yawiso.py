import sys, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
from t4_pilot import teacher_act, MANIFEST
dist = tier_dist(80000, 0)
spec = sample_spec(80000, force_scenario="hover_hold")
dist.n_waypoints = 0.0
for label, yaw_mode in [("teacher_noyaw", 0.0), ("teacher_yaw0.04", 0.04), ("teacher_yaw0.02", 0.02)]:
    env = make_sim_factory(dist, max_steps=700, dynamics=MANIFEST, scenario_spec=spec)(80000)
    obs = env.reset(); ext = float(dist.scene_extent)
    died = None; maxring = 0.0
    for step in range(700):
        a = teacher_act(obs, ext, "hover_hold", float(spec["success_radius"]))
        a[1] = yaw_mode
        obs, r, done = env.step(a)
        maxring = max(maxring, abs(float(obs[9])), abs(float(obs[11])))
        if done:
            died = step
            break
    print(f"{label}: died_at={died} succ={env.succeeded} maxring={maxring:.2f} collided={env.collided}", flush=True)
    env.close()
