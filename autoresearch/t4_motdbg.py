import sys, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
os.environ["HEADLESS_DBG_MOTORS"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
from t4_pilot import MANIFEST
dist = tier_dist(80000, 0)
spec = sample_spec(80000, force_scenario="hover_hold")
dist.n_waypoints = 0.0
env = make_sim_factory(dist, max_steps=700, dynamics=MANIFEST, scenario_spec=spec)(80000)
obs = env.reset(); ext = float(dist.scene_extent)
for step in range(130):
    rel = (obs[0:3] + obs[19:22]) * ext
    vel = obs[3:6] * 10.0
    vy_des = np.clip(0.8 * rel[1], -0.8, 0.8)
    thr = -0.756 + 0.3 * (vy_des - vel[1])
    obs, r, done = env.step(np.clip([0.0, 0.04, 0.0, thr], -1, 1))
    if done: break
env.close()
