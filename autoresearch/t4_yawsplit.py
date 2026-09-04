import sys, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
from t4_pilot import MANIFEST
dist = tier_dist(80000, 0)
spec = sample_spec(80000, force_scenario="hover_hold")
dist.n_waypoints = 0.0
def run(label, kp, kd, alt_hold):
    env = make_sim_factory(dist, max_steps=700, dynamics=MANIFEST, scenario_spec=spec)(80000)
    obs = env.reset(); ext = float(dist.scene_extent)
    died = None; maxring = 0.0; minalt = 99.0; maxalt = -99.0
    for step in range(700):
        rel = (obs[0:3] + obs[19:22]) * ext
        gb = obs[6:9]; rates = obs[9:12]
        act0 = kp * (0.0 - gb[2]) - kd * rates[0]
        act2 = -kp * (0.0 - gb[0]) - kd * rates[2]
        if alt_hold:
            vy_des = np.clip(0.8 * rel[1], -0.8, 0.8)
            thr = -0.756 + 0.3 * (vy_des - obs[4] * 10.0)
        else:
            thr = -0.756
        bearing = float(np.arctan2(rel[2], rel[0]))
        yaw_cmd = float(np.clip(-0.5 * bearing, -0.04, 0.04)) if np.hypot(rel[0], rel[2]) > 0.5 else 0.0
        obs, r, done = env.step(np.clip([act0, yaw_cmd, act2, thr], -1, 1))
        maxring = max(maxring, abs(float(obs[9])), abs(float(obs[11])))
        if done: died = step; break
    print(f"{label}: died_at={died} succ={env.succeeded} maxring={maxring:.2f}", flush=True)
    env.close()
run("althold_only+yaw", 0.0, 0.0, True)
run("kp0.4_only+yaw", 0.4, 0.0, True)
run("kd0.6_only+yaw", 0.0, 0.6, True)
run("kp0.4kd0.6_noalt+yaw", 0.4, 0.6, False)
