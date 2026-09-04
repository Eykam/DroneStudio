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
def run(label, mode):
    env = make_sim_factory(dist, max_steps=700, dynamics=MANIFEST, scenario_spec=spec)(80000)
    obs = env.reset(); ext = float(dist.scene_extent)
    died = None; maxring = 0.0; b0 = None; b1 = None; thr_hold = -0.756; burst_step = [0]
    for step in range(700):
        rel = (obs[0:3] + obs[19:22]) * ext
        gb = obs[6:9]; rates = obs[9:12]; vel = obs[3:6] * 10.0
        rel_xz = float(np.hypot(rel[0], rel[2]))
        bearing = float(np.arctan2(rel[2], rel[0])) if rel_xz > 0.5 else 0.0
        if step == 0: b0 = abs(bearing)
        b1 = abs(bearing)
        misaligned = abs(bearing) > 0.35 and rel_xz > 0.5
        vy_des = np.clip(0.8 * rel[1], -0.8, 0.8)
        if mode == "alt30":
            # 30-step yaw bursts with frozen throttle, 30-step full-nav recovery
            ph = step % 60
            in_burst = misaligned and ph < 30
            yaw_cmd = float(np.clip(-0.5 * bearing - 0.05 * rates[1], -0.04, 0.04)) if in_burst else 0.0
            if in_burst:
                if ph == 0: thr_hold = -0.756 + 0.3 * (vy_des - vel[1])
                thr = thr_hold
            else:
                thr = -0.756 + 0.3 * (vy_des - vel[1])
        elif mode == "slowthr":
            # thrust updates only every 10 steps, continuous yaw
            yaw_cmd = float(np.clip(-0.5 * bearing - 0.05 * rates[1], -0.04, 0.04)) if misaligned else 0.0
            if step % 10 == 0: thr_hold = -0.756 + 0.3 * (vy_des - vel[1])
            thr = thr_hold
        act0 = 0.4 * (0.0 - gb[2]) - 0.6 * rates[0]
        act2 = -0.4 * (0.0 - gb[0]) - 0.6 * rates[2]
        obs, r, done = env.step(np.clip([act0, yaw_cmd, act2, thr], -1, 1))
        maxring = max(maxring, abs(float(obs[9])), abs(float(obs[11])))
        if done: died = step; break
    print(f"{label}: died_at={died} succ={env.succeeded} maxring={maxring:.2f} bearing={b0:.2f}->{b1:.2f}", flush=True)
    env.close()
run("alt30/30", "alt30")
run("slowthr10", "slowthr")
