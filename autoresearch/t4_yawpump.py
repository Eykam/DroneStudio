import sys, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
from t4_pilot import MANIFEST
def run(label, yaw_mode, yaw_g=None, steps=700, seed=80000):
    dist = tier_dist(seed, 0)
    spec = sample_spec(seed, force_scenario="hover_hold")
    dist.n_waypoints = 0.0
    env = make_sim_factory(dist, max_steps=steps, dynamics=MANIFEST, scenario_spec=spec)(seed)
    obs = env.reset()
    if yaw_g: env._call({"cmd": "set_gains", "roll": yaw_g, "pitch": yaw_g, "yaw": yaw_g})
    ext = float(dist.scene_extent)
    died = None; maxring = 0.0; maxyaw = 0.0
    HOVER_THR = -0.756
    for step in range(steps):
        rel = (obs[0:3] + obs[19:22]) * ext
        vel = obs[3:6] * 10.0
        vy_des = np.clip(0.8 * rel[1], -0.8, 0.8)
        thr = HOVER_THR + 0.3 * (vy_des - vel[1])
        if yaw_mode == "const":
            yaw_cmd = 0.04
        else:
            bearing = float(np.arctan2(rel[2], rel[0])) if np.hypot(rel[0], rel[2]) > 0.5 else 0.0
            yaw_cmd = float(np.clip(-0.5 * bearing, -0.04, 0.04))
        obs, r, done = env.step(np.clip([0.0, yaw_cmd, 0.0, thr], -1, 1))
        maxring = max(maxring, abs(float(obs[9])), abs(float(obs[11])))
        maxyaw = max(maxyaw, abs(float(obs[10])))
        if done: died = step; break
    print(f"{label}: died_at={died} succ={env.succeeded} maxring={maxring:.2f} maxyaw={maxyaw:.3f}", flush=True)
    env.close()
run("constyaw+althold", "const")
run("bearingyaw+althold ki=0", "bearing", yaw_g=[0.1, 0.0, 0.001])
run("constyaw+althold ki=0", "const", yaw_g=[0.1, 0.0, 0.001])
