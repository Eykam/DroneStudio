import sys, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
from t4_pilot import MANIFEST
def run(lead_tau, yaw_mag, steps=700, seed=80000, sc="hover_hold"):
    dist = tier_dist(seed, 0)
    spec = sample_spec(seed, force_scenario=sc)
    dist.n_waypoints = 0.0
    env = make_sim_factory(dist, max_steps=steps, dynamics=MANIFEST, scenario_spec=spec)(seed)
    obs = env.reset(); ext = float(dist.scene_extent)
    died = None; maxring = 0.0; b0 = None; b1 = None
    HOVER_THR = -0.756
    for step in range(steps):
        rel = (obs[0:3] + obs[19:22]) * ext
        gb = obs[6:9]; rates = obs[9:12]; vel = obs[3:6] * 10.0
        rel_xz = float(np.hypot(rel[0], rel[2]))
        bearing = float(np.arctan2(rel[2], rel[0])) if rel_xz > 0.5 else 0.0
        if step == 0: b0 = abs(bearing)
        b1 = abs(bearing)
        # full teacher nav
        v_des = np.clip(0.5 * rel, -2.0, 2.0)
        if sc == "hover_hold" and np.linalg.norm(rel) < 2.0 * float(spec["success_radius"]):
            v_des = np.clip(0.4 * rel, -0.25, 0.25)
        damp = 0.5 if np.linalg.norm(rel) > 3.0 else 1.0
        a_des = np.clip(1.2 * (v_des - damp * vel), -2.0, 2.0)
        # yaw command + lead compensation
        omega = float(rates[1]) * 10.0
        yaw_cmd = float(np.clip(-0.5 * bearing - 0.05 * rates[1], -yaw_mag, yaw_mag)) if rel_xz > 0.5 else 0.0
        lead = omega * lead_tau
        c, s = np.cos(lead), np.sin(lead)
        a0 = c * a_des[0] - s * a_des[2]
        a2 = s * a_des[0] + c * a_des[2]
        gx_des = np.clip(a0 / 9.81, -0.30, 0.30)
        gz_des = np.clip(a2 / 9.81, -0.30, 0.30)
        act0 = 0.4 * (gz_des - gb[2]) - 0.6 * rates[0]
        act2 = -0.4 * (gx_des - gb[0]) - 0.6 * rates[2]
        vy_des = np.clip(0.8 * rel[1], -0.8, 0.8)
        thr = HOVER_THR + 0.3 * (vy_des - vel[1])
        obs, r, done = env.step(np.clip([act0, yaw_cmd, act2, thr], -1, 1))
        maxring = max(maxring, abs(float(obs[9])), abs(float(obs[11])))
        if done: died = step; break
    print(f"tau={lead_tau} yaw={yaw_mag}: died_at={died} succ={env.succeeded} maxring={maxring:.2f} bearing={b0:.2f}->{b1:.2f}", flush=True)
    env.close()
for tau in (0.0, 0.1, 0.2, 0.3, 0.45):
    run(tau, 0.04)
