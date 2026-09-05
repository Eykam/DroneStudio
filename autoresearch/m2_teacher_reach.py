"""m2_teacher_reach.py - reachability study: what terminal dxz can a
position PD loop achieve under m2 actuation latency? (dag10 follow-up,
parent-approved 12:48 PM.)

Spawns the drone at alt 1.4m at fixed horizontal offsets from the pad,
runs a position PD through the teacher's exact attitude/thrust pipeline,
measures convergence. Answers: is the 0.5m tier-0 radius reachable, and
what hold time does a descend-when-centered gate need?
"""
import os, sys, json
os.environ["AUTORESEARCH_MOTOR_V2"] = "1"
os.environ["AUTORESEARCH_OBS_V4"] = "1"
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
import t4_pilot
from t4_pilot import HOVER_THR, K_VTHR, KP_ATT, KD_ATT

def pd_act(obs, ext, kp, kd):
    rel = (obs[0:3] + obs[19:22]) * ext
    vel = obs[3:6] * 10.0
    gb = obs[6:9]
    rates = obs[9:12]
    a_des = np.clip(kp * np.array([rel[0], 0.0, rel[2]]) - kd * np.array([vel[0], 0.0, vel[2]]), -2.5, 2.5)
    gx_des = np.clip(a_des[0] / 9.81, -0.30, 0.30)
    gz_des = np.clip(a_des[2] / 9.81, -0.30, 0.30)
    act0 = KP_ATT * (gz_des - gb[2]) - KD_ATT * rates[0]
    act2 = -KP_ATT * (gx_des - gb[0]) - KD_ATT * rates[2]
    soc_ff = float(np.clip(-0.0061 - 0.0794 * (1.0 - float(obs[26])), -0.12, 0.12))
    thr = HOVER_THR + K_VTHR * (0.0 - vel[1]) + soc_ff
    return np.clip(np.array([act0, 0.0, act2, thr]), -1, 1)

def run(dx, v0, kp, kd, seed=77000):
    dist = tier_dist(seed, 0)
    spec = sample_spec(seed, force_scenario="land")
    dist.n_waypoints = 0.0
    env = make_sim_factory(dist, max_steps=200, dynamics=t4_pilot.MANIFEST, scenario_spec=spec)(seed)
    env.reset()  # populates spawn/goal/obs_* scene attrs
    gx, gz = float(env.goal[0]), float(env.goal[2])
    scene = {
        "spawn": [gx + dx, 1.4, gz],
        "goal": [float(x) for x in env.goal],
        "obstacles": [[float(c[0]), float(c[1]), float(c[2]), float(r)]
                      for c, r in zip(env.obs_centers, env.obs_radii)],
        "extent": float(env.dist.scene_extent),
        "max_steps": 200,
        "dynamics_noise": float(env.dist.dynamics_noise),
        "scenario": "land",
        "success_radius": float(spec["success_radius"]),
        "spawn_vel": [v0, 0.0, 0.0],
    }
    resp = env._call({"cmd": "reset", "seed": int(seed), "scene": scene})
    env.pos = np.array(scene["spawn"], dtype=np.float64)
    obs = np.array(resp["obs"], dtype=np.float64)
    ext = float(dist.scene_extent)
    dxz_hist = []
    done = False
    while not done:
        rel = (obs[0:3] + obs[19:22]) * ext
        dxz_hist.append(float(np.hypot(rel[0], rel[2])))
        obs, r, done = env.step(pd_act(obs, ext, kp, kd))
    env.close()
    h = np.array(dxz_hist)
    settle = next((i for i in range(len(h) - 20) if np.all(h[i:i + 20] <= 0.25)), None)
    tail = h[-40:] if len(h) >= 40 else h
    return {"settle_steps": settle, "min": float(h.min()), "tail_mean": float(tail.mean()),
            "tail_pp": float(tail.max() - tail.min()), "ended": bool(done), "n": len(h)}

print("PD reachability: alt=1.4m spawn, hold 10s (200 steps @20Hz)", flush=True)
print(f"{'kp':>4} {'kd':>4} {'dx0':>5} {'v0':>4} | {'settle':>6} {'min':>5} {'tail_mean':>9} {'tail_pp':>7} {'term':>5}", flush=True)
for kp, kd in [(1.2, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.5)]:
    for dx in (0.5, 1.0, 2.0):
        for v0 in (0.0, 0.5):
            r = run(dx, v0, kp, kd)
            st = f"{r['settle_steps']}" if r['settle_steps'] is not None else "never"
            print(f"{kp:4.1f} {kd:4.1f} {dx:5.2f} {v0:4.1f} | {st:>6} {r['min']:5.2f} {r['tail_mean']:9.3f} {r['tail_pp']:7.3f} {str(r['ended']):>5}", flush=True)
print("REACH_DONE", flush=True)
