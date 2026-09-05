"""T11 follow-up: why does land fail under motor_v2? (winner constants)

Dumps per-step alt / vy / thr / phase gates for a few land t0 episodes
under the m2 plant so the failure phase (transit vs corridor vs touchdown)
is visible. Diagnosis only - no tuning here.
"""
import sys, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
os.environ["AUTORESEARCH_MOTOR_V2"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
import t4_pilot

for j in range(3):
    seed = 610000 + j
    dist = tier_dist(seed, 0)
    spec = sample_spec(seed, force_scenario="land")
    dist.n_waypoints = 0.0
    ext = float(dist.scene_extent)
    env = make_sim_factory(dist, max_steps=700, dynamics=t4_pilot.MANIFEST, scenario_spec=spec)(seed)
    obs = env.reset()
    st = {}
    rel0 = (obs[0:3] + obs[19:22]) * ext
    print(f"--- seed {seed} pad rel start=({rel0[0]:.2f},{rel0[1]:.2f},{rel0[2]:.2f}) radius={float(spec['success_radius']):.2f}", flush=True)
    steps = 0
    for i in range(700):
        a = t4_pilot.teacher_act(obs, ext, "land", float(spec["success_radius"]), state=st)
        if i % 20 == 0:
            rel = (obs[0:3] + obs[19:22]) * ext
            vel = obs[3:6] * 10.0
            print(f"  i={i:3d} alt={-rel[1]:6.2f} dxz={np.hypot(rel[0],rel[2]):5.2f} vy={vel[1]:6.2f} thr={a[3]:6.3f}", flush=True)
        obs, r, done = env.step(a)
        steps = i
        if done:
            break
    rel = (obs[0:3] + obs[19:22]) * ext
    vel = obs[3:6] * 10.0
    print(f"  END i={steps} succ={bool(env.succeeded)} alt={-rel[1]:.2f} dxz={np.hypot(rel[0],rel[2]):.2f} vy={vel[1]:.2f} |v|={np.linalg.norm(vel):.2f}", flush=True)
    env.close()
print("M2_LAND_DIAG_DONE", flush=True)
