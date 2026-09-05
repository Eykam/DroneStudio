"""T11 round 2: sweep m2 terminal-commit throttle x land K_VTHR, 1200 steps."""
import sys, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
os.environ["AUTORESEARCH_MOTOR_V2"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
import t4_pilot

def run_ep(seed):
    dist = tier_dist(seed, 0)
    spec = sample_spec(seed, force_scenario="land")
    dist.n_waypoints = 0.0
    ext = float(dist.scene_extent)
    env = make_sim_factory(dist, max_steps=1200, dynamics=t4_pilot.MANIFEST, scenario_spec=spec)(seed)
    obs = env.reset()
    st = {}
    for _ in range(1200):
        a = t4_pilot.teacher_act(obs, ext, "land", float(spec["success_radius"]), state=st)
        obs, r, done = env.step(a)
        if done:
            break
    ok = bool(env.succeeded)
    rel = (obs[0:3] + obs[19:22]) * ext
    vel = obs[3:6] * 10.0
    env.close()
    return ok, -float(rel[1]), float(vel[1])

results = []
for commit in (-0.78, -0.80, -0.82, -0.85):
    for kv in (0.10, 0.18):
        t4_pilot.LAND_COMMIT_THR = commit
        t4_pilot.K_VTHR = kv
        outs = [run_ep(610000 + j) for j in range(6)]
        sr = float(np.mean([o[0] for o in outs]))
        vy_fail = [o[2] for o in outs if not o[0]]
        print(f"COMMIT={commit:.2f} K_VTHR={kv:.2f}: succ {sr:.3f} endvy_fail={np.mean(vy_fail) if vy_fail else 0:.2f}", flush=True)
        results.append((sr, commit, kv))
results.sort(reverse=True)
print(f"LAND2_WINNER COMMIT={results[0][1]} K_VTHR={results[0][2]} succ={results[0][0]:.3f}", flush=True)
print("M2_LAND_TUNE2_DONE", flush=True)
