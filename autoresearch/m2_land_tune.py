"""T11: sweep m2 land constants (cruise offset near pad x terminal vy)."""
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
    env = make_sim_factory(dist, max_steps=700, dynamics=t4_pilot.MANIFEST, scenario_spec=spec)(seed)
    obs = env.reset()
    for _ in range(700):
        a = t4_pilot.teacher_act(obs, ext, "land", float(spec["success_radius"]))
        obs, r, done = env.step(a)
        if done:
            break
    ok = bool(env.succeeded)
    env.close()
    return ok

results = []
for cn in (0.4, 0.6, 0.8):
    for vy in (-0.20, -0.25, -0.30, -0.40):
        t4_pilot.LAND_CRUISE_NEAR = cn
        t4_pilot.LAND_VY_DESCEND = vy
        oks = [run_ep(610000 + j) for j in range(6)]
        sr = float(np.mean(oks))
        results.append((sr, cn, vy))
        print(f"CRUISE_NEAR={cn:.1f} VY_DESC={vy:.2f}: land t0 succ {sr:.3f}", flush=True)
results.sort(reverse=True)
print(f"LAND_WINNER CRUISE_NEAR={results[0][1]} VY_DESC={results[0][2]} succ={results[0][0]:.3f}", flush=True)
print("M2_LAND_TUNE_DONE", flush=True)
