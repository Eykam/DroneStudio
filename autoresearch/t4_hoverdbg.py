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
env = make_sim_factory(dist, max_steps=700, dynamics=MANIFEST, scenario_spec=spec)(80000)
obs = env.reset(); ext = float(dist.scene_extent)
for step in range(700):
    a = teacher_act(obs, ext, "hover_hold", float(spec["success_radius"]))
    obs, r, done = env.step(a)
    if step >= 60:
        rel = (obs[0:3] + obs[19:22]) * ext
        vel = obs[3:6] * 10.0
        print(f"s{step} rel=({rel[0]:.2f},{rel[1]:.2f},{rel[2]:.2f}) vel=({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}) gb=({obs[6]:.2f},{obs[7]:.2f},{obs[8]:.2f}) rates=({obs[9]:.2f},{obs[10]:.3f},{obs[11]:.2f}) a=({a[0]:.2f},{a[1]:.3f},{a[2]:.2f},{a[3]:.2f}) d={done}", flush=True)
    if done: break
env.close()
