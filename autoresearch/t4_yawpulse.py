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
for label, on, off, mag in [("p15/15@0.04", 15, 15, 0.04), ("p10/20@0.03", 10, 20, 0.03), ("p8/25@0.04", 8, 25, 0.04)]:
    env = make_sim_factory(dist, max_steps=700, dynamics=MANIFEST, scenario_spec=spec)(80000)
    obs = env.reset(); ext = float(dist.scene_extent)
    died = None; maxring = 0.0
    b_first = []; b_last = []
    for step in range(700):
        a = teacher_act(obs, ext, "hover_hold", float(spec["success_radius"]))
        rel = (obs[0:3] + obs[19:22]) * ext
        bearing = float(np.arctan2(rel[2], rel[0]))
        a[1] = np.clip(-0.5 * bearing, -mag, mag) if (step % (on + off)) < on and np.hypot(rel[0], rel[2]) > 0.5 else 0.0
        obs, r, done = env.step(a)
        if step < 20: b_first.append(abs(bearing))
        if step > 680 or done: b_last.append(abs(bearing))
        maxring = max(maxring, abs(float(obs[9])), abs(float(obs[11])))
        if done:
            died = step
            break
    print(f"{label}: died_at={died} succ={env.succeeded} maxring={maxring:.2f} bearing={np.mean(b_first):.2f}->{np.mean(b_last):.2f}", flush=True)
    env.close()
