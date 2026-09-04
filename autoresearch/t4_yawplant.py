import sys, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
from t4_pilot import MANIFEST
HOVER_THR = -0.756
dist = tier_dist(80000, 0)
spec = sample_spec(80000, force_scenario="hover_hold")
dist.n_waypoints = 0.0
for label, yawfn in [("const0.04", lambda s: 0.04), ("const0.01", lambda s: 0.01), ("pulsed", lambda s: 0.04 if (s // 10) % 2 == 0 else 0.0)]:
    env = make_sim_factory(dist, max_steps=700, dynamics=MANIFEST, scenario_spec=spec)(80000)
    obs = env.reset()
    died = None
    for step in range(300):
        a = np.array([0.0, yawfn(step), 0.0, HOVER_THR])
        obs, r, done = env.step(a)
        if done:
            died = step
            break
    gb = obs[6:9]; rates = obs[9:12]
    print(f"{label}: died_at={died} gb=({gb[0]:.2f},{gb[1]:.2f},{gb[2]:.2f}) rates=({rates[0]:.2f},{rates[1]:.3f},{rates[2]:.2f}) collided={env.collided}", flush=True)
    env.close()
