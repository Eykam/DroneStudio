import os, json, numpy as np
os.environ["AUTORESEARCH_OBS_V3"] = "1"
from scenario_sampler import tier_dist, sample_spec
from env_sim import make_sim_factory
import ppo_v36 as P

actor = P.MLP(np.random.default_rng(0), P.OBS_DIM, P.HID, P.ACT_DIM)
actor.load(P.warm_start_params())
for lat in (0.05, 0.15, 0.35):
    dist = tier_dist(1234, 3); dist.waypoint_lat = lat
    spec = sample_spec(1234, force_scenario="goto")
    env = make_sim_factory(dist, max_steps=600, dynamics=P.MANIFEST, scenario_spec=spec)(1234)
    o = env.reset()
    wps = np.array(env.waypoints); goal = np.array(env.goal, dtype=np.float64)
    mind = [1e9]*len(wps); mgoal = 1e9
    done = False; steps = 0; r_tot = 0.0
    while not done:
        mu, _ = actor.forward(np.array(o)[None, :])
        o, r, done = env.step(np.tanh(mu[0])); steps += 1; r_tot += r
        pos = np.array(env.last_info.get("pos")) if env.last_info.get("pos") else None
        if pos is not None:
            for i, w in enumerate(wps):
                mind[i] = min(mind[i], float(np.linalg.norm(pos - w)))
            mgoal = min(mgoal, float(np.linalg.norm(pos - goal)))
    print("lat=%.2f steps=%d ret=%.2f wp=%s succ=%s mind=%s mgoal=%.2f wps=%s goal=%s" % (
        lat, steps, r_tot, env.last_info.get("wp"), env.last_info.get("succeeded"),
        [round(m, 2) for m in mind], mgoal, wps.round(2).tolist(), goal.round(2).tolist()))
    env.close()
