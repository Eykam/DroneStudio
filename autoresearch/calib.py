import os, json, numpy as np, sys
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
from scenario_sampler import tier_dist, sample_spec
from env_sim import make_sim_factory
import ppo_v39 as P

actor = P.MLP(np.random.default_rng(0), P.OBS_DIM, P.HID, P.ACT_DIM)
actor.load(P.warm_start_params())
# hover_hold T0: what throttle does the champion hold?
dist = tier_dist(777, 0)
spec = sample_spec(777, force_scenario="hover_hold")
env = make_sim_factory(dist, max_steps=200, dynamics=P.MANIFEST, scenario_spec=spec)(777)
o = env.reset()
thr = []
for _ in range(200):
    mu, _ = actor.forward(np.array(o)[None, :])
    a = np.tanh(mu[0]); thr.append(a[3])
    o, r, done = env.step(a)
    if done: break
print("champion hover: succ", env.succeeded, "steps", env.steps,
      "thr mean %.3f std %.3f" % (np.mean(thr), np.std(thr)),
      "pos", [round(x,2) for x in env.last_info.get("pos", [])],
      "vel", [round(x,2) for x in env.last_info.get("vel", [])])
env.close()
