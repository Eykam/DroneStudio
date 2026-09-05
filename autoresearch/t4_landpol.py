import sys, os, json
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
import t4_common as P
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
from t4_pilot import MANIFEST

POL = os.environ.get("POL", "/workspace/t4_best.json")
TIER = int(os.environ.get("TIER", "0"))
flat = json.load(open(POL))

def run_one(seed):
    rng = np.random.default_rng(0)
    actor = P.MLP(rng, P.OBS_DIM, P.HID, P.ACT_DIM); actor.load(P.bc_to_actor_params(flat))
    wp1, wp2 = P.unpack_wp(flat)
    dist = tier_dist(seed, TIER)
    spec = sample_spec(seed, force_scenario="land")
    dist.n_waypoints = 0.0
    env = make_sim_factory(dist, max_steps=700, dynamics=MANIFEST, scenario_spec=spec)(seed)
    obs = env.reset(); ext = float(dist.scene_extent)
    last = None; prev_alt = None; descent = 0.0
    for _ in range(700):
        mu, _ = actor.forward(obs[None, :])
        wpmu, _ = P.wp_forward(obs[None, :], wp1, wp2)
        obs, r, done = env.step(np.tanh(mu[0] + wpmu[0]))
        rel = (obs[0:3] + obs[19:22]) * ext
        alt_now = float(-rel[1])
        if prev_alt is not None: descent = (prev_alt - alt_now) * 20.0  # m/s at 20Hz policy rate
        prev_alt = alt_now
        last = (float(rel[0]), alt_now, float(rel[2]))
        if done: break
    bv = obs[3:6] * 10.0  # body-frame velocity m/s
    last_bv = (float(bv[0]), float(bv[1]), float(bv[2]))
    succ = bool(env.succeeded); coll = env.collided
    alt = last[1]; xz = float(np.hypot(last[0], last[2]))
    env.close()
    if succ: kind = "success"
    elif coll and alt < 0.3: kind = "ground_miss"
    elif coll: kind = "obstacle_hit"
    else: kind = "timeout"
    return {"seed": seed, "kind": kind, "alt": round(alt, 2), "xz": round(xz, 2),
            "vsink": round(descent, 2), "bvy": round(last_bv[1], 2), "bvxz": round(float(np.hypot(last_bv[0], last_bv[2])), 2), "steps": env.steps}

cells = P.heldout_cells_tiered()
res = [run_one(s) for s in cells[("land", TIER)]]
from collections import Counter
print(json.dumps({"pol": POL, "tier": TIER, "counts": Counter(r["kind"] for r in res), "detail": res}, default=str))
