"""En-route facing eval: success + per-step nose-to-target alignment per cell.

Mirrors t4_common.eval_one but also tracks the yaw error implied by the
yaw-frame rel vector in obs v3 (err = |atan2(-rel[2], rel[0])|). Reports
face20 = fraction of steps with err < 20 deg, and mean err, per cell.
Usage: t4_faceeval.py <policy_flat.json> <label>
"""
import sys, os, json
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ.setdefault("AUTORESEARCH_OBS_V3", "1")
import numpy as np
import t4_common as P
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory


def eval_face(actor_flat, scenario, tier, seed):
    rng = np.random.default_rng(0)
    actor = P.MLP(rng, P.OBS_DIM, P.HID, P.ACT_DIM)
    actor.load(P.bc_to_actor_params(actor_flat))
    wp1, wp2 = P.unpack_wp(actor_flat)
    dist = tier_dist(seed, tier)
    spec = sample_spec(seed, force_scenario=scenario)
    if scenario != "goto":
        dist.n_waypoints = 0.0
    max_steps = 600 if tier == 3 else (400 if scenario == "goto" else 700)
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=P.MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    errs = []
    done = False
    while not done:
        mu, _ = actor.forward(obs[None, :])
        wpmu, _ = P.wp_forward(obs[None, :], wp1, wp2)
        obs, r, done = env.step(np.tanh(mu[0] + wpmu[0]))
        errs.append(abs(float(np.arctan2(-obs[2], obs[0]))))
    ok = bool(env.succeeded)
    env.close()
    errs = np.asarray(errs)
    return ok, float((errs < 0.349).mean()), float(errs.mean())


def main(path, label):
    flat = np.array(json.load(open(path)), dtype=np.float64)
    cells = P.heldout_cells_tiered()
    args = [(flat, sc, t, s) for (sc, t) in P.EVAL_CELLS for s in cells[(sc, t)]]
    res = P.parallel_episodes(eval_face, args)
    out, i = {}, 0
    for k in P.EVAL_CELLS:
        n = len(cells[k])
        chunk = res[i:i + n]
        i += n
        succ = float(np.mean([c[0] for c in chunk]))
        face20 = float(np.mean([c[1] for c in chunk]))
        merr = float(np.mean([c[2] for c in chunk]))
        out[f"{k[0]}_t{k[1]}"] = {"succ": round(succ, 3), "face20": round(face20, 3),
                                   "mean_err": round(merr, 3)}
        P.post_series(f"face20_{k[0]}_t{k[1]}", face20, label)
        P.post_series(f"success_{k[0]}_t{k[1]}", succ, label)
    out["ALL"] = {
        "succ": round(float(np.mean([v["succ"] for v in out.values()])), 3),
        "face20": round(float(np.mean([v["face20"] for v in out.values()])), 3),
        "mean_err": round(float(np.mean([v["mean_err"] for v in out.values()])), 3),
    }
    print(json.dumps({label: out}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
