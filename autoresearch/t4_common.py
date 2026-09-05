"""T4 shared: HID-64 unified-policy net (obs v3.1, 26-dim) + bias-free wp pathway.

Fresh-bootstrap track (user decision 2026-09-04): no champion warm start, no
frozen trunk - the whole net trains from random init on unified all-tier
demos, so T0-T2 and T3 behavior co-develop in one trunk with 2x the width
that seesawed at HID-32 (13-experiment retention wall, d3206fd). WP_HID
stays 32: the wp pathway was never the bottleneck. Width call was mine
(64 vs 128 offered): smallest step doubling trunk capacity, keeps CPU PPO
tractable, clean escalation to 128 if the wall persists at 64.

Champion bc_ppo_v2_best stays untouched; promotion flip frozen (user).
"""
import sys, json, os
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from policy import MLP as BCMlp
from ppo import MLP, Adam, GaussianPolicy
from scenario_sampler import sample_spec, sample_tier, tier_dist, heldout_cells_tiered, hover_max_steps
from env_sim import make_sim_factory
from eval_scenarios import post_series
from parallel_rollout import parallel_episodes

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/v14_g13.manifest.json"
OBS_DIM, ACT_DIM, HID = 26, 4, 64
V2_DIM = 19
WP_IN, WP_HID = 7, 32

DASHBOARD = os.environ.get("DASHBOARD_URL", "").rstrip("/")
TOKEN = os.environ.get("INGEST_TOKEN", "")

def wp_forward(obs, wp1, wp2):
    h = np.maximum(obs[:, V2_DIM:] @ wp1, 0.0)  # bias-free: wp=0 -> out=0
    return h @ wp2, h

def pack_actor(actor, wp1, wp2):
    return actor_to_bc_flat(actor) + list(wp1.ravel()) + list(wp2.ravel())

def unpack_wp(actor_flat):
    n2 = WP_HID * ACT_DIM; n1 = WP_IN * WP_HID
    wp2 = np.array(actor_flat[-n2:], dtype=np.float64).reshape(WP_HID, ACT_DIM)
    wp1 = np.array(actor_flat[-n2 - n1:-n2], dtype=np.float64).reshape(WP_IN, WP_HID)
    return wp1, wp2

def bc_to_actor_params(flat):
    bc = BCMlp(OBS_DIM, ACT_DIM, hidden=HID, seed=0)
    bc.set_flat(np.array(flat, dtype=np.float64))
    return {"w1": bc.W1.copy(), "b1": bc.b1.copy(), "w2": bc.W2.copy(),
            "b2": bc.b2.copy(), "w3": bc.W3.copy(), "b3": bc.b3.copy()}

def actor_to_bc_flat(actor):
    p = actor.params()
    return list(np.concatenate([p[k].ravel() for k in ("w1", "b1", "w2", "b2", "w3", "b3")]))

def eval_one(actor_flat, scenario, tier, seed):
    rng = np.random.default_rng(0)
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM); actor.load(bc_to_actor_params(actor_flat))
    wp1, wp2 = unpack_wp(actor_flat)
    hold_pin = None
    if scenario == "hover_hold60":   # pinned long-hold probe cell
        scenario, hold_pin = "hover_hold", 60.0
    dist = tier_dist(seed, tier)
    spec = sample_spec(seed, force_scenario=scenario, hold_s=hold_pin)
    if scenario != "goto":
        dist.n_waypoints = 0.0
    max_steps = 600 if tier == 3 else (400 if scenario == "goto"
              else hover_max_steps(spec.get("hold_s", 4.0), tier) if scenario == "hover_hold" else 700)
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    done = False
    while not done:
        mu, _ = actor.forward(obs[None, :])
        wpmu, _ = wp_forward(obs[None, :], wp1, wp2)
        obs, r, done = env.step(np.tanh(mu[0] + wpmu[0]))
    ok = bool(env.succeeded)
    env.close()
    return ok

EVAL_CELLS = [("goto", 0), ("goto", 1), ("goto", 2), ("goto", 3),
              ("hover_hold", 0), ("hover_hold", 2), ("hover_hold60", 0), ("land", 0), ("land", 2)]

def eval_all(actor_flat, label):
    cells = heldout_cells_tiered()
    args = [(actor_flat, sc, t, s) for (sc, t) in EVAL_CELLS
            for s in cells[("hover_hold", t) if sc == "hover_hold60" else (sc, t)]]
    res = parallel_episodes(eval_one, args)
    out, i = {}, 0
    for k in EVAL_CELLS:
        n = len(cells[k])
        out[k] = float(np.mean(res[i:i + n]))
        i += n
    for (sc, t), v in out.items():
        post_series(f"success_{sc}_t{t}", v, label)
    return out

def post_status(doc):
    if not (DASHBOARD and TOKEN):
        return
    try:
        import urllib.request
        req = urllib.request.Request(DASHBOARD + "/api/training/status",
            data=json.dumps(doc).encode(),
            headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass
