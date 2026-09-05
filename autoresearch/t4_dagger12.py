"""
dag12 (2026-09-05 PM): LAND-FOCUSED DAgger - experiment (b) of the user
decision after the dag11 null (land_t0 0.0 in all 24 rounds despite a 15/16
teacher; goto reached teacher parity). dag11 recipe otherwise IDENTICAL
(integrator labels, obs v4, IC v1, dual eval, warm start t4_best,
t4_best lineage untouched, own champion file /workspace/t4_dag12_best.json).

Two coordinated levers targeting the terminal-phase sliver that plain BC
averages away:

1. LAND-ONLY MIX: collection mix is 7 near-pad t0 + 1 near-pad t1 +
   2 full-approach t2 land episodes per 10 (was 5 land / 3 hover / 2 goto).
   Anchor is teacher land successes only (same mix).

2. TERMINAL-PHASE LOSS WEIGHTING: samples inside the teacher engagement
   gate (alt<=1.4 AND dxz<=max(0.75, 1.5r)) get weight 4.0 in the BC loss,
   all others 1.0. Weights are per-sample, recorded at collection time from
   the same obs/ext geometry t4_pilot uses.

Standard eval_all27 cells unchanged so land_t0 movement (and any goto/hover
cost) reads against dag10/dag11 history. Experiment, not a promotion
candidate: floors/champion discipline kept for comparability but the yardstick
change note carries forward.
"""
import sys, json, os, time
os.environ["AUTORESEARCH_MOTOR_V2"] = "1"
os.environ["AUTORESEARCH_OBS_V4"] = "1"

def _ic(on):
    """Toggle IC v1 spawn randomization (read per-reset by env_quad)."""
    if on:
        os.environ["AUTORESEARCH_IC_V1"] = "1"
    else:
        os.environ.pop("AUTORESEARCH_IC_V1", None)
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from ppo import MLP, Adam
from scenario_sampler import (sample_spec, tier_dist, hover_max_steps,
                              heldout_cells_tiered)
from env_sim import make_sim_factory
from eval_scenarios import post_series
from parallel_rollout import parallel_episodes
import t4_common as P
from t4_pilot import teacher_act

MANIFEST = P.MANIFEST
OBS_DIM, ACT_DIM, HID, V2_DIM = 27, P.ACT_DIM, P.HID, P.V2_DIM
WP_IN, WP_HID = P.WP_IN, P.WP_HID   # wp still reads obs[19:26] (unchanged 7 cols)
ROUNDS = 24
FLOORS = {("goto", 0): 0.8, ("hover_hold", 0): 0.3, ("land", 0): 0.35}
EVAL_CELLS = P.EVAL_CELLS

def warm_params(flat26):
    """t4_best (26-dim) -> 27-dim params with a zero W1 column for SoC."""
    trunk = P.bc_to_actor_params(flat26)
    w1 = trunk["w1"]                       # (HID, 26) or (26, HID)?
    wp1, wp2 = P.unpack_wp(flat26)
    if w1.shape[-1] == 26:
        trunk["w1"] = np.concatenate([w1, np.zeros((w1.shape[0], 1))], axis=1)
    elif w1.shape[0] == 26:
        trunk["w1"] = np.concatenate([w1, np.zeros((1, w1.shape[1]))], axis=0)
    else:
        raise RuntimeError(f"unexpected w1 shape {w1.shape}")
    return {**trunk, "wp1": wp1, "wp2": wp2}

def student_act(obs, params, actor):
    actor.load({k: params[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")})
    mub, _ = actor.forward(obs[None, :])
    wpmu, _ = P.wp_forward(obs[None, :19+WP_IN] if False else obs[None, :26], params["wp1"], params["wp2"])
    return np.tanh(mub[0] + wpmu[0])

def wp_fwd27(x, wp1, wp2):
    """wp forward over the ORIGINAL 7 wp channels (obs cols 19:26)."""
    h = np.maximum(x[:, V2_DIM:26] @ wp1, 0.0)
    return h @ wp2, h

def _mix(i):
    # dag12: land-only - 7 near-pad t0, 1 near-pad t1, 2 full-approach t2
    m = i % 10
    if m < 7:
        return "land", 0, True
    if m < 8:
        return "land", 1, True
    return "land", 2, False

def _episode(seed, sc, tier, near_pad=False):
    dist = tier_dist(seed, tier)
    spec = sample_spec(seed, force_scenario=sc)
    if near_pad:
        spec["spawn_rel_goal"] = 1.5  # terminal-descent spawn above the pad
    dist.n_waypoints = 0.0
    ext = float(dist.scene_extent)
    ms = (400 if sc == "goto"
          else hover_max_steps(spec.get("hold_s", 4.0), tier) if sc == "hover_hold"
          else 700)
    env = make_sim_factory(dist, max_steps=ms, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    return env, ext, spec, ms

def _term_w(obs, ext, spec):
    """dag12 terminal-phase weight: 4.0 inside the teacher engagement gate."""
    rel = (obs[0:3] + obs[19:22]) * ext
    alt = -float(rel[1]); dxz = float(np.hypot(rel[0], rel[2]))
    r = float(spec["success_radius"])
    return 4.0 if (alt <= 1.4 and dxz <= max(0.75, 1.5 * r)) else 1.0

def collect_anchor(seed0=870000, target=60000):
    """Teacher-flown m2 episodes WITH integrator state; successes only."""
    X, A, W = [], [], []
    for i in range(120):
        seed = seed0 + i
        sc, tier, npad = _mix(i)
        env, ext, spec, ms = _episode(seed, sc, tier, npad)
        obs = env.reset()
        state = {}
        xs, acts = [], []
        for _ in range(ms):
            xs.append(obs.copy())
            acts.append(teacher_act(obs, ext, sc, float(spec["success_radius"]), state=state))
            obs, r, done = env.step(acts[-1])
            if done: break
        if env.succeeded:
            X += xs; A += acts
            W += [_term_w(o, ext, spec) for o in xs]
        env.close()
        if len(X) >= target:
            break
    return X, A, W

def dagger_rollout(params, actor, episodes, seed0):
    X, A, W, succ = [], [], [], []
    for i in range(episodes):
        seed = seed0 + i
        sc, tier, npad = _mix(i)
        env, ext, spec, ms = _episode(seed, sc, tier, npad)
        obs = env.reset()
        state = {}
        for _ in range(ms):
            X.append(obs.copy())
            A.append(teacher_act(obs, ext, sc, float(spec["success_radius"]), state=state))
            W.append(_term_w(obs, ext, spec))
            obs, r, done = env.step(student_act(obs, params, actor))
            if done: break
        succ.append(bool(env.succeeded))
        env.close()
    return X, A, W, float(np.mean(succ))

def train(params, X, A, W, Xa, Aa, Wa, epochs, rng):
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
    opt = Adam(params, lr=5e-4)
    X = np.array(X); A = np.array(A); W = np.array(W); Wa = np.array(Wa)
    for ep in range(epochs):
        idx = rng.permutation(len(X)); ida = rng.permutation(len(Xa))
        for s in range(0, len(X), 2048):
            xa = Xa[ida[s % len(Xa):s % len(Xa) + 1024]]
            aa = Aa[ida[s % len(Xa):s % len(Xa) + 1024]]
            wa = Wa[ida[s % len(Xa):s % len(Xa) + 1024]]
            if len(xa) < 1024:
                rep = int(np.ceil(1024 / max(1, len(xa))))
                xa = np.tile(Xa, (rep, 1))[:1024]; aa = np.tile(Aa, (rep, 1))[:1024]
                wa = np.tile(Wa, rep)[:1024]
            x = np.concatenate([X[idx[s:s + 2048]], xa])
            a = np.concatenate([A[idx[s:s + 2048]], aa])
            w = np.concatenate([W[idx[s:s + 2048]], wa])[:, None]
            actor.load({k: params[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")})
            mub, cache = actor.forward(x)
            wpmu, hwp = wp_fwd27(x, params["wp1"], params["wp2"])
            mu = np.tanh(mub + wpmu)
            dpre = 2.0 * (mu - a) * (1.0 - mu ** 2) * w / (len(x) * ACT_DIM)
            grads = actor.backward(dpre, cache)
            grads["wp2"] = hwp.T @ dpre
            dh = (dpre @ params["wp2"].T) * (hwp > 0)
            grads["wp1"] = x[:, V2_DIM:26].T @ dh
            tot = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
            if tot > 1.0:
                for k in grads: grads[k] *= 1.0 / (tot + 1e-8)
            opt.step(params, grads)
    actor.load({k: params[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")})
    return actor

def pack27(actor, wp1, wp2):
    p = actor.params()
    flat = list(np.concatenate([p[k].ravel() for k in ("w1", "b1", "w2", "b2", "w3", "b3")]))
    return flat + list(wp1.ravel()) + list(wp2.ravel())

def eval_one27(actor_flat, scenario, tier, seed):
    rng = np.random.default_rng(0)
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
    n_wp = WP_IN * WP_HID + WP_HID * ACT_DIM
    wp2 = np.array(actor_flat[-WP_HID * ACT_DIM:], dtype=np.float64).reshape(WP_HID, ACT_DIM)
    wp1 = np.array(actor_flat[-WP_HID * ACT_DIM - WP_IN * WP_HID:-WP_HID * ACT_DIM], dtype=np.float64).reshape(WP_IN, WP_HID)
    trunk_flat = np.array(actor_flat[:-n_wp], dtype=np.float64)
    from policy import MLP as BCMlp
    bc = BCMlp(OBS_DIM, ACT_DIM, hidden=HID, seed=0)
    bc.set_flat(trunk_flat)
    actor.load({"w1": bc.W1.copy(), "b1": bc.b1.copy(), "w2": bc.W2.copy(),
                "b2": bc.b2.copy(), "w3": bc.W3.copy(), "b3": bc.b3.copy()})
    hold_pin = None
    if scenario == "hover_hold60":
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
        wpmu, _ = wp_fwd27(obs[None, :], wp1, wp2)
        obs, r, done = env.step(np.tanh(mu[0] + wpmu[0]))
    ok = bool(env.succeeded)
    env.close()
    return ok

def eval_all27(actor_flat, label):
    cells = heldout_cells_tiered()
    args = [(actor_flat, sc, t, s) for (sc, t) in EVAL_CELLS
            for s in cells[("hover_hold", t) if sc == "hover_hold60" else (sc, t)]]
    res = parallel_episodes(eval_one27, args)
    out, i = {}, 0
    for k in EVAL_CELLS:
        kk = ("hover_hold", k[1]) if k[0] == "hover_hold60" else k
        n = len(cells[kk])
        out[k] = float(np.mean(res[i:i + n]))
        i += n
    for (sc, t), v in out.items():
        post_series(f"success_{sc}_t{t}", v, label)
    return out

def main():
    rng = np.random.default_rng(25)
    t0 = time.time()
    params0 = warm_params(json.load(open("/workspace/t4_best.json")))
    _ic(1)
    Xa, Aa, Wa = collect_anchor()
    print(f"anchor_m2v4: {len(Xa)} samples (integrator teacher, land-only successes, "
          f"term-frac {float(np.mean(np.array(Wa) > 1.0)):.3f})", flush=True)
    np.savez("/workspace/t4_demos_dag12.npz", X=np.array(Xa), A=np.array(Aa), W=np.array(Wa))
    Xa = np.array(Xa); Aa = np.array(Aa); Wa = np.array(Wa)
    XD, AD, WD = [], [], []
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
    best = {"mean": -1e9, "round": 0}
    for k in range(1, ROUNDS + 1):
        params = {k2: v2.copy() for k2, v2 in params0.items()}
        _ic(1)
        X, A, W, succ = dagger_rollout(params, actor, episodes=30, seed0=700000 + 1000 * k)
        _ic(0)
        XD += X; AD += A; WD += W
        print(f"round {k}: student succ {succ:.3f} dataset {len(XD)} wall={time.time()-t0:.0f}s", flush=True)
        actor = train(params, XD, AD, WD, Xa, Aa, Wa, epochs=100, rng=rng)
        flat = pack27(actor, params["wp1"], params["wp2"])
        json.dump(flat, open(f"/workspace/t4_dag12_r{k}.json", "w"))
        res = eval_all27(flat, f"t4-dag12 r{k}")
        _ic(1)
        res_ic = eval_all27(flat, f"t4-dag12-ic r{k}")
        _ic(0)
        mean = float(np.mean(list(res.values())))
        mean_ic = float(np.mean(list(res_ic.values())))
        floors = all(res[c] >= v for c, v in FLOORS.items())
        if floors and mean > best["mean"]:
            best = {"mean": mean, "round": k}
        print(f"round {k} EVAL " + json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}) +
              f" mean={mean:.3f} ic_mean={mean_ic:.3f} floors_m2={floors}", flush=True)
        P.post_status({"training": {"status": "running", "name": "t4_dagger11", "iter": k, "iters": ROUNDS, "note": "YARDSTICK CHANGE: skim penalty; dag12+ land numbers not comparable to dag7-9",
                       "note": f"m2 land curriculum + IC v1; r{k} mean {mean:.3f}"}})
    print("T4DAG12_BEST " + json.dumps(best), flush=True)
    if best["round"] > 0:
        import shutil
        shutil.copy(f"/workspace/t4_dag12_r{best['round']}.json", "/workspace/t4_dag12_best.json")
        # v1-plant regression read on the best (SoC pins to 1.0 on v1)
        os.environ.pop("AUTORESEARCH_MOTOR_V2", None)
        _ic(0)
        res1 = eval_all27(json.load(open("/workspace/t4_dag12_best.json")), "t4-dag12-best-v1plant")
        os.environ["AUTORESEARCH_MOTOR_V2"] = "1"
        m1 = float(np.mean(list(res1.values())))
        print("T4DAG12_BEST_SAVED t4_dag12_r%d mean_m2=%.3f v1_regression=%s mean=%.3f" % (
            best["round"], best["mean"],
            json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res1.items()}), m1), flush=True)
    P.post_status({"training": {"status": "idle", "name": "t4_dagger11",
                   "note": f"m2 obs-v4 land-curriculum + IC-v1 DAgger done; best round {best['round']} mean {best['mean']:.3f}"}})
    print("T4DAG12_DONE wall=%.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
