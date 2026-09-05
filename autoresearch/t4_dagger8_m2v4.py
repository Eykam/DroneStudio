"""dag8m2v4 (2026-09-04 late): m2-fidelity DAgger on obs v4 (27-dim, +SoC).

Root-caused dag7m2's dead land: its rollouts called teacher_act WITHOUT the
per-episode state dict, so the m2 land trim integrator (the fix that makes
the teacher land at all under motor_v2) was INACTIVE - land labels came
from the known-broken stateless teacher (0/24 on the 910000 seeds vs
integrator 11/24). land_t0 = 0.0 across all rounds followed from the
labels, not from student capacity.

dag8 fixes the labels: per-episode state dict passed (integrators active,
t4_pilot mode now selected by the state arg), AND obs v4 (SoC visible) so
the integrator's equilibrium trim - largely a function of SoC - is
Markovian in the student's inputs and therefore imitable by a feedforward
net. The stateless SoC-feedforward variant (fit: trim ~= -0.0061 -
0.0794*(1-soc)) was tested and REJECTED for land (0/24, stall by timeout);
it stays in t4_pilot for hover_hold, where it beat the micro-trim
integrator (0.500 vs ~0.35).

Student: 27-dim trunk warm-started from t4_best with a ZERO W1 column for
SoC (bit-exact v1 behavior at soc=1), wp pathway unchanged (cols 19:26).
Self-contained eval (OBS_DIM=27, same heldout cells/seeds); m2-relative
floors (parent-approved): goto_t0>=0.8 AND hover_hold_t0>=0.3 AND
land_t0>=0.35. Ends with a v1-plant regression eval of the best (motor_v2
toggled off per-episode spawn; SoC reads 1.0 constant).

Own champion file /workspace/t4_dag8m2v4_best.json - t4_best untouched.
"""
import sys, json, os, time
os.environ["AUTORESEARCH_MOTOR_V2"] = "1"
os.environ["AUTORESEARCH_OBS_V4"] = "1"
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
ROUNDS = 8
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
    m = i % 10
    if m < 4:
        return "land", (0 if m < 3 else 2)
    if m < 8:
        return "hover_hold", i % 3
    return "goto", i % 3

def _episode(seed, sc, tier):
    dist = tier_dist(seed, tier)
    spec = sample_spec(seed, force_scenario=sc)
    dist.n_waypoints = 0.0
    ext = float(dist.scene_extent)
    ms = (400 if sc == "goto"
          else hover_max_steps(spec.get("hold_s", 4.0), tier) if sc == "hover_hold"
          else 700)
    env = make_sim_factory(dist, max_steps=ms, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    return env, ext, spec, ms

def collect_anchor(seed0=770000, target=60000):
    """Teacher-flown m2 episodes WITH integrator state; successes only."""
    X, A = [], []
    for i in range(120):
        seed = seed0 + i
        sc, tier = _mix(i)
        env, ext, spec, ms = _episode(seed, sc, tier)
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
        env.close()
        if len(X) >= target:
            break
    return X, A

def dagger_rollout(params, actor, episodes, seed0):
    X, A, succ = [], [], []
    for i in range(episodes):
        seed = seed0 + i
        sc, tier = _mix(i)
        env, ext, spec, ms = _episode(seed, sc, tier)
        obs = env.reset()
        state = {}
        for _ in range(ms):
            X.append(obs.copy())
            A.append(teacher_act(obs, ext, sc, float(spec["success_radius"]), state=state))
            obs, r, done = env.step(student_act(obs, params, actor))
            if done: break
        succ.append(bool(env.succeeded))
        env.close()
    return X, A, float(np.mean(succ))

def train(params, X, A, Xa, Aa, epochs, rng):
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
    opt = Adam(params, lr=5e-4)
    X = np.array(X); A = np.array(A)
    for ep in range(epochs):
        idx = rng.permutation(len(X)); ida = rng.permutation(len(Xa))
        for s in range(0, len(X), 2048):
            xa = Xa[ida[s % len(Xa):s % len(Xa) + 1024]]
            aa = Aa[ida[s % len(Xa):s % len(Xa) + 1024]]
            if len(xa) < 1024:
                rep = int(np.ceil(1024 / max(1, len(xa))))
                xa = np.tile(Xa, (rep, 1))[:1024]; aa = np.tile(Aa, (rep, 1))[:1024]
            x = np.concatenate([X[idx[s:s + 2048]], xa])
            a = np.concatenate([A[idx[s:s + 2048]], aa])
            actor.load({k: params[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")})
            mub, cache = actor.forward(x)
            wpmu, hwp = wp_fwd27(x, params["wp1"], params["wp2"])
            mu = np.tanh(mub + wpmu)
            dpre = 2.0 * (mu - a) * (1.0 - mu ** 2) / (len(x) * ACT_DIM)
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
    Xa, Aa = collect_anchor()
    print(f"anchor_m2v4: {len(Xa)} samples (integrator teacher, successes only)", flush=True)
    np.savez("/workspace/t4_demos_m2v4.npz", X=np.array(Xa), A=np.array(Aa))
    Xa = np.array(Xa); Aa = np.array(Aa)
    XD, AD = [], []
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
    best = {"mean": -1e9, "round": 0}
    for k in range(1, ROUNDS + 1):
        params = {k2: v2.copy() for k2, v2 in params0.items()}
        X, A, succ = dagger_rollout(params, actor, episodes=30, seed0=600000 + 1000 * k)
        XD += X; AD += A
        print(f"round {k}: student succ {succ:.3f} dataset {len(XD)} wall={time.time()-t0:.0f}s", flush=True)
        actor = train(params, XD, AD, Xa, Aa, epochs=100, rng=rng)
        flat = pack27(actor, params["wp1"], params["wp2"])
        json.dump(flat, open(f"/workspace/t4_dag8m2v4_r{k}.json", "w"))
        res = eval_all27(flat, f"t4-dag8m2v4 r{k}")
        mean = float(np.mean(list(res.values())))
        floors = all(res[c] >= v for c, v in FLOORS.items())
        if floors and mean > best["mean"]:
            best = {"mean": mean, "round": k}
        print(f"round {k} EVAL " + json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}) +
              f" mean={mean:.3f} floors_m2={floors}", flush=True)
    print("T4DAG8M2V4_BEST " + json.dumps(best), flush=True)
    if best["round"] > 0:
        import shutil
        shutil.copy(f"/workspace/t4_dag8m2v4_r{best['round']}.json", "/workspace/t4_dag8m2v4_best.json")
        # v1-plant regression read on the best (SoC pins to 1.0 on v1)
        os.environ.pop("AUTORESEARCH_MOTOR_V2", None)
        res1 = eval_all27(json.load(open("/workspace/t4_dag8m2v4_best.json")), "t4-dag8m2v4-best-v1plant")
        os.environ["AUTORESEARCH_MOTOR_V2"] = "1"
        m1 = float(np.mean(list(res1.values())))
        print("T4DAG8M2V4_BEST_SAVED t4_dag8m2v4_r%d mean_m2=%.3f v1_regression=%s mean=%.3f" % (
            best["round"], best["mean"],
            json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res1.items()}), m1), flush=True)
    P.post_status({"training": {"status": "idle", "name": "t4_dagger8_m2v4",
                   "note": f"m2 obs-v4 DAgger done; best round {best['round']} mean {best['mean']:.3f}"}})
    print("T4DAG8M2V4_DONE wall=%.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
