"""DAgger-T3: teacher labels the student's OWN visited states, anchored.

bc2 (anchored BC) got the first nonzero goto_t3 but regressed T0-T2 via
covariate shift. DAgger closes that: each round rolls the current student
on fresh phase A/B scenes, labels every visited state with the scripted
teacher, aggregates, retrains (with the fixed champion anchor set), so the
loss always covers the student's real state distribution.

Init: /workspace/bc_t3_wp2.json (anchored-BC policy). Saves
/workspace/bc_t3_dag_r{k}.json per round. Champion file never touched.
"""
import sys, json, os, time
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from ppo import MLP, Adam
from scenario_sampler import sample_spec, tier_dist, heldout_cells_tiered
from env_sim import make_sim_factory
import ppo_v39 as P
from t3_pilot import teacher_act, t3_dist, PHASES
from t3_bc2 import collect_anchor

MANIFEST = P.MANIFEST
OBS_DIM, ACT_DIM, HID, V2_DIM = P.OBS_DIM, P.ACT_DIM, P.HID, P.V2_DIM
WP_IN, WP_HID = P.WP_IN, P.WP_HID
ROUNDS = 6

def flat_to_params(flat):
    trunk = P.bc_to_actor_params(flat)
    wp1, wp2 = P.unpack_wp(flat)
    return trunk, wp1, wp2

def student_act(obs, actor, wp1, wp2):
    mub, _ = actor.forward(obs[None, :])
    wpmu, _ = P.wp_forward(obs[None, :], wp1, wp2)
    return np.tanh(mub[0] + wpmu[0])

def dagger_rollout(actor, wp1, wp2, episodes, seed0):
    """Student rolls, teacher labels every visited state."""
    X, A, succ, wpf = [], [], [], []
    for i in range(episodes):
        seed = seed0 + i
        knobs = PHASES["A" if i % 2 == 0 else "B"]
        dist = t3_dist(seed, knobs)
        spec = sample_spec(seed, force_scenario="goto")
        ext = float(dist.scene_extent)
        env = make_sim_factory(dist, max_steps=600, dynamics=MANIFEST,
                               scenario_spec=spec)(seed)
        obs = env.reset(); nwp = len(getattr(env, "waypoints", []))
        for _ in range(600):
            a_teach = teacher_act(obs, ext)
            X.append(obs.copy()); A.append(a_teach.copy())
            obs, r, done = env.step(student_act(obs, actor, wp1, wp2))
            if done: break
        succ.append(bool(env.succeeded))
        wpf.append(int(env.last_info.get("wp", 0)) / max(1, nwp))
        env.close()
    return X, A, float(np.mean(succ)), float(np.mean(wpf))

def train(params, X, A, Xa, Aa, epochs, rng):
    champ = P.warm_start_params()
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
    opt = Adam(params, lr=5e-4)
    X = np.array(X); A = np.array(A)
    n = len(X) + len(Xa)
    for ep in range(epochs):
        idx = rng.permutation(len(X)); ida = rng.permutation(len(Xa))
        for s in range(0, max(len(X), len(Xa)), 4096):
            x = np.concatenate([X[idx[s:s + 4096]], Xa[ida[s % len(Xa):s % len(Xa) + 2048]]])
            a = np.concatenate([A[idx[s:s + 4096]], Aa[ida[s % len(Xa):s % len(Xa) + 2048]]])
            w1 = champ["w1"].copy(); w1[V2_DIM:] = params["w1wp"]
            actor.load({**champ, "w1": w1, "w2": params["w2"], "b2": params["b2"],
                        "w3": params["w3"], "b3": params["b3"]})
            mub, cache = actor.forward(x)
            wpmu, hwp = P.wp_forward(x, params["wp1"], params["wp2"])
            mu = np.tanh(mub + wpmu)
            dpre = 2.0 * (mu - a) * (1.0 - mu ** 2) / (len(x) * ACT_DIM)
            gr = actor.backward(dpre, cache)
            gwp2 = hwp.T @ dpre
            dh = (dpre @ params["wp2"].T) * (hwp > 0)
            gwp1 = x[:, V2_DIM:].T @ dh
            grads = {"w1wp": gr["w1"][V2_DIM:], "wp1": gwp1, "wp2": gwp2,
                     "w2": gr["w2"], "b2": gr["b2"], "w3": gr["w3"], "b3": gr["b3"]}
            tot = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
            if tot > 1.0:
                for k in grads: grads[k] *= 1.0 / (tot + 1e-8)
            opt.step(params, grads)
    w1 = champ["w1"].copy(); w1[V2_DIM:] = params["w1wp"]
    actor.load({**champ, "w1": w1, "w2": params["w2"], "b2": params["b2"],
                "w3": params["w3"], "b3": params["b3"]})
    return actor

def main():
    rng = np.random.default_rng(23)
    t0 = time.time()
    flat0 = json.load(open("/workspace/bc_t3_wp2.json"))
    trunk, wp1, wp2 = flat_to_params(flat0)
    params = {"w1wp": trunk["w1"][V2_DIM:].copy(), "w2": trunk["w2"].copy(),
              "b2": trunk["b2"].copy(), "w3": trunk["w3"].copy(),
              "b3": trunk["b3"].copy(), "wp1": wp1, "wp2": wp2}
    Xa, Aa = collect_anchor(rng)
    print(f"anchor: {len(Xa)} champion samples", flush=True)
    XD, AD = [], []
    for k in range(1, ROUNDS + 1):
        w1 = P.warm_start_params()["w1"].copy(); w1[V2_DIM:] = params["w1wp"]
        actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
        actor.load({**P.warm_start_params(), "w1": w1, "w2": params["w2"],
                    "b2": params["b2"], "w3": params["w3"], "b3": params["b3"]})
        X, A, succ, wpf = dagger_rollout(actor, params["wp1"], params["wp2"],
                                         episodes=24, seed0=80000 + 1000 * k)
        XD += X; AD += A
        print(f"round {k}: student succ {succ:.3f} wp_frac {wpf:.3f} "
              f"dataset {len(XD)} wall={time.time()-t0:.0f}s", flush=True)
        actor = train(params, XD, AD, Xa, Aa, epochs=120, rng=rng)
        flat = P.pack_actor(type("Pk", (), {"actor": actor}), params["wp1"], params["wp2"])
        json.dump(flat, open(f"/workspace/bc_t3_dag_r{k}.json", "w"))
        # quick student eval on fresh A/B
        oks, wps = [], []
        for i in range(16):
            seed = 90000 + i
            dist = t3_dist(seed, PHASES["A" if i % 2 == 0 else "B"])
            spec = sample_spec(seed, force_scenario="goto")
            env = make_sim_factory(dist, max_steps=600, dynamics=MANIFEST, scenario_spec=spec)(seed)
            obs = env.reset(); nwp = len(getattr(env, "waypoints", []))
            for _ in range(600):
                obs, r, done = env.step(student_act(obs, actor, params["wp1"], params["wp2"]))
                if done: break
            oks.append(bool(env.succeeded)); wps.append(int(env.last_info.get("wp", 0)) / max(1, nwp))
            env.close()
        print(f"round {k} EVAL fresh A/B: succ {np.mean(oks):.3f} wp_frac {np.mean(wps):.3f}", flush=True)

    flat = json.load(open(f"/workspace/bc_t3_dag_r{ROUNDS}.json"))
    cells = heldout_cells_tiered()
    res = {}
    for (sc, t) in [("goto", 0), ("goto", 1), ("goto", 2), ("goto", 3),
                    ("hover_hold", 0), ("hover_hold", 2), ("land", 0), ("land", 2)]:
        res[(sc, t)] = float(np.mean([P.eval_one(flat, sc, t, s) for s in cells[(sc, t)]]))
    print("T3DAG_EVAL " + json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}), flush=True)
    print("T3DAG_DONE wall=%.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
