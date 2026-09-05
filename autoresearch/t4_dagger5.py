"""T4 stage 5: land-precision + hover-balance DAgger from t4_best (= landfix2 dag2_r4).

landfix2 won land_t2 (0.938) but REGRESSED land_t0 to 0.375 (was 0.75):
touchdowns land 0.5-1.3m off ~0.5m pads (sink rate fine, 0.07-0.17 m/s vs
0.5 limit), 2/16 hover over the pad and never commit. Root process bug:
round selection floored only goto_t0/hover_t0 - land_t0 was unprotected.
This round: land rollouts biased to tier-0 precision (4 of 6 land eps),
hover_hold (now with 2-60s holds) + goto guards; floors add land_t0>=0.7;
best passing round (max mean) becomes t4_best. Outputs t4_dag5_r{k}.json.

Init: /workspace/t4_best.json. Saves /workspace/t4_dag5_r{k}.json.
"""
import sys, json, os, time
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from ppo import MLP, Adam
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
import t4_common as P
from t4_pilot import teacher_act

MANIFEST = P.MANIFEST
OBS_DIM, ACT_DIM, HID, V2_DIM = P.OBS_DIM, P.ACT_DIM, P.HID, P.V2_DIM
ROUNDS = 8

def flat_to_full(flat):
    trunk = P.bc_to_actor_params(flat)
    wp1, wp2 = P.unpack_wp(flat)
    return {**trunk, "wp1": wp1, "wp2": wp2}

def student_act(obs, params, actor):
    actor.load({k: params[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")})
    mub, _ = actor.forward(obs[None, :])
    wpmu, _ = P.wp_forward(obs[None, :], params["wp1"], params["wp2"])
    return np.tanh(mub[0] + wpmu[0])

def dagger_rollout(params, actor, episodes, seed0):
    X, A, succ = [], [], []
    from scenario_sampler import hover_max_steps
    for i in range(episodes):
        seed = seed0 + i
        m = i % 10
        if m < 4:
            sc, tier = "land", (0 if m < 3 else 2)  # 3x T0, 1x T2
        elif m < 8:
            sc, tier = "hover_hold", i % 3
        else:
            sc, tier = "goto", i % 3
        dist = tier_dist(seed, tier)
        spec = sample_spec(seed, force_scenario=sc)
        dist.n_waypoints = 0.0
        ext = float(dist.scene_extent)
        ms = (400 if sc == "goto"
              else hover_max_steps(spec.get("hold_s", 4.0), tier) if sc == "hover_hold"
              else 700)
        env = make_sim_factory(dist, max_steps=ms, dynamics=MANIFEST,
                               scenario_spec=spec)(seed)
        obs = env.reset()
        for _ in range(ms):
            X.append(obs.copy())
            A.append(teacher_act(obs, ext, sc, float(spec["success_radius"])))
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
            wpmu, hwp = P.wp_forward(x, params["wp1"], params["wp2"])
            mu = np.tanh(mub + wpmu)
            dpre = 2.0 * (mu - a) * (1.0 - mu ** 2) / (len(x) * ACT_DIM)
            grads = actor.backward(dpre, cache)
            grads["wp2"] = hwp.T @ dpre
            dh = (dpre @ params["wp2"].T) * (hwp > 0)
            grads["wp1"] = x[:, V2_DIM:].T @ dh
            tot = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
            if tot > 1.0:
                for k in grads: grads[k] *= 1.0 / (tot + 1e-8)
            opt.step(params, grads)
    actor.load({k: params[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")})
    return actor

def main():
    rng = np.random.default_rng(23)
    t0 = time.time()
    params0 = flat_to_full(json.load(open("/workspace/t4_best.json")))
    d = np.load("/workspace/t4_demos.npz", allow_pickle=True)
    sel = rng.permutation(len(d["X"]))[:len(d["X"]) // 4]
    d2 = np.load("/workspace/t4_demos_land.npz", allow_pickle=True)
    Xa = np.concatenate([d["X"][sel], d2["X"]])
    Aa = np.concatenate([d["A"][sel], d2["A"]])
    print(f"anchor: {len(Xa)} (25% original + {len(d2['X'])} land-enriched)", flush=True)
    XD, AD = [], []
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
    best = {"mean": -1e9, "round": 0}
    for k in range(1, ROUNDS + 1):
        params = {k2: v2.copy() for k2, v2 in params0.items()}  # fresh from t4_best
        X, A, succ = dagger_rollout(params, actor, episodes=30, seed0=350000 + 1000 * k)
        XD += X; AD += A
        print(f"round {k}: student succ {succ:.3f} dataset {len(XD)} wall={time.time()-t0:.0f}s", flush=True)
        actor = train(params, XD, AD, Xa, Aa, epochs=100, rng=rng)
        flat = P.pack_actor(actor, params["wp1"], params["wp2"])
        json.dump(flat, open(f"/workspace/t4_dag5_r{k}.json", "w"))
        res = P.eval_all(flat, f"t4-dag5 r{k}")
        mean = float(np.mean(list(res.values())))
        floors = (res[("goto", 0)] >= 0.9 and res[("hover_hold", 0)] >= 0.6
                  and res[("land", 0)] >= 0.7)
        if floors and mean > best["mean"]:
            best = {"mean": mean, "round": k}
        print(f"round {k} EVAL " + json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}) +
              f" mean={mean:.3f} floors={floors}", flush=True)
    print("T4DAG5_BEST " + json.dumps(best), flush=True)
    if best["round"] > 0:
        import shutil
        shutil.copy(f"/workspace/t4_dag5_r{best['round']}.json", "/workspace/t4_best.json")
        print(f"T4BEST_UPDATED t4_dag5_r{best['round']} mean={best['mean']:.3f}", flush=True)
    P.post_status({"training": {"status": "idle", "name": "t4_dagger5",
                   "note": f"T4 land-precision DAgger done; best round {best['round']} mean {best['mean']:.3f}"}})
    print("T4DAG5_DONE wall=%.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
