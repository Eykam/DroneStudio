"""T4 stage 3: DAgger on the fresh HID-64 unified policy - ALL tiers.

Differences from t3_dagger2 (fresh-bootstrap track, user decision 2026-09-04):
- No champion anchor. The anchor's job (hold the base while adding behavior)
  is played by a 25% held-back slice of the ORIGINAL t4 demo set, mixed into
  every batch (t3 lesson: anchor weight must not decay as the set grows).
- Student rolls the full tier x scenario mix (goto/hover/land x T0-T2,
  T3 phases A/B goto), the t4 scenario-aware teacher (t4_pilot.teacher_act)
  labels every visited state.
- Retrain FROM THE BC INIT each round on the aggregated set (retrain-from-
  init beat fine-tune - t3 lesson).

Init: /workspace/t4_bc.json. Saves /workspace/t4_dag_r{k}.json per round,
evals + posts series per round. Champion file never touched.
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
from t3_pilot import t3_dist, PHASES

MANIFEST = P.MANIFEST
OBS_DIM, ACT_DIM, HID, V2_DIM = P.OBS_DIM, P.ACT_DIM, P.HID, P.V2_DIM
ROUNDS = 6

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
    for i in range(episodes):
        seed = seed0 + i
        tier = i % 4
        sc = ("goto", "hover_hold", "land")[i % 3]
        if tier == 3:
            sc = "goto"
            dist = t3_dist(seed, PHASES["A" if i % 2 == 0 else "B"])
            ms = 600
        else:
            dist = tier_dist(seed, tier)
            ms = 400 if sc == "goto" else 700
        spec = sample_spec(seed, force_scenario=sc)
        if sc != "goto":
            dist.n_waypoints = 0.0
        ext = float(dist.scene_extent)
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
    params0 = flat_to_full(json.load(open("/workspace/t4_bc.json")))
    d = np.load("/workspace/t4_demos.npz", allow_pickle=True)
    Xall, Aall = d["X"], d["A"]
    sel = rng.permutation(len(Xall))[:len(Xall) // 4]
    Xa, Aa = Xall[sel], Aall[sel]
    print(f"anchor: {len(Xa)} demo samples", flush=True)
    XD, AD = [], []
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
    for k in range(1, ROUNDS + 1):
        params = {k2: v2.copy() for k2, v2 in params0.items()}  # fresh from BC init
        X, A, succ = dagger_rollout(params, actor, episodes=24, seed0=110000 + 1000 * k)
        XD += X; AD += A
        print(f"round {k}: student succ {succ:.3f} dataset {len(XD)} wall={time.time()-t0:.0f}s", flush=True)
        actor = train(params, XD, AD, Xa, Aa, epochs=120, rng=rng)
        flat = P.pack_actor(actor, params["wp1"], params["wp2"])
        json.dump(flat, open(f"/workspace/t4_dag_r{k}.json", "w"))
        res = P.eval_all(flat, f"t4-dag r{k}")
        print(f"round {k} EVAL " + json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}), flush=True)
    P.post_status({"training": {"status": "idle", "name": "t4_dagger",
                   "note": "T4 stage 3 (DAgger x6) done"}})
    print("T4DAG_DONE wall=%.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
