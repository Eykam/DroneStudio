"""dag7m2 (2026-09-04): motor_v2-fidelity DAgger campaign.

Why: t4_best (v1-plant champion, mean 0.799) collapses to 0.083 under
motor_v2 (series t4best-m2-baseline) - the deployment lineage does not
transfer to the calibrated RS2205 EM/battery plant. This campaign trains
against motor_v2 directly: AUTORESEARCH_MOTOR_V2=1 for the whole process
(rollouts AND eval - series stay comparable to the m2 baseline).

Teacher: t4_pilot.teacher_act, retuned for m2 (dc867e0/d0390eb; env-gated
gains - under m2 it yields goto 0.938 / land 0.438 / hover ~0.31-0.38).

Floors RE-SCOPED for m2 (parent-approved 10:28 PM): the v1 floors
(goto>=0.9/hover>=0.6/land>=0.7) are unreachable under m2 - even the
teacher cannot meet them. m2-relative floors: goto_t0>=0.8 AND
hover_hold_t0>=0.3 AND land_t0>=0.35 (teacher-minus-margin). v1 regression
tracked as a secondary read on the final best, not a per-round gate.

Anchors: v1-plant demo anchors (t4_demos*.npz) are DROPPED - their labels
come from the v1 teacher and fight the m2 trim behavior. Replaced with a
fresh m2-native anchor collected in-process: teacher-flown episodes under
m2 (successes only), same 40/40/20 land/hover/goto oversampling.

Init: /workspace/t4_best.json (warm chain, same as dag6j2). Own champion
file /workspace/t4_dag7m2_best.json - does NOT touch t4_best.json.
Outputs t4_dag7m2_r{k}.json. Fixes dag6j2's best-copy bug (copied from
the wrong campaign prefix - t4_dag6j2_best.json was run-1 r3, corrected).
"""
import sys, json, os, time
os.environ["AUTORESEARCH_MOTOR_V2"] = "1"  # process-wide: rollouts AND eval
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from ppo import MLP, Adam
from scenario_sampler import sample_spec, tier_dist, hover_max_steps
from env_sim import make_sim_factory
import t4_common as P
from t4_pilot import teacher_act

MANIFEST = P.MANIFEST
OBS_DIM, ACT_DIM, HID, V2_DIM = P.OBS_DIM, P.ACT_DIM, P.HID, P.V2_DIM
ROUNDS = 8
FLOORS = {("goto", 0): 0.8, ("hover_hold", 0): 0.3, ("land", 0): 0.35}

def flat_to_full(flat):
    trunk = P.bc_to_actor_params(flat)
    wp1, wp2 = P.unpack_wp(flat)
    return {**trunk, "wp1": wp1, "wp2": wp2}

def student_act(obs, params, actor):
    actor.load({k: params[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")})
    mub, _ = actor.forward(obs[None, :])
    wpmu, _ = P.wp_forward(obs[None, :], params["wp1"], params["wp2"])
    return np.tanh(mub[0] + wpmu[0])

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

def collect_m2_anchor(seed0=770000, target=60000):
    """Teacher-flown m2 episodes; successes only (labels are the teacher
    itself, so failures are label noise)."""
    X, A = [], []
    for i in range(120):  # oversample attempts; teacher yields ~0.35-0.94 by scenario
        seed = seed0 + i
        sc, tier = _mix(i)
        env, ext, spec, ms = _episode(seed, sc, tier)
        obs = env.reset()
        xs, acts = [], []
        for _ in range(ms):
            xs.append(obs.copy())
            acts.append(teacher_act(obs, ext, sc, float(spec["success_radius"])))
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
    rng = np.random.default_rng(24)
    t0 = time.time()
    params0 = flat_to_full(json.load(open("/workspace/t4_best.json")))
    Xa, Aa = collect_m2_anchor()
    print(f"anchor_m2: {len(Xa)} samples (teacher-flown under motor_v2, successes only)", flush=True)
    np.savez("/workspace/t4_demos_m2.npz", X=np.array(Xa), A=np.array(Aa))
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
        flat = P.pack_actor(actor, params["wp1"], params["wp2"])
        json.dump(flat, open(f"/workspace/t4_dag7m2_r{k}.json", "w"))
        res = P.eval_all(flat, f"t4-dag7m2 r{k}")
        mean = float(np.mean(list(res.values())))
        floors = all(res[c] >= v for c, v in FLOORS.items())
        if floors and mean > best["mean"]:
            best = {"mean": mean, "round": k}
        print(f"round {k} EVAL " + json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}) +
              f" mean={mean:.3f} floors_m2={floors}", flush=True)
    print("T4DAG7M2_BEST " + json.dumps(best), flush=True)
    if best["round"] > 0:
        import shutil
        shutil.copy(f"/workspace/t4_dag7m2_r{best['round']}.json", "/workspace/t4_dag7m2_best.json")
        print(f"T4DAG7M2_BEST_SAVED t4_dag7m2_r{best['round']} mean={best['mean']:.3f}", flush=True)
    P.post_status({"training": {"status": "idle", "name": "t4_dagger7_m2",
                   "note": f"m2-fidelity DAgger done; best round {best['round']} mean {best['mean']:.3f} (m2-relative floors)"}})
    print("T4DAG7M2_DONE wall=%.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
