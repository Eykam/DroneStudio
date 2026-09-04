"""BC v2: anchored BC - T3 teacher demos + champion-anchor on T0-T2.

v1 (wp-pathway-only) could not clone the teacher (loss .017, clone succ 0):
the correction needs the full state, not just wp channels. v2 trains
w1[19:26] + w2/b2/w3/b3 + wp-MLP, anchored: champion-behavior MSE on
T0-T2 states in the same batches, so proven behavior is preserved by the
loss, not by freezing. Eval checks every tier after training.

(orig) BC the T3 teacher into the waypoint pathway only.

Trunk stays at the champion (bc_ppo_v2_best zero-pad); only w1 rows 19:26
(the obs v3.1 waypoint channels) and the bias-free wp-MLP (7->32->4) train.
T0-T2 remain bit-exact by construction (wp channels exactly zero there).

Demos: phases A/B only (teacher succ .50/.31; C/D teach dying). Successful
episodes only - the mixed-quality lesson from earlier BC rounds.

Output: /workspace/bc_t3_wp2.json - pack_actor layout (trunk flat + wp tail),
the ppo_v4 warm start.
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

MANIFEST = P.MANIFEST
OBS_DIM, ACT_DIM, HID, V2_DIM = P.OBS_DIM, P.ACT_DIM, P.HID, P.V2_DIM
WP_IN, WP_HID = P.WP_IN, P.WP_HID

def collect(episodes=64, seed0=50000):
    X, A, kept = [], [], 0
    for i in range(episodes):
        seed = seed0 + i
        knobs = PHASES["A" if i % 2 == 0 else "B"]
        dist = t3_dist(seed, knobs)
        spec = sample_spec(seed, force_scenario="goto")
        ext = float(dist.scene_extent)
        env = make_sim_factory(dist, max_steps=600, dynamics=MANIFEST,
                               scenario_spec=spec)(seed)
        obs = env.reset()
        traj = []
        for _ in range(600):
            a = teacher_act(obs, ext)
            traj.append((obs.copy(), a.copy()))
            obs, r, done = env.step(a)
            if done: break
        ok = bool(env.succeeded)
        env.close()
        if ok:
            kept += 1
            X += [t[0] for t in traj]; A += [t[1] for t in traj]
        if i % 8 == 7:
            print(f"collect {i+1}/{episodes} kept={kept} samples={len(X)}", flush=True)
    return np.array(X), np.array(A), kept

def collect_anchor(rng, episodes=32, seed0=70000):
    """Champion behavior-clone targets on T0-T2 states (all steps, success
    or not - the anchor is the champion's action, whatever it is)."""
    champ = P.warm_start_params()
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM); actor.load(champ)
    X, A = [], []
    scs = ("goto", "hover_hold", "land")
    for i in range(episodes):
        seed = seed0 + i
        sc = scs[i % 3]; tier = i % 3  # T0/T1/T2 rotation
        dist = tier_dist(seed, tier)
        spec = sample_spec(seed, force_scenario=sc)
        if sc != "goto":
            dist.n_waypoints = 0.0
        env = make_sim_factory(dist, max_steps=400, dynamics=MANIFEST,
                               scenario_spec=spec)(seed)
        obs = env.reset()
        for _ in range(400):
            mu, _ = actor.forward(obs[None, :])
            a = np.tanh(mu[0])
            X.append(obs.copy()); A.append(a.copy())
            obs, r, done = env.step(a)
            if done: break
        env.close()
    return np.array(X), np.array(A)

def main():
    global params
    rng = np.random.default_rng(11)
    t0 = time.time()
    X, A, kept = collect()
    print(f"BC data: {len(X)} T3 samples from {kept} successful demos", flush=True)
    Xa, Aa = collect_anchor(rng)
    print(f"anchor: {len(Xa)} T0-T2 champion samples", flush=True)
    X = np.concatenate([X, Xa]); A = np.concatenate([A, Aa])

    champ = P.warm_start_params()
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
    params = {"w1wp": np.zeros((WP_IN, HID)),
              "w2": champ["w2"].copy(), "b2": champ["b2"].copy(),
              "w3": champ["w3"].copy(), "b3": champ["b3"].copy(),
              "wp1": rng.normal(0.0, 0.1, (WP_IN, WP_HID)),
              "wp2": np.zeros((WP_HID, ACT_DIM))}
    opt = Adam(params, lr=1e-3)

    n = len(X); EPOCHS = 300; BS = 4096
    for ep in range(1, EPOCHS + 1):
        idx = rng.permutation(n)
        tot_loss = 0.0
        for s in range(0, n, BS):
            mb = idx[s:s + BS]
            x, a = X[mb], A[mb]
            w1 = champ["w1"].copy(); w1[V2_DIM:] = params["w1wp"]
            actor.load({**champ, "w1": w1, "w2": params["w2"], "b2": params["b2"],
                        "w3": params["w3"], "b3": params["b3"]})
            mub, cache = actor.forward(x)
            wpmu, hwp = P.wp_forward(x, params["wp1"], params["wp2"])
            mu = np.tanh(mub + wpmu)
            loss = float(((mu - a) ** 2).mean()); tot_loss += loss * len(mb)
            # d loss/d pre-tanh = 2(mu-a)(1-mu^2)/N
            dpre = 2.0 * (mu - a) * (1.0 - mu ** 2) / (len(mb) * ACT_DIM)
            gr = actor.backward(dpre, cache)
            g_w1wp = gr["w1"][V2_DIM:]
            gwp2 = hwp.T @ dpre
            dh = (dpre @ params["wp2"].T) * (hwp > 0)
            gwp1 = x[:, V2_DIM:].T @ dh
            grads = {"w1wp": g_w1wp, "wp1": gwp1, "wp2": gwp2,
                     "w2": gr["w2"], "b2": gr["b2"], "w3": gr["w3"], "b3": gr["b3"]}
            tot = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
            if tot > 1.0:
                for k in grads: grads[k] *= 1.0 / (tot + 1e-8)
            opt.step(params, grads)
        if ep % 50 == 0:
            print(f"bc ep{ep} loss={tot_loss/n:.5f} wall={time.time()-t0:.0f}s", flush=True)

    w1 = champ["w1"].copy(); w1[V2_DIM:] = params["w1wp"]
    actor.load({**champ, "w1": w1, "w2": params["w2"], "b2": params["b2"],
                "w3": params["w3"], "b3": params["b3"]})
    flat = P.pack_actor(type("P", (), {"actor": actor}), params["wp1"], params["wp2"])
    json.dump(flat, open("/workspace/bc_t3_wp2.json", "w"))

    # eval: held-out goto_t3 + T0-T2 parity spot check
    cells = heldout_cells_tiered()
    def run_cell(sc, t):
        oks = [P.eval_one(flat, sc, t, s) for s in cells[(sc, t)]]
        return float(np.mean(oks))
    res = {(sc, t): run_cell(sc, t) for (sc, t) in
           [("goto", 0), ("goto", 1), ("goto", 2), ("goto", 3),
            ("hover_hold", 0), ("hover_hold", 2), ("land", 0), ("land", 2)]}
    print("T3BC2_EVAL " + json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}), flush=True)

    # clone vs teacher on fresh A/B scenes
    for ph in ("A", "B"):
        oks, wps = [], []
        for i in range(16):
            seed = 60000 + i
            dist = t3_dist(seed, PHASES[ph]); spec = sample_spec(seed, force_scenario="goto")
            env = make_sim_factory(dist, max_steps=600, dynamics=MANIFEST, scenario_spec=spec)(seed)
            obs = env.reset(); nwp = len(getattr(env, "waypoints", []))
            actor2 = MLP(np.random.default_rng(0), OBS_DIM, HID, ACT_DIM)
            actor2.load(P.bc_to_actor_params(flat)); wp1, wp2 = P.unpack_wp(flat)
            for _ in range(600):
                mub, _ = actor2.forward(obs[None, :])
                wpmu, _ = P.wp_forward(obs[None, :], wp1, wp2)
                obs, r, done = env.step(np.tanh(mub[0] + wpmu[0]))
                if done: break
            oks.append(bool(env.succeeded)); wps.append(int(env.last_info.get("wp", 0)) / max(1, nwp))
            env.close()
        print(f"T3BC2_CLONE phase {ph}: succ {np.mean(oks):.3f} wp_frac {np.mean(wps):.3f}", flush=True)
    print("T3BC2_DONE wall=%.0fs" % (time.time() - t0), flush=True)

params = None
if __name__ == "__main__":
    main()
