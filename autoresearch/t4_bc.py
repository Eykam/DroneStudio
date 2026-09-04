"""T4 stage 2: BC on the unified all-tier demo set - fresh HID-64 net.

No champion anchor, no frozen weights (that is the point of the bootstrap):
every parameter trains from random init so T0-T2 and T3 behavior co-develop
in the 64-wide trunk. Structure per t4_common (trunk MLP 26->64->64->4 plus
bias-free wp pathway 7->32->4 added pre-tanh).

Input:  /workspace/t4_demos.npz (t4_pilot)
Output: /workspace/t4_bc.json (pack_actor flat layout)
"""
import sys, json, os, time
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from ppo import MLP, Adam
import t4_common as P

def main():
    rng = np.random.default_rng(23)
    t0 = time.time()
    d = np.load("/workspace/t4_demos.npz", allow_pickle=True)
    X, A = d["X"], d["A"]
    print(f"BC data: {len(X)} samples meta={d['meta']}", flush=True)

    actor = MLP(rng, P.OBS_DIM, P.HID, P.ACT_DIM)
    params = actor.params()
    params["wp1"] = rng.normal(0.0, 0.1, (P.WP_IN, P.WP_HID))
    params["wp2"] = np.zeros((P.WP_HID, P.ACT_DIM))
    opt = Adam(params, lr=1e-3)

    n = len(X); EPOCHS = 300; BS = 4096
    for ep in range(1, EPOCHS + 1):
        idx = rng.permutation(n)
        tot_loss = 0.0
        for s in range(0, n, BS):
            mb = idx[s:s + BS]
            x, a = X[mb], A[mb]
            actor.load({k: params[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")})
            mub, cache = actor.forward(x)
            wpmu, hwp = P.wp_forward(x, params["wp1"], params["wp2"])
            mu = np.tanh(mub + wpmu)
            loss = float(((mu - a) ** 2).mean()); tot_loss += loss * len(mb)
            dpre = 2.0 * (mu - a) * (1.0 - mu ** 2) / (len(mb) * P.ACT_DIM)
            grads = actor.backward(dpre, cache)
            grads["wp2"] = hwp.T @ dpre
            dh = (dpre @ params["wp2"].T) * (hwp > 0)
            grads["wp1"] = x[:, P.V2_DIM:].T @ dh
            tot = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
            if tot > 1.0:
                for k in grads: grads[k] *= 1.0 / (tot + 1e-8)
            opt.step(params, grads)
        if ep % 50 == 0:
            print(f"bc ep{ep} loss={tot_loss/n:.5f} wall={time.time()-t0:.0f}s", flush=True)

    actor.load({k: params[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")})
    flat = P.pack_actor(actor, params["wp1"], params["wp2"])
    json.dump(flat, open("/workspace/t4_bc.json", "w"))

    res = P.eval_all(flat, "t4-bc")
    print("T4BC_EVAL " + json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}), flush=True)
    P.post_status({"training": {"status": "idle", "name": "t4_bc",
        "note": "T4 HID-64 bootstrap stage 2 (BC) done: " +
                " ".join(f"{sc}t{t}={v:.2f}" for (sc, t), v in res.items())}})
    print("T4BC_DONE wall=%.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
