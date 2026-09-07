"""Track B: history-stacked policy under est obs (parent call 2026-09-06 20:37).

The instrumented wall is the MEMORYLESS MLP: observability is fixed (ToF
mount fix, alt probe R2 0.97), teacher is competent (hover 87.5/land 100),
yet a memoryless 25-dim student cannot acquire hover/land under est obs.
History stacking (K=4 v3 est obs = 100-dim) is the cheapest policy-class
change that gives the policy an estimator-state memory: drift direction,
ToF staleness dynamics, and VO-aiding history become visible.

Student: MLP(100,4,h32), warm start = champion W1 in the newest-frame block
(cols 0-18), all other input cols zero -> functionally the champion at iter0.
Teacher: pilot_act3 on GT. Floors via passthrough GT arm (stacked GT v3).

Run:  setsid nohup /workspace/venv-vision/bin/python dagger_est_hist.py > /workspace/dagger_est_hist.log 2>&1 < /dev/null &
"""
import os, json, time
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
from collections import deque
from policy import MLP
from diverse_bc import pilot_act3, bc_train
from scenario_sampler import sample_spec, heldout_cells
from eval_scenarios import post_series, cell_dist
from parallel_rollout import parallel_episodes
from ppo_v2 import post_status, MANIFEST
from eval_estimated import EstEnv

START = os.environ.get("DAG_START", "/workspace/bc_ppo_v2_best.json")
ITERS = int(os.environ.get("DAG_ITERS", "12"))
EPS_PER_ITER = int(os.environ.get("DAG_EPS", "32"))
SCEN_MIX = ("hover_hold", "hover_hold", "land", "land", "goto")
OUT = "/workspace/bc_est_hist"
OBS_DIM = 25
K = 4
HDIM = OBS_DIM * K

def warm_start(path):
    flat19 = np.array(json.load(open(path)), dtype=np.float64)
    n19 = MLP(19, 4, seed=0); n19.set_flat(flat19)
    net = MLP(HDIM, 4, seed=0)
    net.W1 *= 0.0
    net.W1[:19, :] = n19.W1          # newest frame, v2 channels
    net.b1, net.W2, net.b2, net.W3, net.b3 = n19.b1, n19.W2, n19.b2, n19.W3, n19.b3
    return list(net.get_flat())

def _scenario_for(seed):
    rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(0x5C11))
    return SCEN_MIX[rng.integers(0, len(SCEN_MIX))]

def _make_env(dist, spec, max_steps, seed, passthrough=False):
    return EstEnv(dist, seed=seed, max_steps=max_steps, dynamics=MANIFEST,
                  scenario_spec=spec, estimated=True, obs_v2=True, obs_v3=True,
                  vo_aided=True, est_seed=int(seed) + 777, passthrough=passthrough)

def _collect_one(flat, seed):
    net = MLP(HDIM, 4, seed=0)
    net.set_flat(np.array(flat, dtype=np.float64))
    sc = _scenario_for(seed)
    spec = sample_spec(seed, force_scenario=sc)
    dist = cell_dist(seed)
    max_steps = 400 if sc == "goto" else 700
    env = _make_env(dist, spec, max_steps, seed)
    obs = env.reset()
    hist = deque([obs.copy()] * K, maxlen=K)
    traj = []
    for _ in range(max_steps):
        a = pilot_act3(env.last_gt_obs, 1.75, dist.scene_extent)
        traj.append((np.concatenate(hist), a.copy()))
        obs, r, done = env.step(net.act(np.concatenate(hist)))
        hist.appendleft(obs.copy())
        if done:
            break
    env.close()
    return traj

def _eval_one(flat, scenario, seed):
    net = MLP(HDIM, 4, seed=0)
    net.set_flat(np.array(flat, dtype=np.float64))
    spec = sample_spec(seed, force_scenario=scenario)
    dist = cell_dist(seed)
    max_steps = 400 if scenario == "goto" else 700
    env = _make_env(dist, spec, max_steps, seed)
    obs = env.reset()
    hist = deque([obs.copy()] * K, maxlen=K)
    done = False
    while not done:
        obs, r, done = env.step(net.act(np.concatenate(hist)))
        hist.appendleft(obs.copy())
    ok = bool(env.succeeded); env.close()
    return ok

def _eval_one_gt(flat, scenario, seed):
    net = MLP(HDIM, 4, seed=0)
    net.set_flat(np.array(flat, dtype=np.float64))
    spec = sample_spec(seed, force_scenario=scenario)
    dist = cell_dist(seed)
    max_steps = 400 if scenario == "goto" else 700
    env = _make_env(dist, spec, max_steps, seed, passthrough=True)
    obs = env.reset()
    hist = deque([obs.copy()] * K, maxlen=K)
    done = False
    while not done:
        obs, r, done = env.step(net.act(np.concatenate(hist)))
        hist.appendleft(obs.copy())
    ok = bool(env.succeeded); env.close()
    return ok

def main():
    t0 = time.time()
    cur = warm_start(START)
    print(f"DAGHIST start={os.path.basename(START)} K={K} -> {HDIM}-dim iters={ITERS}", flush=True)
    cells = heldout_cells()

    def eval_all(flat):
        args = [(flat, sc, s) for sc, seeds in cells.items() for s in seeds]
        res = parallel_episodes(_eval_one, args)
        out, i = {}, 0
        for sc, seeds in cells.items():
            out[sc] = float(np.mean(res[i:i + len(seeds)]))
            i += len(seeds)
        return out

    def eval_gt_all(flat):
        return {sc: float(np.mean(parallel_episodes(_eval_one_gt,
                    [(flat, sc, s) for s in cells[sc]])))
                for sc in cells}

    res0 = eval_all(cur)
    gt0 = eval_gt_all(cur)
    floors = {"goto": gt0["goto"] - 0.05, "hover_hold": gt0["hover_hold"] - 0.10,
              "land": gt0["land"] - 0.10}
    print("DAGHIST iter0 (EST hist obs): " + json.dumps({k: round(v, 3) for k, v in res0.items()})
          + " GT: " + json.dumps({k: round(v, 3) for k, v in gt0.items()}), flush=True)
    X, Y = [], []
    best_flat = list(cur)
    best_score = float(np.mean(list(res0.values())))
    json.dump(best_flat, open(OUT + "_best.json", "w"))
    for it in range(1, ITERS + 1):
        tc = time.time()
        args = [(list(cur), 9000 + it * 100 + j) for j in range(EPS_PER_ITER)]
        trajs = parallel_episodes(_collect_one, args)
        for tr in trajs:
            for o, a in tr:
                X.append(o); Y.append(a)
        if len(X) > 24000:
            X, Y = X[-24000:], Y[-24000:]
        net = bc_train(np.array(X), np.clip(np.array(Y), -0.95, 0.95),
                       iters=1500, obs_dim=HDIM, init_flat=cur)
        cur = list(net.get_flat())
        json.dump(cur, open(f"{OUT}_i{it}.json", "w"))
        res = eval_all(cur)
        gt = eval_gt_all(cur)
        score = float(np.mean(list(res.values())))
        floors_ok = all(gt[sc] >= floors[sc] for sc in floors)
        if score > best_score and floors_ok:
            best_score = score
            best_flat = list(cur)
            json.dump(best_flat, open(OUT + "_best.json", "w"))
        print(f"DAGHIST_ITER {it}: n={len(X)} wall={time.time()-tc:.0f}s EST "
              + json.dumps({k: round(v, 3) for k, v in res.items()})
              + " GT " + json.dumps({k: round(v, 3) for k, v in gt.items()})
              + f" best={best_score:.3f} floors_ok={floors_ok}", flush=True)
        for sc, v in res.items():
            post_series(f"est_success_{sc}", v, label=f"dag-hist i{it}")
        post_status({
            "candidate": {"name": "bc_est_hist_best.json",
                          "detail": "history-stacked (K=4) policy under v3 est obs",
                          "goto": res["goto"], "hover_hold": res["hover_hold"], "land": res["land"],
                          "note": f"dag-hist i{it} EST | GT g{gt['goto']:.2f} h{gt['hover_hold']:.2f} l{gt['land']:.2f}"},
            "training": {"status": "running", "name": "dagger_est_hist", "iter": it, "iters": ITERS,
                         "note": f"EST g{res['goto']:.2f} h{res['hover_hold']:.2f} l{res['land']:.2f}"},
        })
    post_status({"training": {"status": "complete", "name": "dagger_est_hist",
                              "iter": ITERS, "iters": ITERS,
                              "note": f"best EST mean {best_score:.3f}"}})
    print(f"DAGHIST_DONE best_mean={best_score:.3f} wall={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
