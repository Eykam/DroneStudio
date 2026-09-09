"""DAgger v5 under est v3 obs: v4 recipe on the GPS+ZUPT+MARKER stack (parent direction 2026-09-08: training follows the moved gate).

v3-run1, v3-run2, and hist all failed the same way: bc_train(1500 iters,
lr 3e-3) on aggregated crashing-student states overwrote the warm start in
one iteration (GT hover 62.5 -> 18.8% at i1 every variant). v4 preserves the
warm start: lr 1e-3, 300 iters, and 8/32 episodes per iter are GT-arm anchors
(passthrough; teacher labels GT states, student sees GT v3 obs) integrated
into the same training set - the anchor pulls toward GT behavior instead of
fighting it with a floor check after the damage.

Student: MLP(25,4,h32) zero-padded champion. Teacher: pilot_act3.
Run:  setsid nohup /workspace/venv-vision/bin/python dagger_est_v4.py > /workspace/dagger_est_v4.log 2>&1 < /dev/null &
"""
import os, json, time
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
from policy import MLP
from diverse_bc import pilot_act3, bc_train
from scenario_sampler import sample_spec, heldout_cells
from eval_scenarios import post_series, cell_dist
from parallel_rollout import parallel_episodes
from ppo_v2 import post_status, MANIFEST
from eval_estimated import EstEnv

START = os.environ.get("DAG_START", "/workspace/bc_ppo_v2_best.json")
ITERS = int(os.environ.get("DAG_ITERS", "14"))
EPS_PER_ITER = int(os.environ.get("DAG_EPS", "32"))
GT_ANCHOR = 8
SCEN_MIX = ("hover_hold", "hover_hold", "land", "land", "goto")
OUT = "/workspace/bc_est_dag_v4"
OBS_DIM = 25
BC_ITERS = 300
BC_LR = 1e-3

def warm_start_25(path):
    flat19 = np.array(json.load(open(path)), dtype=np.float64)
    n19 = MLP(19, 4, seed=0); n19.set_flat(flat19)
    n25 = MLP(OBS_DIM, 4, seed=0)
    n25.W1[:19, :] = n19.W1
    n25.W1[19:, :] = 0.0
    n25.b1, n25.W2, n25.b2, n25.W3, n25.b3 = n19.b1, n19.W2, n19.b2, n19.W3, n19.b3
    return list(n25.get_flat())

def _scenario_for(seed):
    rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(0x5C11))
    return SCEN_MIX[rng.integers(0, len(SCEN_MIX))]

def _make_env(dist, spec, max_steps, seed, passthrough=False):
    return EstEnv(dist, seed=seed, max_steps=max_steps, dynamics=MANIFEST,
                  scenario_spec=spec, estimated=True, obs_v2=True, obs_v3=True,
                  vo_aided=True, est_seed=int(seed) + 777, passthrough=passthrough,
                  zupt=True, gps=True, marker=True)

def _collect_one(flat, seed, gt_anchor):
    net = MLP(OBS_DIM, 4, seed=0)
    net.set_flat(np.array(flat, dtype=np.float64))
    sc = _scenario_for(seed) if not gt_anchor else ("hover_hold" if seed % 2 == 0 else "land")
    spec = sample_spec(seed, force_scenario=sc)
    dist = cell_dist(seed)
    max_steps = 400 if sc == "goto" else 700
    env = _make_env(dist, spec, max_steps, seed, passthrough=gt_anchor)
    obs = env.reset()
    traj = []
    for _ in range(max_steps):
        a = pilot_act3(env.last_gt_obs, 1.75, dist.scene_extent)
        traj.append((obs.copy(), a.copy()))
        obs, r, done = env.step(net.act(obs))
        if done:
            break
    env.close()
    return traj

def _eval_one(flat, scenario, seed):
    net = MLP(OBS_DIM, 4, seed=0)
    net.set_flat(np.array(flat, dtype=np.float64))
    spec = sample_spec(seed, force_scenario=scenario)
    dist = cell_dist(seed)
    max_steps = 400 if scenario == "goto" else 700
    env = _make_env(dist, spec, max_steps, seed)
    obs = env.reset()
    done = False
    while not done:
        obs, r, done = env.step(net.act(obs))
    ok = bool(env.succeeded); env.close()
    return ok

def _eval_one_gt(flat, scenario, seed):
    net = MLP(OBS_DIM, 4, seed=0)
    net.set_flat(np.array(flat, dtype=np.float64))
    spec = sample_spec(seed, force_scenario=scenario)
    dist = cell_dist(seed)
    max_steps = 400 if scenario == "goto" else 700
    env = _make_env(dist, spec, max_steps, seed, passthrough=True)
    obs = env.reset()
    done = False
    while not done:
        obs, r, done = env.step(net.act(obs))
    ok = bool(env.succeeded); env.close()
    return ok

def main():
    t0 = time.time()
    cur = warm_start_25(START)
    print(f"DAGV4 start={os.path.basename(START)} recipe: lr={BC_LR} iters={BC_ITERS} anchor={GT_ANCHOR}/{EPS_PER_ITER}", flush=True)
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
    print("DAGV4 iter0 (EST v3 obs): " + json.dumps({k: round(v, 3) for k, v in res0.items()})
          + " GT: " + json.dumps({k: round(v, 3) for k, v in gt0.items()}), flush=True)
    X, Y = [], []
    best_flat = list(cur)
    best_score = float(np.mean(list(res0.values())))
    json.dump(best_flat, open(OUT + "_best.json", "w"))
    for it in range(1, ITERS + 1):
        tc = time.time()
        n_est = EPS_PER_ITER - GT_ANCHOR
        args = [(list(cur), 9000 + it * 100 + j, False) for j in range(n_est)]
        args += [(list(cur), 19000 + it * 100 + j, True) for j in range(GT_ANCHOR)]
        trajs = parallel_episodes(_collect_one, args)
        for tr in trajs:
            for o, a in tr:
                X.append(o); Y.append(a)
        if len(X) > 24000:
            X, Y = X[-24000:], Y[-24000:]
        net = bc_train(np.array(X), np.clip(np.array(Y), -0.95, 0.95),
                       iters=BC_ITERS, obs_dim=OBS_DIM, init_flat=cur, lr=BC_LR)
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
        print(f"DAGV4_ITER {it}: n={len(X)} wall={time.time()-tc:.0f}s EST "
              + json.dumps({k: round(v, 3) for k, v in res.items()})
              + " GT " + json.dumps({k: round(v, 3) for k, v in gt.items()})
              + f" best={best_score:.3f} floors_ok={floors_ok}", flush=True)
        for sc, v in res.items():
            post_series(f"est_success_{sc}", v, label=f"dag-v4 i{it}")
        post_status({
            "candidate": {"name": "bc_est_dag_v4_best.json",
                          "detail": "DAgger v4 est v3 obs: warm-start-preserving recipe + GT anchors",
                          "goto": res["goto"], "hover_hold": res["hover_hold"], "land": res["land"],
                          "note": f"dag-v4 i{it} EST | GT g{gt['goto']:.2f} h{gt['hover_hold']:.2f} l{gt['land']:.2f}"},
            "training": {"status": "running", "name": "dagger_est_v4", "iter": it, "iters": ITERS,
                         "note": f"EST g{res['goto']:.2f} h{res['hover_hold']:.2f} l{res['land']:.2f}"},
        })
    post_status({"training": {"status": "complete", "name": "dagger_est_v4",
                              "iter": ITERS, "iters": ITERS,
                              "note": f"best EST mean {best_score:.3f}"}})
    print(f"DAGV4_DONE best_mean={best_score:.3f} wall={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
