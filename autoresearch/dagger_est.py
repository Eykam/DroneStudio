"""DAgger under ESTIMATED obs (hover/land rescue track).

Student rolls out on est obs (EstEnv: 500Hz ESKF + mag/ToF + synthetic-VO);
teacher pilot_act3 labels each visited state from the GT obs at the SAME
state - dense per-step correction even where student success is 0% (PPO
could not climb that). Warm start bc_ppo_est_best (holds the +10pt goto
gain). GT-goto regression floor preserved from ppo_est discipline.

Run:  cd /workspace/DroneStudio/autoresearch &&
      setsid nohup /workspace/venv-vision/bin/python dagger_est.py > /workspace/dagger_est.log 2>&1 < /dev/null &
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

START = os.environ.get("DAG_START", "/workspace/bc_ppo_est_best.json")
ITERS = int(os.environ.get("DAG_ITERS", "8"))
EPS_PER_ITER = int(os.environ.get("DAG_EPS", "32"))
SCEN_MIX = ("land", "land", "hover_hold", "goto")
OUT = "/workspace/bc_est_dag"

def _scenario_for(seed):
    rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(0x5C11))
    return SCEN_MIX[rng.integers(0, len(SCEN_MIX))]

def _make_env(dist, spec, max_steps, seed):
    return EstEnv(dist, seed=seed, max_steps=max_steps, dynamics=MANIFEST,
                  scenario_spec=spec, estimated=True, obs_v2=True,
                  vo_aided=True, est_seed=int(seed) + 777)

def _collect_one(flat, seed):
    net = MLP(19, 4, seed=0)
    net.set_flat(np.array(flat, dtype=np.float64))
    sc = _scenario_for(seed)
    spec = sample_spec(seed, force_scenario=sc)
    dist = cell_dist(seed)
    max_steps = 400 if sc == "goto" else 700
    env = _make_env(dist, spec, max_steps, seed)
    obs = env.reset()
    traj = []
    for _ in range(max_steps):
        a = pilot_act3(env.last_gt_obs, 1.75, dist.scene_extent)  # teacher on GT
        traj.append((obs.copy(), a.copy()))                        # student sees EST
        obs, r, done = env.step(net.act(obs))
        if done:
            break
    env.close()
    return traj

def _eval_one(flat, scenario, seed):
    net = MLP(19, 4, seed=0)
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
    from env_sim import make_sim_factory
    net = MLP(19, 4, seed=0)
    net.set_flat(np.array(flat, dtype=np.float64))
    spec = sample_spec(seed, force_scenario=scenario)
    dist = cell_dist(seed)
    max_steps = 400 if scenario == "goto" else 700
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    done = False
    while not done:
        obs, r, done = env.step(net.act(obs))
    ok = bool(env.succeeded); env.close()
    return ok

def main():
    t0 = time.time()
    net = MLP(19, 4, seed=0)
    net.set_flat(np.array(json.load(open(START)), dtype=np.float64))
    print(f"DAGGEREST start={os.path.basename(START)} iters={ITERS} eps/iter={EPS_PER_ITER}", flush=True)
    cells = heldout_cells()

    def eval_all(flat):
        args = [(flat, sc, s) for sc, seeds in cells.items() for s in seeds]
        res = parallel_episodes(_eval_one, args)
        out, i = {}, 0
        for sc, seeds in cells.items():
            out[sc] = float(np.mean(res[i:i + len(seeds)]))
            i += len(seeds)
        return out

    def eval_gt_goto(flat):
        args = [(flat, "goto", s) for s in cells["goto"]]
        return float(np.mean(parallel_episodes(_eval_one_gt, args)))

    flat = list(net.get_flat())
    res0 = eval_all(flat)
    gt0 = eval_gt_goto(flat)
    gt_floor = gt0 - 0.05
    print("DAGGEREST iter0 (EST obs): " + json.dumps({k: round(v, 3) for k, v in res0.items()})
          + f" gt_goto={gt0:.3f} floor={gt_floor:.3f}", flush=True)
    for sc, v in res0.items():
        post_series(f"est_success_{sc}", v, label="dagger-est i0")
    X, Y = [], []
    best_flat = list(flat)
    best_score = float(np.mean(list(res0.values())))
    json.dump(best_flat, open(OUT + "_best.json", "w"))
    cur = list(flat)
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
                       iters=1500, obs_dim=19, init_flat=cur)
        cur = list(net.get_flat())
        json.dump(cur, open(f"{OUT}_i{it}.json", "w"))
        res = eval_all(cur)
        gtg = eval_gt_goto(cur)
        score = float(np.mean(list(res.values())))
        print(f"DAGGEREST_ITER {it}: n={len(X)} wall={time.time()-tc:.0f}s " +
              json.dumps({k: round(v, 3) for k, v in res.items()}) +
              f" gt_goto={gtg:.3f} best={best_score:.3f}", flush=True)
        for sc, v in res.items():
            post_series(f"est_success_{sc}", v, label=f"dagger-est i{it}")
        if score > best_score and gtg >= gt_floor:
            best_score = score
            best_flat = list(cur)
            json.dump(best_flat, open(OUT + "_best.json", "w"))
            print(f"DAGGEREST_SAVE iter={it} mean={score:.3f}", flush=True)
        post_status({
            "live_policy": {"name": "bc_ppo_v2_best.json",
                            "detail": "obs v2 (19-dim) GT-trained champion",
                            "note": "flying on /watch"},
            "candidate": {"name": "bc_est_dag_best.json",
                          "detail": "DAgger under estimated obs (GT teacher, est student)",
                          "goto": res["goto"], "hover_hold": res["hover_hold"], "land": res["land"],
                          "note": f"dagger-est i{it} EST-obs eval | GT-goto {gtg:.2f} (floor {gt_floor:.2f})"},
            "training": {"status": "running", "name": "dagger_est", "iter": it, "iters": ITERS,
                         "note": f"EST goto {res['goto']:.2f} hover {res['hover_hold']:.2f} land {res['land']:.2f}"},
            "queue": ["depth-net v1.1 retrain (vision track)"],
        })
    post_status({"training": {"status": "complete", "name": "dagger_est",
                              "iter": ITERS, "iters": ITERS,
                              "note": f"best EST mean {best_score:.3f}"}})
    print(f"DAGGEREST_DONE best_mean={best_score:.3f} wall={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
