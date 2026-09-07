"""Track A: GT-obs hover/land specialist policies (parent call 2026-09-06 20:37).

Teacher pilot_act3 is competent (hover 87.5%, land 100% on eval cells) and GT
obs is perfectly decodable - so GT-arm DAgger on student-visited states is the
guaranteed path to working hover/land skills in sim. One specialist per
scenario, warm start from champion bc_ppo_v2_best, accept on specialist-scenario
improvement. Cross-scenario numbers reported for the record (specialists are
dispatched per-scenario; the champion remains the generalist floor).

Run:  setsid nohup /workspace/venv-vision/bin/python dagger_gt_specialist.py hover_hold > /workspace/dag_gt_hover.log 2>&1 < /dev/null &
"""
import os, json, sys, time
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
from policy import MLP
from diverse_bc import pilot_act3, bc_train
from scenario_sampler import sample_spec, heldout_cells
from eval_scenarios import post_series, cell_dist
from parallel_rollout import parallel_episodes
from ppo_v2 import post_status, MANIFEST
from env_sim import make_sim_factory

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "hover_hold"
assert SCENARIO in ("hover_hold", "land")
START = "/workspace/bc_ppo_v2_best.json"
OUT = f"/workspace/bc_gt_{SCENARIO}"
ITERS = int(os.environ.get("DAG_ITERS", "10"))
EPS_PER_ITER = int(os.environ.get("DAG_EPS", "24"))

def _collect_one(flat, seed):
    net = MLP(19, 4, seed=0)
    net.set_flat(np.array(flat, dtype=np.float64))
    spec = sample_spec(seed, force_scenario=SCENARIO)
    dist = cell_dist(seed)
    max_steps = 700
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    traj = []
    for _ in range(max_steps):
        a = pilot_act3(obs, 1.75, dist.scene_extent)   # teacher labels GT obs
        traj.append((obs.copy(), a.copy()))
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
    env = make_sim_factory(dist, max_steps=400 if scenario == "goto" else 700,
                           dynamics=MANIFEST, scenario_spec=spec)(seed)
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
    cells = heldout_cells()

    def eval_all(flat):
        args = [(flat, sc, s) for sc, seeds in cells.items() for s in seeds]
        res = parallel_episodes(_eval_one, args)
        out, i = {}, 0
        for sc, seeds in cells.items():
            out[sc] = float(np.mean(res[i:i + len(seeds)]))
            i += len(seeds)
        return out

    cur = list(net.get_flat())
    res0 = eval_all(cur)
    print(f"DAGGT-{SCENARIO} iter0: " + json.dumps({k: round(v, 3) for k, v in res0.items()}), flush=True)
    best_spec = res0[SCENARIO]
    best_flat = list(cur)
    json.dump(best_flat, open(OUT + "_best.json", "w"))
    X, Y = [], []
    for it in range(1, ITERS + 1):
        tc = time.time()
        args = [(list(cur), 20000 + it * 100 + j) for j in range(EPS_PER_ITER)]
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
        improved = res[SCENARIO] > best_spec
        if improved:
            best_spec = res[SCENARIO]
            best_flat = list(cur)
            json.dump(best_flat, open(OUT + "_best.json", "w"))
        print(f"DAGGT-{SCENARIO} iter{it}: n={len(X)} wall={time.time()-tc:.0f}s "
              + json.dumps({k: round(v, 3) for k, v in res.items()})
              + f" best_{SCENARIO}={best_spec:.3f}", flush=True)
        post_series(f"gt_success_{SCENARIO}", res[SCENARIO], f"daggt-{SCENARIO} i{it}")
        post_status({
            "training": {"status": "running", "name": f"daggt_{SCENARIO}", "iter": it, "iters": ITERS,
                         "note": f"GT specialist {SCENARIO} {res[SCENARIO]:.2f} (champ {res0[SCENARIO]:.2f}, teacher ref hover .875 land 1.0)"},
            "candidate": {"name": f"bc_gt_{SCENARIO}_best.json",
                          "detail": f"GT-obs {SCENARIO} specialist (DAgger from teacher pilot_act3)",
                          "note": f"i{it} GT {SCENARIO} {res[SCENARIO]:.2f}"},
        })
    post_status({"training": {"status": "complete", "name": f"daggt_{SCENARIO}", "iter": ITERS, "iters": ITERS,
                              "note": f"best {SCENARIO} {best_spec:.3f} (champion {res0[SCENARIO]:.3f})"}})
    print(f"DAGGT-{SCENARIO} DONE best={best_spec:.3f} wall={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
