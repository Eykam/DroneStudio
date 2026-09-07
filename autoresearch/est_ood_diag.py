"""Diagnostic (c): are est hover/land eval cells OOD for the aggregated
DAgger v4 dataset? (parent direction 2026-09-07)

A) spec parity: training seed blocks vs held-out eval blocks (no sim).
B) student (bc_est_dag_v4_best) visitation on TRAINING hover/land cells.
C) teacher (pilot_act3) visitation on EVAL hover/land cells = states needed
   to succeed.
D) coverage: NN distance from teacher eval states (esp. success region) to
   the student training visitation cloud.
"""
import os, json
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
from policy import MLP
from diverse_bc import pilot_act3
from scenario_sampler import sample_spec, heldout_cells, HELDOUT_BASE
from eval_scenarios import cell_dist
from parallel_rollout import parallel_episodes
from ppo_v2 import MANIFEST
from eval_estimated import EstEnv

OBS_DIM = 25
V4BEST = "/workspace/bc_est_dag_v4_best.json"

def make_env(dist, spec, max_steps, seed, passthrough=False):
    return EstEnv(dist, seed=seed, max_steps=max_steps, dynamics=MANIFEST,
                  scenario_spec=spec, estimated=True, obs_v2=True, obs_v3=True,
                  vo_aided=True, est_seed=int(seed) + 777, passthrough=passthrough)

def spec_parity():
    rows = {}
    for sc, tr_range, ev_base in (("hover_hold", range(9000, 9400), 88000),
                                  ("land", range(9000, 9400), 99000)):
        def feats(seeds):
            r, h, gd, de = [], [], [], []
            for s in seeds:
                sp = sample_spec(s, force_scenario=sc)
                d = cell_dist(s)
                r.append(sp["success_radius"]); gd.append(d.goal_distance)
                de.append(d.obstacle_density)
                h.append(sp.get("hold_s", 0.0))
            return dict(radius=(min(r), max(r)), hold_s=(min(h), max(h)),
                        goal_dist=sorted(set(gd)), density=sorted(set(de)))
        rows[sc] = {"train": feats(tr_range), "eval": feats(range(ev_base, ev_base + 16))}
    return rows

def student_trace(flat, sc, seed):
    net = MLP(OBS_DIM, 4, seed=0); net.set_flat(np.array(flat, dtype=np.float64))
    spec = sample_spec(seed, force_scenario=sc)
    dist = cell_dist(seed)
    ms = 700
    env = make_env(dist, spec, ms, seed)
    env.reset()
    pts = []
    for _ in range(ms):
        a = net.act(env._est_obs() if False else None) if False else None
        # student acts on est obs: replicate v4 loop (net.act on env-returned obs)
        break
    env.close()
    # rerun properly: env.step returns est obs
    env = make_env(dist, spec, ms, seed)
    obs = env.reset()
    pts = []
    for _ in range(ms):
        a = net.act(obs)
        obs, r, done = env.step(a)
        p = np.array(env.last_info.get("pos", [np.nan]*3), dtype=float)
        v = np.array(env.last_info.get("vel", [np.nan]*3), dtype=float)
        g = env.goal
        pts.append(np.concatenate([p, v, g - p, [np.linalg.norm(g - p)]]))
        if done:
            break
    succ = bool(env.succeeded); coll = bool(env.collided); steps = env.steps
    rad = spec["success_radius"]
    env.close()
    return dict(sc=sc, succ=succ, coll=coll, steps=steps, radius=rad,
                pts=np.array(pts, dtype=np.float32))

def teacher_trace(sc, seed):
    spec = sample_spec(seed, force_scenario=sc)
    dist = cell_dist(seed)
    ms = 700 if sc == "goto" else 2100
    env = make_env(dist, spec, ms, seed, passthrough=True)
    obs = env.reset()
    pts = []
    for _ in range(ms):
        a = pilot_act3(env.last_gt_obs, 1.75, dist.scene_extent)
        obs, r, done = env.step(a)
        p = np.array(env.last_info.get("pos", [np.nan]*3), dtype=float)
        v = np.array(env.last_info.get("vel", [np.nan]*3), dtype=float)
        g = env.goal
        pts.append(np.concatenate([p, v, g - p, [np.linalg.norm(g - p)]]))
        if done:
            break
    succ = bool(env.succeeded); steps = env.steps
    rad = spec["success_radius"]
    env.close()
    return dict(sc=sc, succ=succ, steps=steps, radius=rad,
                pts=np.array(pts, dtype=np.float32))

def main():
    print("=== A) spec parity ===", flush=True)
    print(json.dumps(spec_parity(), indent=1, default=str), flush=True)

    flat = json.load(open(V4BEST))
    cells = heldout_cells()
    # B) student on TRAINING seeds (same seed style as v4 collection)
    tr_args = []
    for j in range(12):
        tr_args.append((flat, "hover_hold", 9000 + 500 + j))
        tr_args.append((flat, "land", 9000 + 600 + j))
    stu = parallel_episodes(student_trace, tr_args)
    print("=== B) student visitation on training cells ===", flush=True)
    S = []
    for r in stu:
        d = r["pts"][:, 9] if len(r["pts"]) else np.array([np.nan])
        within = float(np.mean(d < r["radius"])) if len(r["pts"]) else 0.0
        S.append(dict(sc=r["sc"], succ=r["succ"], coll=r["coll"], steps=r["steps"],
                      min_dist=round(float(np.nanmin(d)), 2),
                      final_dist=round(float(d[-1]), 2),
                      frac_steps_within_radius=round(within, 4)))
        print(json.dumps(S[-1]), flush=True)

    # C) teacher on EVAL cells
    te_args = [("hover_hold", s) for s in cells["hover_hold"]] + \
              [("land", s) for s in cells["land"]]
    tea = parallel_episodes(teacher_trace, te_args)
    print("=== C) teacher visitation on eval cells ===", flush=True)
    for r in tea:
        d = r["pts"][:, 9]
        within = float(np.mean(d < r["radius"]))
        print(json.dumps(dict(sc=r["sc"], succ=r["succ"], steps=r["steps"],
                              min_dist=round(float(d.min()), 2),
                              frac_steps_within_radius=round(within, 4))), flush=True)

    # D) coverage: NN from teacher success-region states to student cloud
    def cloud(rows, success_only=False):
        P = []
        for r in rows:
            pts = r["pts"]
            if success_only:
                mask = pts[:, 9] < r["radius"]
                if r["sc"] == "land":
                    mask |= (np.arange(len(pts)) >= max(0, len(pts) - 100))
                pts = pts[mask]
            if len(pts):
                P.append(pts)
        return np.concatenate(P) if P else np.empty((0, 10), dtype=np.float32)
    SC = cloud(stu)                # all student-visited states (training)
    TE = cloud(tea, success_only=True)  # teacher states in the success region (eval)
    print(f"=== D) coverage: student cloud {SC.shape[0]} states, teacher success-region {TE.shape[0]} states ===", flush=True)
    if len(SC) and len(TE):
        scale = np.maximum(SC.std(axis=0), 1e-6)
        SCn, TEn = SC / scale, TE / scale
        # chunked brute-force NN
        nn = []
        B = 4096
        for i in range(0, len(TEn), B):
            t = TEn[i:i + B]
            d2 = ((t[:, None, :] - SCn[None, :, :]) ** 2).sum(-1)
            nn.append(np.sqrt(d2.min(axis=1)))
        nn = np.concatenate(nn)
        print(json.dumps(dict(nn_median=round(float(np.median(nn)), 3),
                              nn_p90=round(float(np.percentile(nn, 90)), 3),
                              nn_max=round(float(nn.max()), 3),
                              frac_nn_gt_2sig=round(float(np.mean(nn > 2.0)), 4))), flush=True)
        # per-dim range comparison (student cloud vs teacher success region)
        dims = ["px", "py", "pz", "vx", "vy", "vz", "dx", "dy", "dz", "dist"]
        for k, name in enumerate(dims):
            print(f"dim {name}: student [{SC[:,k].min():.2f},{SC[:,k].max():.2f}] "
                  f"teacher_success [{TE[:,k].min():.2f},{TE[:,k].max():.2f}]", flush=True)

if __name__ == "__main__":
    main()
