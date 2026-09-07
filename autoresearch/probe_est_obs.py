"""Linear-probe informativeness test: can the policy's 19-dim obs vector
decode TRUE goal-relative position / velocity during hover vs goto?

Arms {est-obs, GT-passthrough} x {goto, hover_hold}, champion policy.
Per-step (obs, target) pairs; targets are GT state expressed in the SAME
per-step frame the obs vector was built with (filter yaw for est arm, GT yaw
for GT arm). Ridge probe, episode-disjoint 8/4 train/test split.
R2 near the GT arm => obs informative (policy-learning problem);
R2 collapse in hover est arm => structural insufficiency (obs redesign).
"""
import os, json, sys
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from scenario_sampler import sample_spec, heldout_cells
from eval_scenarios import cell_dist
from ppo_v2 import MANIFEST
from eval_estimated import EstEnv, load_policy, yaw_frame
from sensors.ekf import R_of

POLICY = "/workspace/bc_ppo_v2_best.json"
N_EPS = 12

def collect(scenario, obs_mode, seeds):
    net, od = load_policy(POLICY)
    X, Yp, Yv, lens = [], [], [], []
    for seed in seeds:
        dist = cell_dist(seed)
        spec = sample_spec(seed, force_scenario=scenario)
        env = EstEnv(dist, seed=seed, max_steps=400 if scenario == "goto" else 700,
                     dynamics=MANIFEST, scenario_spec=spec, estimated=True, obs_v2=True,
                     vo_aided=True, est_seed=seed + 777, passthrough=(obs_mode == "gt"))
        obs = env.reset()
        done = False
        n = 0
        while not done:
            a = net.act(obs)
            obs, r, done = env.step(a)
            info = env.last_info
            p_gt = np.array(info["pos"], dtype=float)
            v_gt = np.array(info["vel"], dtype=float)
            q_ref = np.array(info["quat"], dtype=float) if obs_mode == "gt" else env.kf.q
            fwd = R_of(q_ref) @ np.array([1.0, 0.0, 0.0])
            yaw = np.arctan2(-fwd[2], fwd[0])
            ext = max(env.dist.scene_extent, 1.0)
            X.append(obs.copy())
            Yp.append(yaw_frame(env.goal - p_gt, yaw) / ext)
            Yv.append(yaw_frame(v_gt, yaw) / 10.0)
            n += 1
        lens.append(n)
        env.close()
    return np.array(X), np.array(Yp), np.array(Yv), lens

def probe(X, Y, ep_lens, lam=1e-3):
    bounds = np.concatenate([[0], np.cumsum(ep_lens)])
    tr = np.zeros(len(X), bool)
    for i in range(min(8, len(ep_lens) - 1)):
        tr[bounds[i]:bounds[i + 1]] = True
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Z = (X - mu) / sd
    Zb = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    W = np.linalg.solve(Zb[tr].T @ Zb[tr] + lam * np.eye(Zb.shape[1]), Zb[tr].T @ Y[tr])
    pred = Zb[~tr] @ W
    yt = Y[~tr]
    ss_res = ((pred - yt) ** 2).sum(0)
    ss_tot = ((yt - yt.mean(0)) ** 2).sum(0) + 1e-12
    rmse = np.sqrt(((pred - yt) ** 2).mean(0))
    return (1 - ss_res / ss_tot).round(3).tolist(), rmse.round(4).tolist()

cells = heldout_cells()
res = {}
for scenario in ("goto", "hover_hold"):
    for obs_mode in ("est", "gt"):
        X, Yp, Yv, lens = collect(scenario, obs_mode, cells[scenario][:N_EPS])
        pos_r2, pos_rmse = probe(X, Yp, lens)
        vel_r2, vel_rmse = probe(X, Yv, lens)
        res[f"{scenario}_{obs_mode}"] = {
            "n": len(X), "ep_lens": lens,
            "pos_r2": pos_r2, "pos_rmse_norm": pos_rmse,
            "vel_r2": vel_r2, "vel_rmse_norm": vel_rmse}
        print(f"{scenario}/{obs_mode}: n={len(X)} pos_r2={pos_r2} vel_r2={vel_r2}", flush=True)
json.dump(res, open("/workspace/vision_model/probe_est_obs.json", "w"), indent=1)
print("PROBE DONE", flush=True)
