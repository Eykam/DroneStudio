import os, json, sys
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from scenario_sampler import sample_spec
from eval_scenarios import cell_dist
import eval_estimated as EE
from eval_estimated import EstEnv, R_of
from ppo_v2 import MANIFEST
from policy import MLP

net = MLP(25, 4, seed=0)
net.set_flat(np.array(json.load(open("/workspace/bc_est_dag_v6_best.json")), dtype=np.float64))

COS65 = float(np.cos(np.deg2rad(65)))
allrows = []
for si in range(16):
    seed = 88000 + si * 137
    spec = sample_spec(seed, force_scenario="hover_hold")
    dist = cell_dist(seed)
    env = EstEnv(dist, seed=seed, max_steps=2100, dynamics=MANIFEST,
                 scenario_spec=spec, estimated=True, obs_v2=True, obs_v3=True,
                 vo_aided=True, est_seed=seed + 777, zupt=True, gps=True, marker=True)
    obs = env.reset()
    prev_p = None
    for t in range(2100):
        a = net.act(obs)
        obs, r, done = env.step(a)
        q_t = np.array(env.last_info["quat"], dtype=float)
        p_t = np.array(env.last_info["pos"], dtype=float)
        m = np.array([env.goal[0], 0.0, env.goal[2]])
        v_w = m - p_t
        rng_m = float(np.linalg.norm(v_w))
        z_b = R_of(q_t).T @ v_w
        cos_ang = float(-z_b[1] / max(rng_m, 1e-9))
        valid = (rng_m < 5.0) and (cos_ang > COS65)
        # tilt of body down-axis off world down
        down_b = R_of(q_t) @ np.array([0.0, -1.0, 0.0])
        tilt = float(np.arccos(np.clip(-down_b[1], -1, 1)))
        herr = float(np.hypot(p_t[0] - env.goal[0], p_t[2] - env.goal[2]))
        hspeed = 0.0 if prev_p is None else float(np.hypot(p_t[0]-prev_p[0], p_t[2]-prev_p[2]) / 0.05)
        prev_p = p_t
        allrows.append([si, t, np.arccos(np.clip(cos_ang, -1, 1)), rng_m, float(valid), tilt, herr, hspeed])
        if done:
            break
    env.close()
    print(f"seed {si} done steps={t+1}", flush=True)

R = np.array([r[2:] for r in allrows])
offaxis, rng_m, valid, tilt, herr, hspeed = np.rad2deg(R[:,0]), R[:,1], R[:,2]>0.5, np.rad2deg(R[:,3]), R[:,4], R[:,5]
print(f"\nsteps={len(valid)} overall marker-valid fraction = {valid.mean():.3f}")
print(f"off-axis angle deg: mean {offaxis.mean():.1f} p50 {np.percentile(offaxis,50):.1f} p90 {np.percentile(offaxis,90):.1f} p99 {np.percentile(offaxis,99):.1f}")
print(f"range m: mean {rng_m.mean():.2f} p99 {np.percentile(rng_m,99):.2f}; fraction >=5m: {(rng_m>=5.0).mean():.4f}")
print(f"tilt deg: mean {tilt.mean():.2f} p90 {np.percentile(tilt,90):.2f} p99 {np.percentile(tilt,99):.2f} max {tilt.max():.2f}")
# corrections vs steady: correction = actively translating (hspeed>0.15) or tilted (>5deg)
corr = (hspeed > 0.15) | (tilt > 5.0)
steady = (hspeed < 0.10) & (tilt < 3.0)
print(f"\ncorrection steps: {corr.mean():.3f} of all; marker-valid during corrections = {valid[corr].mean():.3f} (n={corr.sum()})")
print(f"steady steps: {steady.mean():.3f} of all; marker-valid during steady = {valid[steady].mean():.3f} (n={steady.sum()})")
for lo, hi in [(0,2),(2,5),(5,10),(10,20),(20,45)]:
    m = (tilt>=lo)&(tilt<hi)
    if m.sum(): print(f"tilt [{lo:2d},{hi:2d}) deg: n={m.sum():6d} valid={valid[m].mean():.3f}")
# how close to cone edge when invalid
print(f"\ninvalid steps: {(~valid).sum()} ({(~valid).mean():.4f}); of those, range-out {(rng_m[~valid]>=5.0).mean():.3f}, angle-out {(offaxis[~valid]>=65.0).mean():.3f}")
if (~valid).sum():
    print(f"off-axis during invalid: mean {offaxis[~valid].mean():.1f}; tilt during invalid: mean {tilt[~valid].mean():.1f}")
