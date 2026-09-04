"""Diverse demo collection -> BC -> publish if better on reference evals.

Teacher: scripted_pilot.pilot_act with per-episode speed jitter + potential-
field obstacle avoidance (nearest-obstacle vector from obs[12:15], /20 scale,
points at obstacle CENTER; radii ~0.5-2.5m so repulsion starts at 3.5m).

Diversity: goals 2/5/10/15m, obstacle densities 0/0.05/0.1/0.2, fresh scene
seeds (azimuth/altitude/goal geometry vary per episode), speed profiles
(vmax 1.0-2.5 m/s). Keep successful trajectories only.
"""
import sys, json, time
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from scene_schema import SceneDistribution
from env_sim import make_sim_factory
from policy import MLP
import scripted_pilot

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"

def pilot_act2(obs, vmax, ext=10.0):
    rel = obs[0:3] * ext           # world rel goal, m (obs = rel / scene_extent)
    vel = obs[3:6] * 10.0          # world vel, m/s (obs = v / 10)
    gb = obs[6:9]
    v_des = np.clip(0.5 * rel, -vmax, vmax)
    # damp against HALF the true velocity: 0.1 (obs-raw) orbits the goal,
    # 1.0 over-brakes; 0.5 swept to 100% at 2/5/10m, >=0.87 at 15/25m.
    a_des = np.clip(1.2 * (v_des - 0.5 * vel), -2.0, 2.0)
    # obstacle repulsion (potential field on nearest-obstacle center)
    dvec = obs[12:15] * ext
    d = float(np.linalg.norm(dvec))
    if 1e-3 < d < 3.5:
        a_des = a_des - 3.0 * (3.5 - d) / 3.5 * (dvec / d)
    a_des = np.clip(a_des, -3.0, 3.0)
    gx_des = np.clip(a_des[0] / 9.81, -0.30, 0.30)
    gz_des = np.clip(a_des[2] / 9.81, -0.30, 0.30)
    rates = obs[9:12]
    kp, kd = 0.4, 1.0
    act0 = kp * (gz_des - gb[2]) - kd * rates[0]
    act2 = -kp * (gx_des - gb[0]) - kd * rates[2]
    vy_des = np.clip(0.8 * rel[1], -1.5, 1.5)
    thr = -0.6 + 0.15 * (vy_des - vel[1])
    return np.clip(np.array([act0, 0.0, act2, thr]), -1, 1)

def pilot_act3(obs, vmax, ext=10.0):
    """19-dim (obs v2) scenario-aware teacher.

    obs v2 vectors are yaw-frame, where this controller's implicit yaw=0
    world<->body mapping (same math as pilot_act2) is exact at any heading.
    Scenario from one-hot dims 15:18, success radius from dim 18 (x extent).
    """
    rel = obs[0:3] * ext           # yaw-frame rel goal, m
    vel = obs[3:6] * 10.0          # yaw-frame vel, m/s (y component = world)
    gb = obs[6:9]
    scenario = ("goto", "hover_hold", "land")[int(np.argmax(obs[15:18]))]
    radius = max(0.05, float(obs[18]) * ext)

    dist_xz = float(np.hypot(rel[0], rel[2]))
    alt = -float(rel[1])           # land pads sit at ground level
    land_descend = (scenario == "land" and alt <= 1.4
                    and dist_xz <= max(1.0, 2.0 * radius))

    if scenario == "land" and not land_descend:
        rel = rel + np.array([0.0, 1.2, 0.0])  # phase 1: above the pad
        vmax = min(vmax, 1.2)
    v_des = np.clip(0.5 * rel, -vmax, vmax)
    if scenario == "hover_hold" and np.linalg.norm(rel) < 2.0 * radius:
        v_des = np.clip(0.4 * rel, -0.25, 0.25)  # brake and hold
    if land_descend:
        v_des = np.array([np.clip(0.6 * rel[0], -0.3, 0.3), 0.0,
                          np.clip(0.6 * rel[2], -0.3, 0.3)])

    # damping: 0.5 far (fast legs, swept for goto), 1.0 near (precision:
    # half-damping limit-cycles around the target and never enters a tight
    # radius, let alone holds one)
    damp = 0.5 if float(np.linalg.norm(rel)) > 3.0 else 1.0
    a_des = np.clip(1.2 * (v_des - damp * vel), -2.0, 2.0)
    dvec = obs[12:15] * ext
    d = float(np.linalg.norm(dvec))
    if 1e-3 < d < 3.5 and not land_descend:  # pad corridor stays clear
        a_des = a_des - 3.0 * (3.5 - d) / 3.5 * (dvec / d)
    a_des = np.clip(a_des, -3.0, 3.0)
    gx_des = np.clip(a_des[0] / 9.81, -0.30, 0.30)
    gz_des = np.clip(a_des[2] / 9.81, -0.30, 0.30)
    rates = obs[9:12]
    kp, kd = 0.4, 1.0
    act0 = kp * (gz_des - gb[2]) - kd * rates[0]
    act2 = -kp * (gx_des - gb[0]) - kd * rates[2]
    if land_descend:
        vy_des = -0.4
    elif scenario == "hover_hold":
        vy_des = np.clip(0.8 * rel[1], -0.8, 0.8)
    else:
        vy_des = np.clip(0.8 * rel[1], -1.5, 1.5)
    # hover throttle is ~-0.778 under the 11N/motor fixture (validated by
    # the land touchdown tests); the old -0.6 baseline + 0.15 gain settles
    # at vy_des + 1.19 m/s and never descends to a pad
    thr = -0.778 + 0.3 * (vy_des - vel[1])
    return np.clip(np.array([act0, 0.0, act2, thr]), -1, 1)

def collect(target_eps=48, max_attempts=400):
    rng = np.random.default_rng(7)
    X, Y, kept, attempts = [], [], 0, 0
    stats = {"succ_by_density": {}}
    while kept < target_eps and attempts < max_attempts:
        gd = float(rng.choice([2, 5, 10, 15]))
        dens = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
        vmax = 1.5
        seed = int(rng.integers(1, 1 << 30))
        dist = SceneDistribution(obstacle_density=dens, corridor_width=4.0,
            scene_extent=max(10.0, gd * 2), goal_distance=gd, light_direction_entropy=0.3,
            texture_variety=0.0, dynamics_noise=0.0)
        env = make_sim_factory(dist, max_steps=400, dynamics=MANIFEST)(seed)
        attempts += 1
        obs = env.reset()
        traj = []
        for _ in range(400):
            a = pilot_act2(obs, vmax, max(10.0, gd * 2))
            traj.append((obs.copy(), a.copy()))
            obs, r, done = env.step(a)
            if done:
                break
        ok = bool(env.succeeded)
        key = str(dens)
        s, n = stats["succ_by_density"].get(key, (0, 0))
        stats["succ_by_density"][key] = (s + int(ok), n + 1)
        if ok:
            kept += 1
            for o, a in traj:
                X.append(o); Y.append(a)
        env.close()
        if attempts % 40 == 0:
            print(f"attempts {attempts}: kept {kept}, by density {stats['succ_by_density']}", flush=True)
    return np.array(X), np.clip(np.array(Y), -0.95, 0.95), kept, attempts, stats

def bc_train(X, Y, iters=2000, obs_dim=15, init_flat=None):
    net = MLP(obs_dim, 4, seed=0)
    if init_flat is not None:
        net.set_flat(np.array(init_flat, dtype=np.float64))
    W = [net.W1, net.b1, net.W2, net.b2, net.W3, net.b3]
    m = [np.zeros_like(w) for w in W]; v = [np.zeros_like(w) for w in W]
    lr, b1, b2, eps = 3e-3, 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        h1 = np.tanh(X @ W[0] + W[1]); h2 = np.tanh(h1 @ W[2] + W[3]); out = np.tanh(h2 @ W[4] + W[5])
        err = out - Y
        d = 2 * err / err.size * (1 - out ** 2)
        gW3 = h2.T @ d; gb3 = d.sum(0)
        d2 = (d @ W[4].T) * (1 - h2 ** 2)
        gW2 = h1.T @ d2; gb2 = d2.sum(0)
        d1 = (d2 @ W[2].T) * (1 - h1 ** 2)
        gW1 = X.T @ d1; gb1 = d1.sum(0)
        for j, g in enumerate((gW1, gb1, gW2, gb2, gW3, gb3)):
            m[j] = b1 * m[j] + (1 - b1) * g; v[j] = b2 * v[j] + (1 - b2) * g * g
            W[j] -= lr * (m[j] / (1 - b1 ** t)) / (np.sqrt(v[j] / (1 - b2 ** t)) + eps)
        if t % 1000 == 0:
            print(f"bc iter {t}: loss {float(np.mean(err**2)):.5f}", flush=True)
    net.W1, net.b1, net.W2, net.b2, net.W3, net.b3 = W
    return net

def eval_net(net, gd, dens, n=24, seed0=77000):
    dist = SceneDistribution(obstacle_density=dens, corridor_width=4.0,
        scene_extent=max(10.0, gd * 2), goal_distance=gd, light_direction_entropy=0.3,
        texture_variety=0.0, dynamics_noise=0.0)
    factory = make_sim_factory(dist, max_steps=400, dynamics=MANIFEST)
    succ, lens = [], []
    for i in range(n):
        env = factory(seed0 + i)
        obs = env.reset()
        for _ in range(400):
            obs, r, done = env.step(net.act(obs))
            if done:
                break
        succ.append(bool(env.succeeded)); lens.append(env.steps)
        env.close()
    return float(np.mean(succ)), float(np.mean(lens))

if __name__ == "__main__":
    X, Y, kept, attempts, stats = collect()
    print("COLLECT_DONE", json.dumps({"samples": len(X), "eps": kept, "attempts": attempts, "by_density": stats["succ_by_density"]}), flush=True)
    np.savez("/workspace/demo_diverse.npz", X=X, Y=Y)
    net = bc_train(X, Y)
    results = {}
    for gd, dens in ((2, 0.0), (5, 0.0), (10, 0.0), (5, 0.1), (10, 0.1), (15, 0.0)):
        s, l = eval_net(net, float(gd), float(dens))
        results[f"{gd}m_d{dens}"] = {"success": s, "steps": round(l, 1)}
        print("BC_DIVERSE_RESULT", gd, dens, json.dumps(results[f"{gd}m_d{dens}"]), flush=True)
    json.dump(net.get_flat().tolist(), open("/workspace/bc_flat_diverse2.json", "w"))
    json.dump(results, open("/workspace/bc_diverse_results.json", "w"), indent=2)
