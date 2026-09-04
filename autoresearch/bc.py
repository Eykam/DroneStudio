import sys, json, time
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from scripted_pilot import pilot_act
from scene_schema import SceneDistribution
from env_sim import make_sim_factory
from policy import MLP

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"

def collect(gd, n_eps, seed0):
    dist = SceneDistribution(obstacle_density=0.0, corridor_width=10.0,
        scene_extent=max(10.0, gd*2), goal_distance=gd, light_direction_entropy=0.0,
        texture_variety=0.0, dynamics_noise=0.0)
    factory = make_sim_factory(dist, max_steps=400, dynamics=MANIFEST)
    X, Y, kept = [], [], 0
    i = 0
    while kept < n_eps and i < n_eps * 6:
        env = factory(seed0 + i); i += 1
        obs = env.reset()
        traj = []
        for _ in range(400):
            a = pilot_act(obs)
            traj.append((obs.copy(), a.copy()))
            obs, r, done = env.step(a)
            if done: break
        if env.succeeded:
            kept += 1
            for o, a in traj:
                X.append(o); Y.append(a)
        env.close()
    return X, Y, kept

X, Y = [], []
for gd, n in ((2.0, 8), (5.0, 8), (10.0, 8)):
    x, y, kept = collect(gd, n, 40000 + int(gd*100))
    X += x; Y += y
    print(f"collected goal {gd}m: {kept} success eps, {len(y)} samples", flush=True)
X = np.array(X); Y = np.clip(np.array(Y), -0.95, 0.95)
print("total samples", len(X), flush=True)

# BC: regress tanh-output MLP onto pilot actions (Adam, full-batch)
net = MLP(15, 4, seed=0)
W = [net.W1, net.b1, net.W2, net.b2, net.W3, net.b3]
m = [np.zeros_like(w) for w in W]; v = [np.zeros_like(w) for w in W]
lr, b1, b2, eps = 3e-3, 0.9, 0.999, 1e-8
for t in range(1, 1501):
    h1 = np.tanh(X @ W[0] + W[1]); h2 = np.tanh(h1 @ W[2] + W[3]); out = np.tanh(h2 @ W[4] + W[5])
    err = out - Y
    loss = float(np.mean(err**2))
    d = 2 * err / err.size * (1 - out**2)
    gW3 = h2.T @ d; gb3 = d.sum(0)
    d2 = (d @ W[4].T) * (1 - h2**2)
    gW2 = h1.T @ d2; gb2 = d2.sum(0)
    d1 = (d2 @ W[2].T) * (1 - h1**2)
    gW1 = X.T @ d1; gb1 = d1.sum(0)
    for j, g in enumerate((gW1, gb1, gW2, gb2, gW3, gb3)):
        m[j] = b1*m[j] + (1-b1)*g; v[j] = b2*v[j] + (1-b2)*g*g
        W[j] -= lr * (m[j]/(1-b1**t)) / (np.sqrt(v[j]/(1-b2**t)) + eps)
    if t % 500 == 0: print(f"bc iter {t}: loss {loss:.5f}", flush=True)

net.W1, net.b1, net.W2, net.b2, net.W3, net.b3 = W
np.savez("/workspace/bc_policy.npz", **{f"p{i}": w for i, w in enumerate(W)})
json.dump(net.get_flat().tolist(), open("/workspace/bc_flat.json", "w"))

# zero-shot eval
for gd in (2.0, 5.0, 10.0):
    dist = SceneDistribution(obstacle_density=0.0, corridor_width=10.0,
        scene_extent=max(10.0, gd*2), goal_distance=gd, light_direction_entropy=0.0,
        texture_variety=0.0, dynamics_noise=0.0)
    factory = make_sim_factory(dist, max_steps=400, dynamics=MANIFEST)
    succ, lens = [], []
    for i in range(24):
        env = factory(50000 + i)
        obs = env.reset()
        for _ in range(400):
            obs, r, done = env.step(net.act(obs))
            if done: break
        succ.append(bool(env.succeeded)); lens.append(env.steps)
        env.close()
    print("BC_RESULT", json.dumps({"goal_m": gd, "success": float(np.mean(succ)), "steps": round(float(np.mean(lens)),1)}), flush=True)
