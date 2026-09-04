"""Targeted DAgger rescue at 25m (ladder stuck at 0.125 published / 0.167 measured).
Student rollouts at 25m-heavy mix, labeled by the FIXED ext-aware teacher,
retrain. Seeds from the live bc_flat.json (dagger10 best). Only accepts a
policy that improves 25m success without regressing 10m/2m. Streams to the
dashboard like the curriculum runner.
"""
import sys, json, os, urllib.request
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from scene_schema import SceneDistribution
from env_sim import make_sim_factory
from policy import MLP
from diverse_bc import pilot_act2, bc_train, eval_net

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"
DASHBOARD = os.environ.get("DASHBOARD_URL", "").rstrip("/")
TOKEN = os.environ.get("INGEST_TOKEN", "")
rng = np.random.default_rng(25)

def post_series(series, y, label=None):
    if not (DASHBOARD and TOKEN):
        return
    try:
        pt = {"y": float(y)}
        if label:
            pt["label"] = str(label)
        req = urllib.request.Request(
            DASHBOARD + "/api/series",
            data=json.dumps({"series": series, "point": pt}).encode(),
            headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass

def dist25():
    gd = float(rng.choice([25, 25, 25, 25, 15, 10]))
    dens = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
    return SceneDistribution(obstacle_density=dens, corridor_width=4.0,
        scene_extent=max(10.0, gd * 2), goal_distance=gd, light_direction_entropy=0.3,
        texture_variety=0.0, dynamics_noise=0.0)

def student_rollouts(net, n_eps=16):
    X, Y = [], []
    for _ in range(n_eps):
        ddist = dist25()
        env = make_sim_factory(ddist, max_steps=400, dynamics=MANIFEST)(int(rng.integers(1, 1 << 30)))
        obs = env.reset()
        for _ in range(400):
            X.append(obs.copy())
            Y.append(pilot_act2(obs, 1.75, ddist.scene_extent).copy())
            obs, r, done = env.step(net.act(obs))
            if done:
                break
        env.close()
    return X, Y

X, Y = [], []
net = MLP(15, 4, seed=0)
net.set_flat(np.array(json.load(open("/workspace/bc_flat.json")), dtype=np.float64))

best_s25 = 0.167  # measured 25m d0.1 of the live policy; must beat this to save
best_flat = list(net.get_flat())
for it in range(1, 11):
    xs, ys = student_rollouts(net, n_eps=16)
    X += xs; Y += ys
    if len(X) > 24000:
        X, Y = X[-24000:], Y[-24000:]
    net = bc_train(np.array(X), np.clip(np.array(Y), -0.95, 0.95), iters=1500)
    json.dump(list(net.get_flat()), open(f"/workspace/bc_flat_dagger25_iter{it}.json", "w"))  # checkpoint every iter
    s2, _ = eval_net(net, 2.0, 0.0, n=16)
    s10, _ = eval_net(net, 10.0, 0.1, n=16)
    s15, _ = eval_net(net, 15.0, 0.1, n=16)
    s25, _ = eval_net(net, 25.0, 0.1, n=16)
    print(f"DAGGER25_ITER {it}: n={len(X)} succ 2m={s2:.3f} 10m+d0.1={s10:.3f} 15m={s15:.3f} 25m={s25:.3f}", flush=True)
    post_series("dagger25", s25, label=f"i{it} 10m={s10:.2f}")
    if s25 > best_s25 and s10 >= 0.75 and s2 >= 0.8:
        best_s25 = s25
        best_flat = list(net.get_flat())
        json.dump(best_flat, open("/workspace/bc_flat_dagger25.json", "w"))
        print(f"DAGGER25_SAVE iter={it} s25={s25:.3f}", flush=True)

if best_s25 > 0.167:
    json.dump(best_flat, open("/workspace/bc_flat.json", "w"))
    post_series("policy", best_s25, label="dagger25 25m")
    print(f"DAGGER25_DONE published best_s25={best_s25:.3f}", flush=True)
else:
    print(f"DAGGER25_DONE no improvement (best_s25={best_s25:.3f})", flush=True)
