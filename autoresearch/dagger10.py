"""DAgger rescue round targeted at the 10m stage (ladder stuck at 0.083).
Student rolls out on a 10m-biased distribution, teacher labels visited
states, retrain. Seeds from the iter-4 DAgger best. Only accepts a policy
that improves 10m success without regressing 2m.
"""
import sys, json
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from scene_schema import SceneDistribution
from env_sim import make_sim_factory
from policy import MLP
from diverse_bc import pilot_act2, bc_train, eval_net

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"
rng = np.random.default_rng(23)

def dist10():
    gd = float(rng.choice([10, 10, 10, 15, 5]))
    dens = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
    return SceneDistribution(obstacle_density=dens, corridor_width=4.0,
        scene_extent=max(10.0, gd * 2), goal_distance=gd, light_direction_entropy=0.3,
        texture_variety=0.0, dynamics_noise=0.0)

def student_rollouts(net, n_eps=16):
    X, Y = [], []
    for _ in range(n_eps):
        ddist = dist10()
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

# NOTE: demo_diverse.npz long-range labels were produced by the buggy
# (pre-ext-fix) teacher - excluded; student states labeled by the fixed
# teacher only.
X, Y = [], []

net = MLP(15, 4, seed=0)
net.set_flat(np.array(json.load(open("/workspace/bc_flat_dagger.json")), dtype=np.float64))

best_s10 = 0.083  # curriculum's measured 10m best; must beat this to save
for it in range(1, 7):
    xs, ys = student_rollouts(net, n_eps=16)
    X += xs; Y += ys
    if len(X) > 20000:
        X, Y = X[-20000:], Y[-20000:]
    net = bc_train(np.array(X), np.clip(np.array(Y), -0.95, 0.95), iters=1500)
    s2, _ = eval_net(net, 2.0, 0.0)
    s5o, _ = eval_net(net, 5.0, 0.1)
    s10, _ = eval_net(net, 10.0, 0.0)
    s10o, _ = eval_net(net, 10.0, 0.1)
    print(f"DAGGER10_ITER {it}: n={len(X)} succ 2m={s2:.3f} 5m+d0.1={s5o:.3f} 10m={s10:.3f} 10m+d0.1={s10o:.3f}", flush=True)
    if s10 > best_s10 and s2 >= 0.8:
        json.dump(net.get_flat().tolist(), open("/workspace/bc_flat_dagger10.json", "w"))
        best_s10 = s10
        print(f"  new best 10m={s10:.3f} -> /workspace/bc_flat_dagger10.json", flush=True)
print("DAGGER10_DONE", json.dumps({"best_s10": best_s10}), flush=True)
