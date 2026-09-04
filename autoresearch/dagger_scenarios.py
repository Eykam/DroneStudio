"""Scenario-sampler DAgger #1 - the v14-g13 fixture switch.

First training run on the latest adopted CAD manifest (v14-g13), obs v2
(19-dim, yaw-relative), scenario sampler (goto/hover_hold/land). Warm-start:
net2wider of the published v1 dagger25 net (behavior preserved at init; the
4 new input columns learn from scratch).

Flow: teacher BC seed set -> DAgger loop (student rollouts labeled by
pilot_act3) -> per-iteration held-out eval on fixed seed blocks -> checkpoint
every iter -> best mean success to bc_flat_v2.json + per-scenario dashboard
series. bc_flat.json (the live 15-dim /watch policy) is NOT touched.
"""
import sys, json, os
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
from scene_schema import SceneDistribution
from env_sim import make_sim_factory
from policy import MLP, widen_input
from diverse_bc import pilot_act3, bc_train
from scenario_sampler import sample_spec, heldout_cells
from eval_scenarios import run_cell, post_series

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/v14_g13.manifest.json"
rng = np.random.default_rng(707)

def train_dist(seed):
    r = np.random.default_rng(np.uint64(seed) ^ np.uint64(0xD1A9))
    gd = float(r.uniform(2.0, 25.0))
    dens = float(r.choice([0.0, 0.05, 0.1, 0.2]))
    return SceneDistribution(obstacle_density=dens, corridor_width=4.0,
        scene_extent=max(10.0, gd * 2), goal_distance=gd,
        light_direction_entropy=0.3, texture_variety=0.0, dynamics_noise=0.0)

def run_episode(net, seed):
    spec = sample_spec(seed)
    dist = train_dist(seed)
    max_steps = 400 if spec["scenario"] == "goto" else 700
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    traj = []
    for _ in range(max_steps):
        a = pilot_act3(obs, 1.75, dist.scene_extent)
        traj.append((obs.copy(), a.copy()))
        obs, r, done = env.step(a if net is None else net.act(obs))
        if done:
            break
    env.close()
    return traj

if __name__ != "__main__":
    raise SystemExit("dagger_scenarios is a script, not a module")

X, Y = [], []
for i in range(48):
    for o, a in run_episode(None, 1000 + i):
        X.append(o); Y.append(a)
    if (i + 1) % 12 == 0:
        print(f"BC collect {i+1}/48 samples={len(X)}", flush=True)

net15 = MLP(15, 4, seed=0)
net15.set_flat(np.array(json.load(open("/workspace/bc_flat.json")), dtype=np.float64))
net = widen_input(net15, 19)
net = bc_train(np.array(X), np.clip(np.array(Y), -0.95, 0.95),
               iters=2000, obs_dim=19, init_flat=net.get_flat())
json.dump(list(net.get_flat()), open("/workspace/bc_flat_v2_iter0.json", "w"))

cells = heldout_cells()
def eval_all(net):
    actor = lambda obs, ext: net.act(obs)
    return {sc: run_cell(actor, sc, seeds)[0] for sc, seeds in cells.items()}

base = eval_all(net)
print("DAGGERSCEN iter0 (warm BC): " + json.dumps({k: round(v, 3) for k, v in base.items()}), flush=True)
for sc, v in base.items():
    post_series("success_" + sc, v, label="dagger-scen i0 (warm BC)")
best_mean = float(np.mean(list(base.values())))
best_flat = list(net.get_flat())

for it in range(1, 11):
    for j in range(16):
        for o, a in run_episode(net, 2000 + it * 100 + j):
            X.append(o); Y.append(a)
    if len(X) > 24000:
        X, Y = X[-24000:], Y[-24000:]
    net = bc_train(np.array(X), np.clip(np.array(Y), -0.95, 0.95),
                   iters=1500, obs_dim=19, init_flat=net.get_flat())
    json.dump(list(net.get_flat()), open(f"/workspace/bc_flat_v2_iter{it}.json", "w"))
    res = eval_all(net)
    mean = float(np.mean(list(res.values())))
    print(f"DAGGERSCEN_ITER {it}: n={len(X)} " + " ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
    post_series("dagger_scen", mean, label=f"i{it} " + " ".join(f"{k}={v:.2f}" for k, v in res.items()))
    for sc, v in res.items():
        post_series("success_" + sc, v, label=f"dagger-scen i{it}")
    if mean > best_mean:
        best_mean = mean
        best_flat = list(net.get_flat())
        print(f"DAGGERSCEN_SAVE iter={it} mean={mean:.3f}", flush=True)

json.dump(best_flat, open("/workspace/bc_flat_v2.json", "w"))
post_series("policy_v2", best_mean, label="dagger-scen best mean")
print(f"DAGGERSCEN_DONE best_mean={best_mean:.3f} -> bc_flat_v2.json", flush=True)
