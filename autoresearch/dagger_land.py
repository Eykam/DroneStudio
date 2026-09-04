"""Land-focused DAgger continuation (scenario-sampler run 2).

Run 1 (dagger_scenarios, v14-g13 + com + 11N) landed goto 1.0 / hover 0.812 /
land 0.125. Student land failures: unbraked dives (vy -5..-10), pad
near-misses, and park-above-pad timeouts - the descent phase is ~5% of
samples and dim18 (radius/extent ~0.01-0.05) is a weak phase-transition
signal. Attack: land-oversampled rollouts (60%) from the run-1 best net so
descent states accumulate; goto/hover kept in the mix to hold their gains.

Checkpoints every iteration; publishes land-focused gates to the same
per-scenario dashboard series; best (by land with floors) -> bc_flat_v2.json.
"""
import sys, json, os
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
from scene_schema import SceneDistribution
from env_sim import make_sim_factory
from policy import MLP
from diverse_bc import pilot_act3, bc_train
from scenario_sampler import sample_spec, heldout_cells
from eval_scenarios import run_cell, post_series

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/v14_g13.manifest.json"
rng = np.random.default_rng(909)
LAND_MIX = ("land", "land", "land", "hover_hold", "goto", "goto")  # 50/17/33

def train_dist(seed):
    r = np.random.default_rng(np.uint64(seed) ^ np.uint64(0xD1A9))
    gd = float(r.uniform(2.0, 25.0))
    dens = float(r.choice([0.0, 0.05, 0.1, 0.2]))
    return SceneDistribution(obstacle_density=dens, corridor_width=4.0,
        scene_extent=max(10.0, gd * 2), goal_distance=gd,
        light_direction_entropy=0.3, texture_variety=0.0, dynamics_noise=0.0)

def run_episode(net, seed):
    sc = LAND_MIX[rng.integers(0, len(LAND_MIX))]
    spec = sample_spec(seed, force_scenario=sc)
    dist = train_dist(seed)
    max_steps = 400 if sc == "goto" else 700
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    traj = []
    for _ in range(max_steps):
        a = pilot_act3(obs, 1.75, dist.scene_extent)
        traj.append((obs.copy(), a.copy()))
        obs, r, done = env.step(net.act(obs))
        if done:
            break
    env.close()
    return traj

net = MLP(19, 4, seed=0)
net.set_flat(np.array(json.load(open("/workspace/bc_flat_v2.json")), dtype=np.float64))
cells = heldout_cells()

def eval_all(net):
    actor = lambda obs, ext: net.act(obs)
    return {sc: run_cell(actor, sc, seeds)[0] for sc, seeds in cells.items()}

X, Y = [], []
best_flat = list(net.get_flat())
res0 = eval_all(net)
print("DAGGERLAND iter0 (run-1 best): " + json.dumps({k: round(v, 3) for k, v in res0.items()}), flush=True)
best_land = res0["land"]
for it in range(1, 9):
    for j in range(20):
        for o, a in run_episode(net, 5000 + it * 100 + j):
            X.append(o); Y.append(a)
    if len(X) > 24000:
        X, Y = X[-24000:], Y[-24000:]
    net = bc_train(np.array(X), np.clip(np.array(Y), -0.95, 0.95),
                   iters=1500, obs_dim=19, init_flat=net.get_flat())
    json.dump(list(net.get_flat()), open(f"/workspace/bc_flat_v2_land{it}.json", "w"))
    res = eval_all(net)
    print(f"DAGGERLAND_ITER {it}: n={len(X)} " + " ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
    for sc, v in res.items():
        post_series("success_" + sc, v, label=f"dagger-land i{it}")
    # accept: land improves AND goto/hover hold their run-1 levels
    if res["land"] > best_land and res["goto"] >= 0.9 and res["hover_hold"] >= 0.6:
        best_land = res["land"]
        best_flat = list(net.get_flat())
        json.dump(best_flat, open("/workspace/bc_flat_v2.json", "w"))
        print(f"DAGGERLAND_SAVE iter={it} land={res['land']:.3f}", flush=True)

post_series("policy_v2", float(np.mean([eval_all(net)[k] for k in ("goto", "hover_hold", "land")])), label="dagger-land final mean")
print(f"DAGGERLAND_DONE best_land={best_land:.3f}", flush=True)
