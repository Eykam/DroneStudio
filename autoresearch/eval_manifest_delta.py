import os, json, sys
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from scenario_sampler import sample_spec, heldout_cells
from eval_scenarios import cell_dist
from parallel_rollout import parallel_episodes
from policy import MLP
from ppo_v2 import MANIFEST
from eval_estimated import EstEnv, load_policy
from env_sim import make_sim_factory

V61 = "/workspace/DroneStudio/autoresearch/fixtures/v61_g60a.manifest.json"
POLICY = "/workspace/bc_ppo_v2_best.json"

def eval_gt(manifest, scenario, seed):
    net, od = load_policy(POLICY)
    spec = sample_spec(seed, force_scenario=scenario)
    dist = cell_dist(seed)
    env = make_sim_factory(dist, max_steps=400 if scenario == "goto" else 700,
                           dynamics=manifest, scenario_spec=spec)(seed)
    obs = env.reset(); done = False
    while not done:
        obs, r, done = env.step(net.act(obs))
    ok = bool(env.succeeded); env.close(); return ok

def eval_est(manifest, scenario, seed):
    flat = np.array(json.load(open(POLICY)), dtype=np.float64)
    n19, od = load_policy(POLICY)
    net = MLP(25, 4, seed=0)
    net.W1[:19, :] = n19.W1; n25 = net
    n25.W1[19:, :] = 0.0
    n25.b1, n25.W2, n25.b2, n25.W3, n25.b3 = n19.b1, n19.W2, n19.b2, n19.W3, n19.b3
    spec = sample_spec(seed, force_scenario=scenario)
    dist = cell_dist(seed)
    env = EstEnv(dist, seed=seed, max_steps=400 if scenario == "goto" else 700,
                 dynamics=manifest, scenario_spec=spec, estimated=True, obs_v2=True,
                 obs_v3=True, vo_aided=True, est_seed=seed + 777)
    obs = env.reset(); done = False
    while not done:
        obs, r, done = env.step(n25.act(obs))
    ok = bool(env.succeeded); env.close(); return ok

cells = heldout_cells()
out = {}
for tag, manifest in (("v14", MANIFEST), ("v61", V61)):
    for arm, fn in (("gt", eval_gt), ("est_v3", eval_est)):
        for sc in ("goto", "hover_hold", "land"):
            res = parallel_episodes(fn, [(manifest, sc, s) for s in cells[sc]])
            out[f"{tag}/{arm}/{sc}"] = round(float(np.mean(res)), 3)
            print(f"{tag}/{arm}/{sc}: {out[f'{tag}/{arm}/{sc}']}", flush=True)
json.dump(out, open("/workspace/vision_model/manifest_delta_eval.json", "w"), indent=1)
