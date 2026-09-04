"""Held-out per-scenario eval on fixed seed blocks (never trained on):
goto 77000+, hover_hold 88000+, land 99000+ (n=16 each). Publishes one
dashboard series per scenario plus realism metrics (land touchdown
vertical speed, hover hold drift). Distance/density per seed are drawn
deterministically from the block seed so cells are stable across runs.

Usage: eval_scenarios.py <policy_flat.json | teacher>
Requires AUTORESEARCH_OBS_V2=1 for policy evals (19-dim); teacher always
runs 19-dim (pilot_act3).
"""
import sys, json, os, urllib.request
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from scene_schema import SceneDistribution
from env_sim import make_sim_factory
from scenario_sampler import sample_spec, heldout_cells

MANIFEST = os.environ.get("AUTORESEARCH_MANIFEST",
    "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json")
DASHBOARD = os.environ.get("DASHBOARD_URL", "").rstrip("/")
TOKEN = os.environ.get("INGEST_TOKEN", "")

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

def cell_dist(seed):
    rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(0xE7A1))
    gd = float(rng.choice([2, 5, 10, 15, 25]))
    dens = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
    return SceneDistribution(obstacle_density=dens, corridor_width=4.0,
        scene_extent=max(10.0, gd * 2), goal_distance=gd,
        light_direction_entropy=0.3, texture_variety=0.0, dynamics_noise=0.0)

def run_cell(actor, scenario, seeds, max_steps=None):
    # hover/land need travel + maneuver time on top of the goto budget
    if max_steps is None:
        max_steps = 400 if scenario == "goto" else 700
    succ, extra = [], []
    for seed in seeds:
        dist = cell_dist(seed)
        spec = sample_spec(seed, force_scenario=scenario)
        env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                               scenario_spec=spec)(seed)
        obs = env.reset()
        hold_speeds = []
        for _ in range(max_steps):
            a = actor(obs, dist.scene_extent)
            obs, r, done = env.step(a)
            info = getattr(env, "last_info", {})
            if info.get("hold_steps", 0) > 0:
                v = info.get("vel", [0, 0, 0])
                hold_speeds.append(float(np.linalg.norm(v)))
            if done:
                break
        ok = bool(env.succeeded)
        succ.append(ok)
        if scenario == "land" and ok:
            extra.append(abs(float(env.last_info["vel"][1])))
        if scenario == "hover_hold" and hold_speeds:
            extra.append(float(np.mean(hold_speeds)))
        env.close()
    return float(np.mean(succ)), (float(np.mean(extra)) if extra else None)

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "teacher"
    publish = "--publish" in sys.argv
    if target == "teacher":
        from diverse_bc import pilot_act3
        actor = lambda obs, ext: pilot_act3(obs, 1.75, ext)
        tag = "teacher"
    else:
        os.environ.setdefault("AUTORESEARCH_OBS_V2", "1")
        from policy import MLP
        net = MLP(19, 4, seed=0)
        net.set_flat(np.array(json.load(open(target)), dtype=np.float64))
        actor = lambda obs, ext: net.act(obs)
        tag = os.path.basename(target).replace(".json", "")
    results = {}
    for scenario, seeds in heldout_cells().items():
        s, metric = run_cell(actor, scenario, seeds)
        results[scenario] = {"success": s, "metric": metric}
        print("EVAL", tag, scenario, json.dumps(results[scenario]), flush=True)
        if publish:
            post_series("success_" + scenario, s, tag)
            if metric is not None:
                mname = "touchdown_vs" if scenario == "land" else "hold_drift"
                post_series(mname + "_" + scenario, metric, tag)
    json.dump(results, open(f"/workspace/eval_{tag}_scenarios.json", "w"), indent=2)

if __name__ == "__main__":
    main()
