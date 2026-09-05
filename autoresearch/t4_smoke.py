import sys, os, json
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
from scenario_sampler import sample_spec, tier_dist, hover_max_steps
from env_sim import make_sim_factory
from t4_pilot import teacher_act, MANIFEST
for sc, tier, ms in (("goto",0,400),("hover_hold",0,None),("land",0,700),("goto",2,400)):
    dist = tier_dist(80000 + tier*10000, tier)
    spec = sample_spec(80000 + tier*10000, force_scenario=sc)
    if sc == "hover_hold": ms = hover_max_steps(spec.get("hold_s", 4.0), tier)
    if sc != "goto": dist.n_waypoints = 0.0
    env = make_sim_factory(dist, max_steps=ms, dynamics=MANIFEST, scenario_spec=spec)(80000 + tier*10000)
    obs = env.reset(); ext = float(dist.scene_extent)
    for _ in range(ms):
        obs, r, done = env.step(teacher_act(obs, ext, sc, float(spec["success_radius"])))
        if done: break
    print(json.dumps({"sc": sc, "tier": tier, "succ": bool(env.succeeded), "steps": env.steps}), flush=True)
    env.close()
