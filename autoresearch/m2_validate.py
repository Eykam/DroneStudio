"""T11: larger-n m2 teacher validation (16 seeds/scenario, tier 0)."""
import sys, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
os.environ["AUTORESEARCH_MOTOR_V2"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist, hover_max_steps
from env_sim import make_sim_factory
import t4_pilot

def run_ep(seed, sc, ms):
    dist = tier_dist(seed, 0)
    spec = sample_spec(seed, force_scenario=sc)
    if sc != "goto":
        dist.n_waypoints = 0.0
    ext = float(dist.scene_extent)
    env = make_sim_factory(dist, max_steps=ms, dynamics=t4_pilot.MANIFEST, scenario_spec=spec)(seed)
    obs = env.reset()
    st = {}
    for _ in range(ms):
        a = t4_pilot.teacher_act(obs, ext, sc, float(spec["success_radius"]), state=st)
        obs, r, done = env.step(a)
        if done:
            break
    ok = bool(env.succeeded)
    rel = (obs[0:3] + obs[19:22]) * ext
    vel = obs[3:6] * 10.0
    env.close()
    return ok, -float(rel[1]), float(vel[1]), float(np.hypot(rel[0], rel[2]))

for sc, ms in (("hover_hold", None), ("goto", 400), ("land", 1200)):
    outs = []
    for j in range(16):
        m = ms if ms else hover_max_steps(4.0, 0)
        outs.append(run_ep(620000 + j, sc, m))
    sr = float(np.mean([o[0] for o in outs]))
    fails = [o for o in outs if not o[0]]
    print(f"{sc} t0 n=16: succ {sr:.3f}", flush=True)
    for o in fails:
        print(f"   FAIL alt={o[1]:.2f} vy={o[2]:.2f} dxz={o[3]:.2f}", flush=True)
print("M2_VALIDATE_DONE", flush=True)
