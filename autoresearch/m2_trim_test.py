"""T11: validate the m2 trim integrator across scenarios (winner constants)."""
import sys, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
os.environ["AUTORESEARCH_MOTOR_V2"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist, hover_max_steps
from env_sim import make_sim_factory
import t4_pilot

def run_ep(seed, sc, tier, ms):
    dist = tier_dist(seed, tier)
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

for sc, tier, n, ms in (("hover_hold", 0, 6, None), ("land", 0, 6, 1200), ("goto", 0, 4, 400)):
    outs = []
    for j in range(n):
        seed = 620000 + j
        dist_hold = 4.0
        m = ms if ms else hover_max_steps(dist_hold, tier)
        outs.append(run_ep(seed, sc, tier, m))
    sr = float(np.mean([o[0] for o in outs]))
    print(f"{sc} t{tier}: succ {sr:.3f} " +
          " ".join(f"({'OK' if o[0] else 'X'} alt={o[1]:.2f} vy={o[2]:.2f} dxz={o[3]:.2f})" for o in outs), flush=True)
print("M2_TRIM_TEST_DONE", flush=True)
