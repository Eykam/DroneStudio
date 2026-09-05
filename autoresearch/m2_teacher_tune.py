"""T11: retune the scripted teacher for the motor_v2 plant.

The t4 teacher is hand-tuned for the 40ms lag plant; under motor_v2's
~200-300ms EM spin-up the vertical P loop (gain 0.3) porpoises and
collides. This sweeps (K_VTHR x KP_ATT/KD_ATT) on hover_hold tier-0
episodes under AUTORESEARCH_MOTOR_V2=1 and reports success + vertical
oscillation. HOVER_THR stays at the T9 analytic -0.742 (validated stable
hover). Winner gets a goto/land sanity pass. Constants live env-gated in
t4_pilot.py (M2 branch); base-plant tuning is untouched.
"""
import sys, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
os.environ["AUTORESEARCH_MOTOR_V2"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist, hover_max_steps
from env_sim import make_sim_factory
import t4_pilot

MANIFEST = t4_pilot.MANIFEST

def run_ep(seed, sc, tier):
    dist = tier_dist(seed, tier)
    spec = sample_spec(seed, force_scenario=sc)
    if sc != "goto":
        dist.n_waypoints = 0.0
    ext = float(dist.scene_extent)
    ms = 400 if sc == "goto" else hover_max_steps(spec.get("hold_s", 4.0), tier) if sc == "hover_hold" else 700
    env = make_sim_factory(dist, max_steps=ms, dynamics=MANIFEST, scenario_spec=spec)(seed)
    obs = env.reset()
    vys = []
    for _ in range(ms):
        a = t4_pilot.teacher_act(obs, ext, sc, float(spec["success_radius"]))
        vys.append(float(obs[4]) * 10.0)
        obs, r, done = env.step(a)
        if done:
            break
    ok = bool(env.succeeded)
    env.close()
    tail = vys[len(vys) // 3:] if len(vys) > 6 else vys
    return ok, float(np.std(tail)) if tail else 9.9

print("sweep: hover_hold t0, 6 seeds each (m2 plant)", flush=True)
results = []
for kv in (0.05, 0.10, 0.15, 0.20, 0.30):
    for kp, kd in ((0.40, 0.60), (0.30, 0.45), (0.25, 0.40)):
        t4_pilot.K_VTHR = kv
        t4_pilot.KP_ATT, t4_pilot.KD_ATT = kp, kd
        oks, oscs = [], []
        for j in range(6):
            ok, osc = run_ep(600000 + j, "hover_hold", 0)
            oks.append(ok); oscs.append(osc)
        sr = float(np.mean(oks)); om = float(np.mean(oscs))
        results.append((sr, -om, kv, kp, kd))
        print(f"K_VTHR={kv:.2f} KP={kp:.2f} KD={kd:.2f}: succ {sr:.3f} osc {om:.3f}", flush=True)

results.sort(reverse=True)
sr, nom, kv, kp, kd = results[0]
print(f"WINNER K_VTHR={kv} KP={kp} KD={kd} succ={sr:.3f} osc={-nom:.3f}", flush=True)
t4_pilot.K_VTHR = kv
t4_pilot.KP_ATT, t4_pilot.KD_ATT = kp, kd
for sc in ("goto", "land"):
    oks = [run_ep(610000 + j, sc, 0)[0] for j in range(4)]
    print(f"sanity {sc} t0: succ {np.mean(oks):.3f}", flush=True)
print("M2_TUNE_DONE", flush=True)
