"""T4 stage 1: unified teacher demos - one cascaded teacher, ALL tiers (obs v3.1).

User decision 2026-09-04: fresh HID-64 full bootstrap, one unified policy,
whole chain (pilot -> BC -> DAgger -> PPO) at the new width on tiered
scenes + obs v3. This collects the demo set.

teacher_act (t3_pilot) steers at obs[0:3]+obs[19:22] = current target -
identical math on every tier because obs v3.1 zeroes the wp delta when no
waypoints are pending. Successful episodes only (mixed-quality lesson from
earlier BC rounds). T3 uses curriculum phases A/B only (C/D teach dying).

Output: /workspace/t4_demos.npz (X, A, meta).
"""
import sys, json, os, time
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
from t3_pilot import teacher_act, t3_dist, PHASES

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/v14_g13.manifest.json"

def run_one(seed, dist, spec, max_steps):
    ext = float(dist.scene_extent)
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    traj = []
    for _ in range(max_steps):
        a = teacher_act(obs, ext)
        traj.append((obs.copy(), a.copy()))
        obs, r, done = env.step(a)
        if done: break
    ok = bool(env.succeeded)
    env.close()
    return traj if ok else None

def main():
    t0 = time.time()
    X, A, meta = [], [], {}
    for tier in (0, 1, 2):
        for sc in ("goto", "hover_hold", "land"):
            ks = 0
            for j in range(48):
                seed = 80000 + tier * 10000 + hash(sc) % 1000 * 3 + j
                dist = tier_dist(seed, tier)
                spec = sample_spec(seed, force_scenario=sc)
                if sc != "goto":
                    dist.n_waypoints = 0.0
                tr = run_one(seed, dist, spec, 400 if sc == "goto" else 700)
                if tr:
                    ks += 1
                    X += [t[0] for t in tr]; A += [t[1] for t in tr]
            meta[f"{sc}_t{tier}"] = ks
            print(f"collect {sc}_t{tier}: kept {ks}/48 samples={len(X)}", flush=True)
    for k, ph in enumerate(("A", "B")):
        ks = 0
        for j in range(64):
            seed = 90000 + k * 1000 + j
            dist = t3_dist(seed, PHASES[ph])
            spec = sample_spec(seed, force_scenario="goto")
            tr = run_one(seed, dist, spec, 600)
            if tr:
                ks += 1
                X += [t[0] for t in tr]; A += [t[1] for t in tr]
        meta[f"t3_{ph}"] = ks
        print(f"collect t3_{ph}: kept {ks}/64 samples={len(X)}", flush=True)
    np.savez("/workspace/t4_demos.npz", X=np.array(X), A=np.array(A), meta=json.dumps(meta))
    print("T4PILOT_DONE " + json.dumps(meta) + f" samples={len(X)} wall={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
