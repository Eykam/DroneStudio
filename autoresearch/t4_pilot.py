"""T4 stage 1: unified teacher demos - one cascaded teacher, ALL tiers (obs v3.1).

User decision 2026-09-04: fresh HID-64 full bootstrap, one unified policy,
whole chain (pilot -> BC -> DAgger -> PPO) at the new width on tiered
scenes + obs v3. This collects the demo set.

Teacher = diverse_bc.pilot_act3 (the scenario-aware champion-era teacher:
goto + hover-hold braking + two-phase land + potential-field obstacle
avoidance) adapted for obs v3.1 (current target = obs[0:3]+obs[19:22],
waypoint delta zeroes out on T0-T2) and the v14 fixture hover throttle
(-0.756, from t3_pilot; pilot_act3's -0.778 was the 11N/motor fixture).
Scenario + success radius come from the spec, not the obs one-hot.

Successful episodes only (mixed-quality lesson from earlier BC rounds).
T3 uses curriculum phases A/B only (C/D teach dying).

Output: /workspace/t4_demos.npz (X, A, meta).
"""
import sys, json, os, time
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from env_sim import make_sim_factory
from t3_pilot import t3_dist, PHASES

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/v14_g13.manifest.json"

def teacher_act(obs, ext, scenario, radius):
    rel = (obs[0:3] + obs[19:22]) * ext   # current-target rel, meters
    vel = obs[3:6] * 10.0                 # world vel, m/s (obs = v/10)
    gb = obs[6:9]
    vmax = 2.0
    dvec = obs[12:15] * ext               # nearest-obstacle vector
    d = float(np.linalg.norm(dvec))
    dist_xz = float(np.hypot(rel[0], rel[2]))
    alt = -float(rel[1])                  # land pads sit at ground level
    land_descend = (scenario == "land" and alt <= 1.4
                    and dist_xz <= max(1.0, 2.0 * radius))
    if scenario == "land" and not land_descend:
        rel = rel + np.array([0.0, 1.2, 0.0])   # phase 1: above the pad
        vmax = min(vmax, 1.2)
    v_des = np.clip(0.5 * rel, -vmax, vmax)
    if scenario == "hover_hold" and np.linalg.norm(rel) < 2.0 * radius:
        v_des = np.clip(0.4 * rel, -0.25, 0.25)  # brake and hold
    if land_descend:
        v_des = np.array([np.clip(0.6 * rel[0], -0.3, 0.3), 0.0,
                          np.clip(0.6 * rel[2], -0.3, 0.3)])
    if scenario == "goto" and 1e-3 < d < 6.0:
        # cap only the velocity component TOWARD the obstacle: fast sliding
        # along walls stays allowed, gap passages stay possible
        u = dvec / d
        allow = max(0.3, 0.6 * (d - 0.8))
        v_to = float(np.dot(v_des, u))
        if v_to > allow:
            v_des = v_des - (v_to - allow) * u
    # damping: 0.5 far, 1.0 near (half-damping limit-cycles at the target)
    damp = 0.5 if (float(np.linalg.norm(rel)) > 3.0
                   and not (scenario == "goto" and d < 6.0)) else 1.0
    a_des = np.clip(1.2 * (v_des - damp * vel), -2.0, 2.0)
    if 1e-3 < d < 3.5 and not land_descend:   # pad corridor stays clear
        a_des = a_des - 3.0 * (3.5 - d) / 3.5 * (dvec / d)
    if scenario == "goto" and 1e-3 < d < 5.0:
        # fixed-physics approach is faster (~4.2 m/s); brake actively for walls
        u = dvec / d
        closing = float(np.dot(vel, u))
        if closing > 0.0:
            a_des = a_des - min(2.5, 1.4 * closing) * u
    a_des = np.clip(a_des, -3.0, 3.0)
    gx_des = np.clip(a_des[0] / 9.81, -0.30, 0.30)
    gz_des = np.clip(a_des[2] / 9.81, -0.30, 0.30)
    rates = obs[9:12]                     # obs scale (/10), as the v1/v3 pilots used
    kp, kd = 0.4, 0.6                     # v14-proven pair (t3_pilot); kd=1.0 was the 11N fixture
    act0 = kp * (gz_des - gb[2]) - kd * rates[0]
    act2 = -kp * (gx_des - gb[0]) - kd * rates[2]
    if land_descend:
        vy_des = -0.4
    elif scenario == "hover_hold":
        vy_des = np.clip(0.8 * rel[1], -0.8, 0.8)
    else:
        vy_des = np.clip(0.8 * rel[1], -1.5, 1.5)
    thr = -0.756 + 0.3 * (vy_des - vel[1])
    # bearing-seeking yaw (fixed flight stack supports closed-loop yaw now):
    # gently face the target; obs rel is yaw-frame so atan2 gives the
    # heading error directly. Disabled during precision land descent.
    yaw_cmd = 0.0
    if not land_descend:
        yaw_err = float(np.arctan2(-rel[2], rel[0]))  # sign verified on fixed physics (yawpure test)
        if abs(yaw_err) < 2.0:
            # refine heading only when roughly facing the target; a full
            # turn at the atan2 antipode bang-bangs and traps the position
            # loop - translate out instead
            yaw_cmd = float(np.clip(0.5 * yaw_err, -0.04, 0.04))
    return np.clip(np.array([act0, yaw_cmd, act2, thr]), -1, 1)

def run_one(seed, dist, spec, max_steps):
    ext = float(dist.scene_extent)
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    traj = []
    for _ in range(max_steps):
        a = teacher_act(obs, ext, spec["scenario"], float(spec["success_radius"]))
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
                seed = 80000 + tier * 10000 + j
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
