"""m2 land probe (dag9 follow-up): why is land_t0 pinned at 0.0 under motor_v2?

A) Teacher census on heldout land t0 seeds (integrator teacher, per-step
   trace): classify every failure - stall-above-gate (equilibrium ~1.42m,
   gate never fires), park-in-gate (gate fires, vy~0 sustained), touchdown
   too fast (|vs|>0.5), off-pad, obstacle crash.
B) Forced-descent reachability: same seeds; teacher drives transit, but once
   in the gate region the terminal thrust law (vy_des=-0.25 + SoC trim) is
   FORCED regardless of the gate flag. Tests whether the touchdown criterion
   (dxz<=radius, |vs|<=0.5) is reachable under m2 at all.
C) Student gap: dag8c r24 actor thrust vs teacher thrust along teacher
   traces (terminal region vs rest), plus the student's own failure classes.
"""
import sys, os, json
os.environ["AUTORESEARCH_MOTOR_V2"] = "1"
os.environ["AUTORESEARCH_OBS_V4"] = "1"
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from scenario_sampler import sample_spec, tier_dist, heldout_cells_tiered
from env_sim import make_sim_factory
import t4_pilot
from t4_pilot import teacher_act, HOVER_THR, K_VTHR, LAND_VY_DESCEND
from t4_dagger9 import wp_fwd27, warm_params
from ppo import MLP
from policy import MLP as BCMlp

MANIFEST = t4_pilot.MANIFEST
OBS_DIM, ACT_DIM, HID = 27, 4, 256
SEEDS = heldout_cells_tiered()[("land", 0)]

def mk(seed):
    dist = tier_dist(seed, 0)
    spec = sample_spec(seed, force_scenario="land")
    dist.n_waypoints = 0.0
    env = make_sim_factory(dist, max_steps=700, dynamics=MANIFEST, scenario_spec=spec)(seed)
    return env, float(spec["success_radius"])

def trace(env, radius, act_fn):
    obs = env.reset()
    st = {}
    rows = []
    done = False
    while not done:
        a = act_fn(obs, radius, st)
        ext = obs[15] if False else None
        alt = -float((obs[0:3] + obs[19:22])[1]) * 1.0
        # rel in meters for logging (ext cancels: obs rel is normalized by ext)
        relm = (obs[0:3] + obs[19:22])
        vel = obs[3:6] * 10.0
        obs, r, done = env.step(a)
        rows.append((float(alt), float(np.hypot(relm[0], relm[2])), float(vel[1]), float(a[3]),
                     bool(st.get("_gate", False))))
    info = env.last_info
    return rows, bool(env.succeeded), bool(env.collided), info

def teacher_fn(obs, radius, st):
    ext = 1.0
    # teacher_act expects normalized obs as produced by env; it internally
    # scales by ext = scene extent - recover it the way the pilot does:
    raise RuntimeError("placeholder")

def run_teacher(env, radius):
    obs = env.reset()
    st = {}
    rows = []
    done = False
    # recover extent the way t4_pilot does: rel = (obs[0:3]+obs[19:22]) * ext
    # ext is scene extent; env_sim stores it on dist - read from env
    ext = float(env.dist.scene_extent)
    while not done:
        gate = (bool(-float((obs[0:3] + obs[19:22])[1]) * ext <= 1.4
                and np.hypot((obs[0:3] + obs[19:22])[0] * ext, (obs[0:3] + obs[19:22])[2] * ext) <= max(1.0, 2.0 * radius)))
        st["_gate"] = gate
        a = teacher_act(obs, ext, "land", radius, state=st)
        relm = (obs[0:3] + obs[19:22]) * ext
        vel = obs[3:6] * 10.0
        obs, r, done = env.step(a)
        rows.append((float(-relm[1]), float(np.hypot(relm[0], relm[2])), float(vel[1]), float(a[3]), gate))
    return rows, bool(env.succeeded), bool(env.collided), env.last_info

def run_forced(env, radius):
    """Teacher transit, but in the gate region FORCE the descent thrust law."""
    obs = env.reset()
    st = {}
    trim = 0.0
    rows = []
    done = False
    ext = float(env.dist.scene_extent)
    while not done:
        a = np.array(teacher_act(obs, ext, "land", radius, state=st), dtype=np.float64)
        relm = (obs[0:3] + obs[19:22]) * ext
        vel = obs[3:6] * 10.0
        alt = -float(relm[1]); dxz = float(np.hypot(relm[0], relm[2]))
        in_region = alt <= 1.45 and dxz <= max(1.0, 2.0 * radius)
        if in_region:
            vy_des = LAND_VY_DESCEND
            thr = HOVER_THR + K_VTHR * (vy_des - float(vel[1])) + trim
            trim = float(np.clip(trim + 0.0008 * (vy_des - float(vel[1])), -0.12, 0.12))
            a[3] = np.clip(thr, -1.0, 1.0)
        obs, r, done = env.step(a)
        rows.append((alt, dxz, float(vel[1]), float(a[3]), in_region))
    return rows, bool(env.succeeded), bool(env.collided), env.last_info

def classify(rows, succ, collided, info, radius):
    if succ:
        return "success"
    pos = info.get("pos", [None] * 3)
    vel = info.get("vel", [None] * 3)
    if collided:
        if pos[1] is not None and pos[1] < 0.15:
            dxz = float(np.hypot(pos[0], pos[2]))  # pos is world; pad at goal x/z - approx via info? use rows dxz
            vs = abs(vel[1]) if vel[1] is not None else -1
            return "touchdown_too_fast(vs=%.2f)" % vs if (rows and rows[-1][1] <= radius) else "off_pad(dxz=%.2f)" % (rows[-1][1] if rows else -1)
        return "obstacle_crash"
    # timeout: stall classification
    tail = rows[-50:] if len(rows) >= 50 else rows
    mean_vy = float(np.mean([abs(r[2]) for r in tail])) if tail else -1
    any_gate = any(r[4] for r in tail)
    alt = tail[-1][0] if tail else -1
    if mean_vy < 0.08:
        return ("park_in_gate(alt=%.2f)" % alt) if any_gate else ("stall_above_gate(alt=%.2f)" % alt)
    return "timeout_moving(vy=%.2f,alt=%.2f)" % (mean_vy, alt)

def student_actor():
    flat = np.array(json.load(open("/workspace/t4_dag8c_r24.json")), dtype=np.float64)
    rng = np.random.default_rng(0)
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
    WP_IN, WP_HID = 7, HID
    n_wp = WP_IN * WP_HID + WP_HID * ACT_DIM
    wp2 = flat[-WP_HID * ACT_DIM:].reshape(WP_HID, ACT_DIM)
    wp1 = flat[-WP_HID * ACT_DIM - WP_IN * WP_HID:-WP_HID * ACT_DIM].reshape(WP_IN, WP_HID)
    trunk = flat[:-n_wp]
    bc = BCMlp(OBS_DIM, ACT_DIM, hidden=HID, seed=0)
    bc.set_flat(trunk)
    actor.load({"w1": bc.W1.copy(), "b1": bc.b1.copy(), "w2": bc.W2.copy(),
                "b2": bc.b2.copy(), "w3": bc.W3.copy(), "b3": bc.b3.copy()})
    return actor, wp1, wp2

def student_act(actor, wp1, wp2, obs):
    mu, _ = actor.forward(obs[None, :])
    wpmu, _ = wp_fwd27(obs[None, :], wp1, wp2)
    return np.tanh(mu[0] + wpmu[0])

out = {"seeds": len(SEEDS)}

# ---- Phase A: teacher census
pa = {}
traces = {}
for s in SEEDS:
    env, radius = mk(s)
    rows, succ, collided, info = run_teacher(env, radius)
    traces[s] = (rows, radius)
    cls = classify(rows, succ, collided, info, radius)
    pa[cls.split("(")[0]] = pa.get(cls.split("(")[0], 0) + 1
    print(f"A seed={s} {cls}", flush=True)
    env.close()
out["A_teacher_census"] = pa

# ---- Phase B: forced descent
pb = {}
vs_list = []
for s in SEEDS:
    env, radius = mk(s)
    rows, succ, collided, info = run_forced(env, radius)
    cls = classify(rows, succ, collided, info, radius)
    pb[cls.split("(")[0]] = pb.get(cls.split("(")[0], 0) + 1
    if info.get("vel"): vs_list.append(abs(float(info["vel"][1])))
    print(f"B seed={s} {cls}", flush=True)
    env.close()
out["B_forced_descent"] = pb
out["B_touchdown_vs"] = {"mean": float(np.mean(vs_list)) if vs_list else None,
                         "max": float(np.max(vs_list)) if vs_list else None}

# ---- Phase C: student gap on teacher traces + student solo failures
actor, wp1, wp2 = student_actor()
gaps_term, gaps_rest = [], []
for s in SEEDS:
    rows, radius = traces[s]
    env, radius2 = mk(s)
    obs = env.reset()
    st = {}
    done = False
    ext = float(env.dist.scene_extent)
    while not done:
        ta = teacher_act(obs, ext, "land", radius, state=st)
        sa = student_act(actor, wp1, wp2, obs)
        relm = (obs[0:3] + obs[19:22]) * ext
        alt = -float(relm[1]); dxz = float(np.hypot(relm[0], relm[2]))
        gap = float(sa[3] - ta[3])
        if alt <= 1.45 and dxz <= max(1.0, 2.0 * radius):
            gaps_term.append(gap)
        else:
            gaps_rest.append(gap)
        obs, r, done = env.step(ta)
    env.close()
out["C_gap_terminal"] = {"mean": float(np.mean(gaps_term)), "p10": float(np.percentile(gaps_term, 10)),
                          "p90": float(np.percentile(gaps_term, 90)), "n": len(gaps_term)}
out["C_gap_rest"] = {"mean": float(np.mean(gaps_rest)), "n": len(gaps_rest)}

pc = {}
for s in SEEDS:
    env, radius = mk(s)
    obs = env.reset()
    done = False
    rows = []
    while not done:
        a = student_act(actor, wp1, wp2, obs)
        relm = (obs[0:3] + obs[19:22])
        vel = obs[3:6] * 10.0
        obs, r, done = env.step(a)
        rows.append((float(-relm[1]), float(np.hypot(relm[0], relm[2])), float(vel[1]), float(a[3]), False))
    cls = classify(rows, bool(env.succeeded), bool(env.collided), env.last_info, radius)
    pc[cls.split("(")[0]] = pc.get(cls.split("(")[0], 0) + 1
    print(f"C seed={s} student {cls}", flush=True)
    env.close()
out["C_student_census"] = pc

json.dump(out, open("/workspace/m2_land_probe.json", "w"), indent=1)
print("PROBE_SUMMARY " + json.dumps(out), flush=True)
print("M2_LAND_PROBE_DONE", flush=True)
