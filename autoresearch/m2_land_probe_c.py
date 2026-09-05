"""Phase C only: dag8c r24 student thrust gap vs teacher + student solo census."""
import sys, os, json
os.environ["AUTORESEARCH_MOTOR_V2"] = "1"
os.environ["AUTORESEARCH_OBS_V4"] = "1"
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from scenario_sampler import sample_spec, tier_dist, heldout_cells_tiered
from env_sim import make_sim_factory
import t4_pilot
from t4_pilot import teacher_act
import t4_common as P
from t4_dagger9 import wp_fwd27
from ppo import MLP
from policy import MLP as BCMlp

SEEDS = heldout_cells_tiered()[("land", 0)]
MANIFEST = t4_pilot.MANIFEST
OBS_DIM, ACT_DIM, HID = 27, P.ACT_DIM, P.HID
WP_IN, WP_HID = P.WP_IN, P.WP_HID

def mk(seed):
    dist = tier_dist(seed, 0)
    spec = sample_spec(seed, force_scenario="land")
    dist.n_waypoints = 0.0
    env = make_sim_factory(dist, max_steps=700, dynamics=MANIFEST, scenario_spec=spec)(seed)
    return env, float(spec["success_radius"])

def classify(rows, succ, collided, info, radius):
    if succ:
        return "success"
    pos = info.get("pos", [None] * 3)
    vel = info.get("vel", [None] * 3)
    if collided:
        if pos[1] is not None and pos[1] < 0.15:
            vs = abs(vel[1]) if vel[1] is not None else -1
            return "touchdown_too_fast(vs=%.2f)" % vs if (rows and rows[-1][1] <= radius) else "off_pad(dxz=%.2f)" % (rows[-1][1] if rows else -1)
        return "obstacle_crash"
    tail = rows[-50:] if len(rows) >= 50 else rows
    mean_vy = float(np.mean([abs(r[2]) for r in tail])) if tail else -1
    alt = tail[-1][0] if tail else -1
    if mean_vy < 0.08:
        return "park_in_gate(alt=%.2f)" % alt
    return "timeout_moving(vy=%.2f,alt=%.2f)" % (mean_vy, alt)

flat = np.array(json.load(open("/workspace/t4_dag8c_r24.json")), dtype=np.float64)
rng = np.random.default_rng(0)
actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
n2 = WP_HID * ACT_DIM; n1 = WP_IN * WP_HID
wp2 = flat[-n2:].reshape(WP_HID, ACT_DIM)
wp1 = flat[-n2 - n1:-n2].reshape(WP_IN, WP_HID)
trunk = flat[:-n1 - n2]
bc = BCMlp(OBS_DIM, ACT_DIM, hidden=HID, seed=0)
bc.set_flat(trunk)
actor.load({"w1": bc.W1.copy(), "b1": bc.b1.copy(), "w2": bc.W2.copy(),
            "b2": bc.b2.copy(), "w3": bc.W3.copy(), "b3": bc.b3.copy()})

def student_act(obs):
    mu, _ = actor.forward(obs[None, :])
    wpmu, _ = wp_fwd27(obs[None, :], wp1, wp2)
    return np.tanh(mu[0] + wpmu[0])

gaps_term, gaps_rest = [], []
for s in SEEDS:
    env, radius = mk(s)
    obs = env.reset()
    st = {}
    done = False
    ext = float(env.dist.scene_extent)
    while not done:
        ta = teacher_act(obs, ext, "land", radius, state=st)
        sa = student_act(obs)
        relm = (obs[0:3] + obs[19:22]) * ext
        alt = -float(relm[1]); dxz = float(np.hypot(relm[0], relm[2]))
        g = float(sa[3] - ta[3])
        (gaps_term if (alt <= 1.45 and dxz <= max(1.0, 2.0 * radius)) else gaps_rest).append(g)
        obs, r, done = env.step(ta)
    env.close()
print("C gap terminal: mean=%.3f p10=%.3f p50=%.3f p90=%.3f n=%d" % (
    np.mean(gaps_term), np.percentile(gaps_term, 10), np.percentile(gaps_term, 50),
    np.percentile(gaps_term, 90), len(gaps_term)), flush=True)
print("C gap rest: mean=%.3f n=%d" % (np.mean(gaps_rest), len(gaps_rest)), flush=True)

pc = {}
for s in SEEDS:
    env, radius = mk(s)
    obs = env.reset()
    done = False
    rows = []
    while not done:
        a = student_act(obs)
        relm = (obs[0:3] + obs[19:22])
        vel = obs[3:6] * 10.0
        obs, r, done = env.step(a)
        rows.append((float(-relm[1]), float(np.hypot(relm[0], relm[2])), float(vel[1]), float(a[3]), False))
    cls = classify(rows, bool(env.succeeded), bool(env.collided), env.last_info, radius)
    pc[cls.split("(")[0]] = pc.get(cls.split("(")[0], 0) + 1
    print("C seed=%d student %s" % (s, cls), flush=True)
    env.close()
print("C_STUDENT_CENSUS " + json.dumps(pc), flush=True)
print("PROBE_C_DONE", flush=True)
