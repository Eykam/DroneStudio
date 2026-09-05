"""T4 PPO-6: residual RL pilot (2026-09-04, approved SIM next step).

Why residual: five full/partial-net PPO variants (t4_ppo..t4_ppo5) all
collapsed hover_hold to ~0.0 within 80 updates while goto survived
(best_mean 0.45-0.56 vs the DAgger champion 0.799): on-policy return
chasing trades precision away. This pilot freezes the champion outright
(trunk + wp pathway + exploration std) and learns ONLY a bounded additive
residual on the pre-tanh action:

    mu = champion_mu(obs) + RESID_BOUND * tanh(R2 @ relu(R1 @ obs))

R2 is zero-init, so the u0 policy is bit-identical to t4_best - RL starts
from 0.799-cell behavior and can only add bounded corrections. log_std is
frozen at exp(-2.5) (training it was part of the collapse dynamic).

Recipe carried from t4_ppo5 where earned: outcome-dominant shaping +50 /
alive 0.02 / -10 / -10, wp-pass +10, yaw-facing lambda 0.01, gamma 0.999,
lam 0.95, per-scenario advantage normalization, T3 soft phase curriculum,
land-heavy mix, critic warmup 5 updates, floors goto_t0>=0.9 AND
hover_hold_t0>=0.6 AND land_t0>=0.7, abort after 8 consecutive floor
failures.

Checkpoints {"champion": flat, "R1": ..., "R2": ...} -> t4_ppo6_uN.json,
best-with-floors -> t4_ppo6_best.json. Series "t4-ppo6 uN" per cell +
t4_ppo6_return / t4_ppo6_face. t4_best.json never touched.
"""
import sys, json, os, time
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scene_schema import SceneDistribution
from env_sim import make_sim_factory
from ppo import MLP, Adam
from scenario_sampler import sample_spec, sample_tier, tier_dist, hover_max_steps
import t4_common as P

MANIFEST = P.MANIFEST
OBS_DIM, ACT_DIM, HID, V2_DIM = P.OBS_DIM, P.ACT_DIM, P.HID, P.V2_DIM
RESID_HID, RESID_BOUND = 32, 0.3
LOG_STD = np.full(ACT_DIM, -2.5)          # frozen
LR_R, LR_C, CLIP, GAMMA, LAM = 1e-3, 1e-3, 0.2, 0.999, 0.95
EPOCHS, MINIBATCH = 4, 2048
UPDATES, EPISODES_PER_UPDATE = 60, 32
PHASES = [{"lat": 0.05, "dens": 0.0,  "corr": 4.0},
          {"lat": 0.10, "dens": 0.02, "corr": 4.0},
          {"lat": 0.20, "dens": 0.05, "corr": 4.0},
          {"lat": 0.35, "dens": 0.10, "corr": 2.0}]
def phase_weights(u):
    if u <= 10: return [1.0, 0.0, 0.0, 0.0]
    if u <= 30: return [0.6, 0.4, 0.0, 0.0]
    if u <= 50: return [0.3, 0.5, 0.2, 0.0]
    if u <= 65: return [0.0, 0.3, 0.5, 0.2]
    return [0.0, 0.0, 0.4, 0.6]
CRITIC_WARMUP = 5
YAW_LAMBDA = float(os.environ.get("T4_YAW_LAMBDA", "0.01"))
SUCCESS_BONUS, CRASH_PENALTY, TIMEOUT_PENALTY = 50.0, 10.0, 10.0
ALIVE_BONUS = 0.02
MIX = ("land", "land", "land", "hover_hold", "goto", "goto")

def scene_for(seed, scenario, knobs):
    tier = sample_tier(seed)
    if tier == 3:
        scenario = "goto"
        base = tier_dist(seed, 3)
        dist = SceneDistribution.from_vector(base.to_vector())
        dist.waypoint_lat = knobs["lat"]
        dist.obstacle_density = knobs["dens"]
        dist.corridor_width = knobs["corr"]
    else:
        dist = tier_dist(seed, tier)
    spec = sample_spec(seed, force_scenario=scenario)
    if spec["scenario"] != "goto":
        dist.n_waypoints = 0.0
    return dist, spec, tier

def resid_mu(obs_row, R1, R2):
    h1 = np.maximum(obs_row[None, :] @ R1, 0.0)
    return RESID_BOUND * np.tanh(h1 @ R2)[0]

def rollout_one(actor_flat, R1l, R2l, critic_flat, seed, scenario, knobs):
    rng = np.random.default_rng(np.uint64(seed))
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM); actor.load(P.bc_to_actor_params(actor_flat))
    wp1, wp2 = P.unpack_wp(actor_flat)
    R1 = np.array(R1l, dtype=np.float64); R2 = np.array(R2l, dtype=np.float64)
    critic = MLP(rng, OBS_DIM, HID, 1)
    keys = ("w1", "b1", "w2", "b2", "w3", "b3")
    p, i = {}, 0
    for k in keys:
        ref = getattr(critic, k)
        n = ref.size
        p[k] = np.array(critic_flat[i:i + n], dtype=np.float64).reshape(ref.shape)
        i += n
    critic.load(p)
    std = np.exp(LOG_STD)
    dist, spec, tier = scene_for(seed, scenario, knobs)
    max_steps = 600 if tier == 3 else (400 if spec["scenario"] == "goto" else 700)
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    assert len(obs) == OBS_DIM, f"obs dim {len(obs)} != {OBS_DIM}"
    obs_b, act_b, rew_b, val_b, logp_b, done_b = [], [], [], [], [], []
    done = False
    prev_wp = 0
    ext = float(dist.scene_extent)
    face_b = []
    while not done:
        mu, _ = actor.forward(obs[None, :])
        wpmu, _ = P.wp_forward(obs[None, :], wp1, wp2)
        mu = mu[0] + wpmu[0] + resid_mu(obs, R1, R2)
        a = mu + std * rng.standard_normal(ACT_DIM)
        logp = float(-0.5 * (((a - mu) / std) ** 2).sum()
                     - LOG_STD.sum() - 0.5 * ACT_DIM * np.log(2 * np.pi))
        v, _ = critic.forward(obs[None, :])
        nxt, r, done = env.step(np.tanh(a))
        relx = float((nxt[0] + nxt[19])) * ext
        relz = float((nxt[2] + nxt[21])) * ext
        rxz = float(np.hypot(relx, relz))
        face = 0.5 * (1.0 + float(np.cos(np.arctan2(relz, relx)))) if rxz > 0.5 else 1.0
        face_b.append(face)
        wp_now = int(env.last_info.get("wp", prev_wp))
        wp_passes = max(0, wp_now - prev_wp)
        prev_wp = wp_now
        r = ALIVE_BONUS + 10.0 * wp_passes + YAW_LAMBDA * face
        if done:
            if env.succeeded:
                r += SUCCESS_BONUS
            elif env.collided:
                r -= CRASH_PENALTY
            else:
                r -= TIMEOUT_PENALTY
        obs_b.append(obs); act_b.append(a); rew_b.append(float(r))
        val_b.append(float(v[0, 0])); logp_b.append(logp); done_b.append(bool(done))
        obs = nxt
    env.close()
    return obs_b, act_b, rew_b, val_b, logp_b, done_b, spec["scenario"], float(np.mean(face_b))

def eval_one_res(pack, scenario, tier, seed):
    rng = np.random.default_rng(0)
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM)
    actor.load(P.bc_to_actor_params(pack["champion"]))
    wp1, wp2 = P.unpack_wp(pack["champion"])
    R1 = np.array(pack["R1"], dtype=np.float64); R2 = np.array(pack["R2"], dtype=np.float64)
    hold_pin = None
    if scenario == "hover_hold60":
        scenario, hold_pin = "hover_hold", 60.0
    dist = tier_dist(seed, tier)
    spec = sample_spec(seed, force_scenario=scenario, hold_s=hold_pin)
    if scenario != "goto":
        dist.n_waypoints = 0.0
    max_steps = 600 if tier == 3 else (400 if scenario == "goto"
              else hover_max_steps(spec.get("hold_s", 4.0), tier) if scenario == "hover_hold" else 700)
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    done = False
    while not done:
        mu, _ = actor.forward(obs[None, :])
        wpmu, _ = P.wp_forward(obs[None, :], wp1, wp2)
        obs, r, done = env.step(np.tanh(mu[0] + wpmu[0] + resid_mu(obs, R1, R2)))
    ok = bool(env.succeeded)
    env.close()
    return ok

def eval_all_res(pack, label):
    from scenario_sampler import heldout_cells_tiered
    cells = heldout_cells_tiered()
    args = [(pack, sc, t, s) for (sc, t) in P.EVAL_CELLS
            for s in cells[("hover_hold", t) if sc == "hover_hold60" else (sc, t)]]
    res = P.parallel_episodes(eval_one_res, args)
    out, i = {}, 0
    for k in P.EVAL_CELLS:
        kk = ("hover_hold", k[1]) if k[0] == "hover_hold60" else k
        n = len(cells[kk])
        out[k] = float(np.mean(res[i:i + n]))
        i += n
    for (sc, t), v in out.items():
        P.post_series(f"success_{sc}_t{t}", v, label)
    return out

def critic_flat_of(critic):
    p = critic.params()
    return list(np.concatenate([p[k].ravel() for k in ("w1", "b1", "w2", "b2", "w3", "b3")]))

def main():
    rng = np.random.default_rng(11)
    flat = json.load(open("/workspace/t4_best.json"))
    WP1, WP2 = P.unpack_wp(flat)                      # frozen
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM)           # frozen reference
    actor.load(P.bc_to_actor_params(flat))
    R1 = np.random.default_rng(5).normal(0, 1.0 / np.sqrt(OBS_DIM), (OBS_DIM, RESID_HID))
    R2 = np.zeros((RESID_HID, ACT_DIM))               # zero -> u0 == champion
    r_opt = Adam({"R1": R1, "R2": R2}, lr=LR_R)
    critic = MLP(rng, OBS_DIM, HID, 1)
    critic_opt = Adam(critic.params(), lr=LR_C)

    pack = {"champion": flat, "R1": R1.tolist(), "R2": R2.tolist()}
    res0 = eval_all_res(pack, "t4-ppo6 u0")
    print("T4PPO6 u0 (residual=0 == champion): " +
          json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res0.items()}), flush=True)
    best_mean = -1e9
    json.dump(pack, open("/workspace/t4_ppo6_u0.json", "w"))
    floor_fails = 0

    for u in range(1, UPDATES + 1):
        t0 = time.time()
        c_flat = critic_flat_of(critic)
        wts = phase_weights(u)
        scs = [MIX[int(rng.integers(0, len(MIX)))] for _ in range(EPISODES_PER_UPDATE)]
        args = [(flat, R1.tolist(), R2.tolist(), c_flat,
                 int(rng.integers(0, 2**31 - 1)), sc,
                 PHASES[int(rng.choice(4, p=wts))]) for sc in scs]
        eps = P.parallel_episodes(rollout_one, args)

        obs_b, act_b, adv_b, ret_b, logp_b = [], [], [], [], []
        ep_scen = [e[6] for e in eps]
        mean_face = float(np.mean([e[7] for e in eps]))
        for ob, ab, rb, vb, lb, db, _sc, _f in eps:
            rew = np.array(rb); val = np.array(vb); done = np.array(db, dtype=np.float64)
            adv = np.zeros_like(rew); lastgae = 0.0
            for t in reversed(range(len(rew))):
                nv = val[t + 1] if t + 1 < len(rew) else 0.0
                nt = 1.0 - done[t]
                delta = rew[t] + GAMMA * nv * nt - val[t]
                lastgae = delta + GAMMA * LAM * nt * lastgae
                adv[t] = lastgae
            obs_b += ob; act_b += ab; logp_b += lb
            adv_b += list(adv); ret_b += list(adv + val)
        obs_b = np.array(obs_b); act_b = np.array(act_b)
        adv_b = np.array(adv_b); ret_b = np.array(ret_b); logp_b = np.array(logp_b)
        scen_arr = np.array([sc for sc, e in zip(ep_scen, eps) for _ in e[0]])
        for sc in set(ep_scen):
            m = scen_arr == sc
            adv_b[m] = (adv_b[m] - adv_b[m].mean()) / (adv_b[m].std() + 1e-8)
        mean_ret = float(np.mean([sum(e[2]) for e in eps]))

        std = np.exp(LOG_STD)
        n = len(adv_b); idx = np.arange(n)
        for _ in range(EPOCHS):
            rng.shuffle(idx)
            for start in range(0, n, MINIBATCH):
                mb = idx[start:start + MINIBATCH]
                mo, ma = obs_b[mb], act_b[mb]
                madv, mret, mlogp = adv_b[mb], ret_b[mb], logp_b[mb]
                mub, _ = actor.forward(mo)
                wpmu, _ = P.wp_forward(mo, WP1, WP2)
                pre1 = mo @ R1
                h1 = np.maximum(pre1, 0.0)
                z = h1 @ R2
                mu = mub + wpmu + RESID_BOUND * np.tanh(z)
                logp = (-0.5 * (((ma - mu) / std) ** 2).sum(-1)
                        - LOG_STD.sum() - 0.5 * ACT_DIM * np.log(2 * np.pi))
                ratio = np.exp(logp - mlogp)
                coeff = -madv * np.where(
                    ((madv >= 0) & (ratio < 1 + CLIP)) | ((madv < 0) & (ratio > 1 - CLIP)),
                    ratio, 0.0) / len(mb)
                if u > CRITIC_WARMUP:
                    dmu = coeff[:, None] * (ma - mu) / (std ** 2)[None, :]
                    dz = dmu * RESID_BOUND * (1.0 - np.tanh(z) ** 2)
                    gR2 = h1.T @ dz
                    dh = dz @ R2.T
                    dh[pre1 <= 0] = 0.0
                    gR1 = mo.T @ dh
                    rgrads = {"R1": gR1, "R2": gR2}
                    tot = np.sqrt(sum((g ** 2).sum() for g in rgrads.values()))
                    if tot > 0.5:
                        for k in rgrads:
                            rgrads[k] *= 0.5 / (tot + 1e-8)
                    rp = {"R1": R1, "R2": R2}
                    r_opt.step(rp, rgrads)
                    R1, R2 = rp["R1"], rp["R2"]
                v, vcache = critic.forward(mo)
                dv = 2.0 * (v[:, 0] - mret)[:, None] / len(mb)
                cgrads = critic.backward(dv, vcache)
                tot = np.sqrt(sum((g ** 2).sum() for g in cgrads.values()))
                if tot > 0.5:
                    for k in cgrads:
                        cgrads[k] *= 0.5 / (tot + 1e-8)
                critic_opt.step(critic.params(), cgrads)

        pack = {"champion": flat, "R1": R1.tolist(), "R2": R2.tolist()}
        json.dump(pack, open(f"/workspace/t4_ppo6_u{u}.json", "w"))
        res = eval_all_res(pack, f"t4-ppo6 u{u}")
        mean = float(np.mean(list(res.values())))
        floors_ok = (res[("goto", 0)] >= 0.9 and res[("hover_hold", 0)] >= 0.6
                     and res[("land", 0)] >= 0.7)
        floor_fails = 0 if floors_ok else floor_fails + 1
        if floors_ok and mean > best_mean:
            best_mean = mean
            json.dump(pack, open("/workspace/t4_ppo6_best.json", "w"))
        P.post_series("t4_ppo6_return", mean_ret, f"u{u}")
        P.post_series("t4_ppo6_face", mean_face, f"u{u}")
        if floor_fails > 8:
            print(f"T4PPO6 ABORT u{u}: floors failed 8 consecutive updates "
                  f"(precision collapse guard)", flush=True)
            break
        resid_norm = float(np.abs(R2).max())
        print(f"T4PPO6 u{u}: ret={mean_ret:.2f} face={mean_face:.3f} " +
              json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}) +
              f" best_mean={best_mean:.3f} |R2|max={resid_norm:.4f} wall={time.time()-t0:.0f}s", flush=True)
        P.post_status({
            "training": {"status": "running", "name": "t4_ppo6_residual", "iter": u, "iters": UPDATES,
                         "note": f"residual RL on frozen champion; goto_t0 {res[('goto',0)]:.2f} hover_t0 {res[('hover_hold',0)]:.2f} land_t0 {res[('land',0)]:.2f} mean {mean:.3f}"}})

    P.post_status({"training": {"status": "idle", "name": "t4_ppo6_residual",
                                "note": f"residual RL pilot finished; best mean {best_mean:.3f}"}})
    print(f"T4PPO6_DONE best_mean={best_mean:.3f}", flush=True)

if __name__ == "__main__":
    main()
