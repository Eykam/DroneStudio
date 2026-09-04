"""PPO T4.4: trainability-matched tiers. t4_ppo3 (PBRS) still lost hover:
60% of hover/land training episodes land on T1/T2 (obstacle scenes) where a
0.5-policy succeeds ~nowhere -> advantage is noise -> precision decays by
random walk. The champion learned precision on T0-only. Fix: hover/land
train on T0 (learnable), goto keeps the full tier mix.

(t4_ppo3 header) PPO T4.3: potential-based reward shaping (PBRS) for precision scenarios.

t4_ppo2 (gentle LR + curriculum) proved LR is not the wall: goto LEARNED
(goto_t2 0.31->0.56, t3 0.31->0.38) but hover still died 0.5->0 and land
never moved - sparse outcomes give hover/land zero gradient until the
policy can already hold/touch down. PBRS fixes that without reward hacking:
Phi(s) = -dist_to_target / success_radius; r_shape = W*(gamma*Phi(s') -
Phi(s)) telescopes to a bounded potential difference (policy-invariant in
the limit), pays approach, cannot be farmed by drifting. W = 2.0 (a 10m
approach pays ~+20 total vs +50 success). Applied to all scenarios.

(t4_ppo2 header) PPO T4.2: the optimizer-aggression test. Full-net HID-64 like t4_ppo, but:
(a) LR 3e-5 (10x gentler - t4_ppo at 3e-4 collapsed the DAgger warm start's
    precision behaviors by u80 despite ret staying ~+10: the objective pays
    safe-drift-to-timeout over hold-attempts once holding is hard),
(b) scenario curriculum instead of land-heavy-from-step-0: goto+hover only
    until u40, land phased in u41-60, land-heavy guard mix after u60.
Same init (t4_dag_r6). If precision survives here, the retention wall is
optimizer aggression on the tiered mix, not capacity.

(t4_ppo header) PPO T4: full-net PPO on the fresh HID-64 unified bootstrap.

Fresh-bootstrap track (user decision 2026-09-04): init from the t4 DAgger
policy (t4_dag_r6.json), EVERY parameter trains - no champion warm start,
no frozen trunk (the frozen trunk was the HID-32 retention-wall workaround;
HID-64 is the user's fix, this run tests it).

Recipe carried from the v3.x campaign (the parts that earned their keep):
- outcome-dominant shaping +50 / alive 0.02 / -10 / -10, gamma 0.999
- per-scenario advantage normalization (v3.4)
- T3 soft phase curriculum (v3.9, no hard switches), full tier mix 40/30/20/10
- land-heavy scenario mix (hover/land forgetting guard, ppo_v2)
- LOG_STD_INIT -3.0 (precision policy), log_std trains (full-net PPO)
- critic warmup: actor frozen for the first CRITIC_WARMUP updates
- best-with-floors: goto_t0 >= 0.9 AND hover_hold_t0 >= 0.6

Checkpoints /workspace/t4_ppo_uN.json, best -> /workspace/t4_ppo_best.json.
Series "t4-ppo uN" per cell + t4_ppo_return. Champion file never touched.
"""
import sys, json, os, time
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scene_schema import SceneDistribution
from env_sim import make_sim_factory
from ppo import MLP, Adam, GaussianPolicy
from scenario_sampler import sample_spec, sample_tier, tier_dist
import t4_common as P

MANIFEST = P.MANIFEST
OBS_DIM, ACT_DIM, HID, V2_DIM = P.OBS_DIM, P.ACT_DIM, P.HID, P.V2_DIM
WP_IN, WP_HID = P.WP_IN, P.WP_HID
LR, CLIP, GAMMA, LAM = 3e-5, 0.2, 0.999, 0.95
EPOCHS, MINIBATCH = 4, 2048
UPDATES, EPISODES_PER_UPDATE = 80, 32
PHASES = [{"lat": 0.05, "dens": 0.0,  "corr": 4.0},   # A
          {"lat": 0.10, "dens": 0.02, "corr": 4.0},   # B
          {"lat": 0.20, "dens": 0.05, "corr": 4.0},   # C
          {"lat": 0.35, "dens": 0.10, "corr": 2.0}]   # D: full T3
def phase_weights(u):
    if u <= 10: return [1.0, 0.0, 0.0, 0.0]
    if u <= 30: return [0.6, 0.4, 0.0, 0.0]
    if u <= 50: return [0.3, 0.5, 0.2, 0.0]
    if u <= 65: return [0.0, 0.3, 0.5, 0.2]
    return [0.0, 0.0, 0.4, 0.6]
CRITIC_WARMUP = 5
LOG_STD_INIT = -3.0
SUCCESS_BONUS, CRASH_PENALTY, TIMEOUT_PENALTY = 50.0, 10.0, 10.0
ALIVE_BONUS = 0.02
# t4.2: scenario curriculum - land phased in only after hover stabilizes
def scen_weights(u):
    if u <= 40: return {"goto": 0.5, "hover_hold": 0.5, "land": 0.0}
    if u <= 60: return {"goto": 0.375, "hover_hold": 0.375, "land": 0.25}
    return {"goto": 0.25, "hover_hold": 0.25, "land": 0.5}

def scene_for(seed, scenario, knobs):
    tier = sample_tier(seed)
    if scenario != "goto":
        tier = 0   # precision scenarios train where they are learnable (t4.4)
    if tier == 3:
        scenario = "goto"   # waypoints live on nav legs
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

def rollout_one(actor_flat, log_std, critic_flat, seed, scenario, knobs):
    rng = np.random.default_rng(np.uint64(seed))
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM); actor.load(P.bc_to_actor_params(actor_flat))
    wp1, wp2 = P.unpack_wp(actor_flat)
    critic = MLP(rng, OBS_DIM, HID, 1)
    keys = ("w1", "b1", "w2", "b2", "w3", "b3")
    p, i = {}, 0
    for k in keys:
        ref = getattr(critic, k)
        n = ref.size
        p[k] = np.array(critic_flat[i:i + n], dtype=np.float64).reshape(ref.shape)
        i += n
    critic.load(p)
    std = np.exp(np.array(log_std, dtype=np.float64))
    dist, spec, tier = scene_for(seed, scenario, knobs)
    max_steps = 600 if tier == 3 else (400 if spec["scenario"] == "goto" else 700)
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    assert len(obs) == OBS_DIM, f"obs dim {len(obs)} != {OBS_DIM}"
    ext = float(dist.scene_extent)
    inv_r = 1.0 / max(0.05, float(spec["success_radius"]))
    phi_prev = -float(np.linalg.norm((obs[0:3] + obs[19:22]) * ext)) * inv_r
    obs_b, act_b, rew_b, val_b, logp_b, done_b = [], [], [], [], [], []
    done = False
    prev_wp = 0
    while not done:
        mu, _ = actor.forward(obs[None, :])
        wpmu, _ = P.wp_forward(obs[None, :], wp1, wp2)
        mu = mu[0] + wpmu[0]
        a = mu + std * rng.standard_normal(ACT_DIM)
        logp = float(-0.5 * (((a - mu) / std) ** 2).sum()
                     - np.log(std).sum() - 0.5 * ACT_DIM * np.log(2 * np.pi))
        v, _ = critic.forward(obs[None, :])
        nxt, r, done = env.step(np.tanh(a))
        wp_now = int(env.last_info.get("wp", prev_wp))
        wp_passes = max(0, wp_now - prev_wp)
        prev_wp = wp_now
        r = ALIVE_BONUS + 10.0 * wp_passes
        phi = -float(np.linalg.norm((nxt[0:3] + nxt[19:22]) * ext)) * inv_r
        r += 2.0 * (GAMMA * phi - phi_prev)   # PBRS: bounded, cannot be farmed
        phi_prev = phi
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
    return obs_b, act_b, rew_b, val_b, logp_b, done_b, spec["scenario"]

def critic_flat_of(critic):
    p = critic.params()
    return list(np.concatenate([p[k].ravel() for k in ("w1", "b1", "w2", "b2", "w3", "b3")]))

def main():
    rng = np.random.default_rng(7)
    init = json.load(open("/workspace/t4_dag_r6.json"))
    policy = GaussianPolicy(rng, OBS_DIM, ACT_DIM, HID, LOG_STD_INIT)
    policy.actor.load(P.bc_to_actor_params(init))
    WP1, WP2 = P.unpack_wp(init)          # trained DAgger values, keep training them
    wp_opt = Adam({"wp1": WP1, "wp2": WP2}, lr=LR)
    critic = MLP(rng, OBS_DIM, HID, 1)
    actor_opt = Adam({**policy.actor.params(), "log_std": policy.log_std}, lr=LR)
    critic_opt = Adam(critic.params(), lr=1e-3)

    cur_flat = P.pack_actor(policy.actor, WP1, WP2)
    res0 = P.eval_all(cur_flat, "t4-ppo4 u0")
    print("T4PPO4 u0 (t4_dag_r6 init): " +
          json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res0.items()}), flush=True)
    best_mean = -1e9
    best_flat = cur_flat
    json.dump(best_flat, open("/workspace/t4_ppo4_u0.json", "w"))

    for u in range(1, UPDATES + 1):
        t0 = time.time()
        a_flat = P.pack_actor(policy.actor, WP1, WP2)
        c_flat = critic_flat_of(critic)
        ls = list(policy.log_std)
        wts = phase_weights(u)
        sw = scen_weights(u)
        pool, pw = list(sw.keys()), [sw[k] for k in sw]
        scs = [pool[int(rng.choice(len(pool), p=pw))] for _ in range(EPISODES_PER_UPDATE)]
        args = [(a_flat, ls, c_flat, int(rng.integers(0, 2**31 - 1)), sc,
                 PHASES[int(rng.choice(4, p=wts))])
                for sc in scs]
        eps = P.parallel_episodes(rollout_one, args)

        obs_b, act_b, adv_b, ret_b, logp_b = [], [], [], [], []
        ep_scen = [e[6] for e in eps]
        for ob, ab, rb, vb, lb, db, _sc in eps:
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
        # per-scenario advantage normalization (v3.4)
        scen_arr = np.array([sc for sc, e in zip(ep_scen, eps) for _ in e[0]])
        for sc in set(ep_scen):
            m = scen_arr == sc
            adv_b[m] = (adv_b[m] - adv_b[m].mean()) / (adv_b[m].std() + 1e-8)
        mean_ret = float(np.mean([sum(e[2]) for e in eps]))

        n = len(adv_b); idx = np.arange(n)
        for _ in range(EPOCHS):
            rng.shuffle(idx)
            for start in range(0, n, MINIBATCH):
                mb = idx[start:start + MINIBATCH]
                mo, ma = obs_b[mb], act_b[mb]
                madv, mret, mlogp = adv_b[mb], ret_b[mb], logp_b[mb]
                mub, cache = policy.actor.forward(mo)
                wpmu, hwp = P.wp_forward(mo, WP1, WP2)
                mu = mub + wpmu
                std = np.exp(policy.log_std)
                logp = (-0.5 * (((ma - mu) / std) ** 2).sum(-1)
                        - policy.log_std.sum() - 0.5 * ACT_DIM * np.log(2 * np.pi))
                ratio = np.exp(logp - mlogp)
                coeff = -madv * np.where(
                    ((madv >= 0) & (ratio < 1 + CLIP)) | ((madv < 0) & (ratio > 1 - CLIP)),
                    ratio, 0.0) / len(mb)
                if u > CRITIC_WARMUP:  # actor frozen during critic warmup
                    dmu = coeff[:, None] * (ma - mu) / (std ** 2)[None, :]
                    agrads = policy.actor.backward(dmu, cache)   # FULL net trains
                    dls = (coeff[:, None] * (((ma - mu) / std) ** 2 - 1.0)).sum(0)
                    agrads["log_std"] = dls
                    allp = policy.actor.params(); allp["log_std"] = policy.log_std
                    tot = np.sqrt(sum((g ** 2).sum() for g in agrads.values()))
                    if tot > 0.5:
                        for k in agrads:
                            agrads[k] *= 0.5 / (tot + 1e-8)
                    actor_opt.step(allp, agrads)
                    policy.log_std = allp["log_std"]
                    gwp2 = hwp.T @ dmu
                    dh = (dmu @ WP2.T) * (hwp > 0)
                    gwp1 = mo[:, V2_DIM:].T @ dh
                    wgrads = {"wp1": gwp1, "wp2": gwp2}
                    tot = np.sqrt(sum((g ** 2).sum() for g in wgrads.values()))
                    if tot > 0.5:
                        for k in wgrads:
                            wgrads[k] *= 0.5 / (tot + 1e-8)
                    wpp = {"wp1": WP1, "wp2": WP2}
                    wp_opt.step(wpp, wgrads)
                    WP1, WP2 = wpp["wp1"], wpp["wp2"]
                v, vcache = critic.forward(mo)
                dv = 2.0 * (v[:, 0] - mret)[:, None] / len(mb)
                cgrads = critic.backward(dv, vcache)
                tot = np.sqrt(sum((g ** 2).sum() for g in cgrads.values()))
                if tot > 0.5:
                    for k in cgrads:
                        cgrads[k] *= 0.5 / (tot + 1e-8)
                critic_opt.step(critic.params(), cgrads)

        cur_flat = P.pack_actor(policy.actor, WP1, WP2)
        json.dump(cur_flat, open(f"/workspace/t4_ppo4_u{u}.json", "w"))
        res = P.eval_all(cur_flat, f"t4-ppo4 u{u}")
        mean = float(np.mean(list(res.values())))
        floors_ok = res[("goto", 0)] >= 0.9 and res[("hover_hold", 0)] >= 0.6
        if floors_ok and mean > best_mean:
            best_mean = mean
            best_flat = cur_flat
            json.dump(best_flat, open("/workspace/t4_ppo4_best.json", "w"))
        P.post_series("t4_ppo4_return", mean_ret, f"u{u}")
        print(f"T4PPO4 u{u}: ret={mean_ret:.2f} " +
              json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}) +
              f" best_mean={best_mean:.3f} log_std={np.mean(policy.log_std):.3f} wall={time.time()-t0:.0f}s", flush=True)
        P.post_status({
            "training": {"status": "running", "name": "t4_ppo4", "iter": u, "iters": UPDATES,
                         "note": f"T4 HID-64 full-net; goto_t0 {res[('goto',0)]:.2f} hover_t0 {res[('hover_hold',0)]:.2f} land_t0 {res[('land',0)]:.2f} goto_t3 {res[('goto',3)]:.2f}"},
            "queue": ["promotion flip of bc_ppo_v2_best: awaiting user go", "ee-flight SB1 in flight on box3"],
        })

    json.dump(best_flat, open("/workspace/t4_ppo4_best.json", "w"))
    P.post_status({"training": {"status": "idle", "name": "t4_ppo4",
                                "note": f"T4 HID-64 PPO finished {UPDATES} updates; best mean {best_mean:.3f}"}})
    print(f"T4PPO4_DONE best_mean={best_mean:.3f}", flush=True)

if __name__ == "__main__":
    main()
