"""PPO v3.9: soft T3 curriculum + dedicated waypoint pathway.

v3.7/v3.8 showed the frozen probe learns clear slalom (+11.2 ret) but
collapses at hard phase transitions. v3.9: (1) per-episode phase mixture
(no hard curriculum switch), (2) capacity via a bias-free wp-MLP
(7->32->4) added to mu: wp channels are exactly 0 on T0-T2 (obs v3.1
delta encoding) so the pathway is bit-exact-safe there forever, and
zero-init output layer keeps u0 exactly the champion on T3 too.

(original v3 header) PPO v3: tiered-scene training on obs v3 (26-dim), Phase 1+2 combined.

Warm-starts the actor from bc_ppo_v2_best.json (19-dim obs v2 champion):
obs v3's first 19 dims are exactly obs v2 (same order/scales), so the actor
loads with W1 zero-padded for the 7 new waypoint dims - an exact function
preserving start. Critic is fresh (26-dim).

Scenes: per-episode tier via scenario_sampler.sample_tier at the approved
40/30/20/10 mix (T0 open / T1 clutter / T2 gap-walls / T3 waypoint slalom).
T3 forces goto (waypoints live on nav legs); waypoints are zeroed for
hover/land specs. Outcome-dominant shaping unchanged from ppo_v2.

Eval: fixed held-out cells per (scenario, tier) - heldout_cells_tiered()
(16 seeds each; T3 cells at base+3000). Series success_<sc>_t<t> labeled
"ppo-v3 uN" - same series as the dagger_v3 baselines, different label.
Dense scenes are partially observable (nearest-obstacle vector only);
per-tier numbers are levels, not regressions (dashboard caveat stands).

Checkpoints: /workspace/bc_ppo_v3_uN.json every update; best-with-floors
(floors on tier-0 goto/hover) -> /workspace/bc_ppo_v39_best.json. Shares NO
state with prior tracks (never writes bc_flat*.json or bc_ppo_v2*.json).

Run:  nohup /workspace/venv/bin/python ppo_v3.py > /workspace/ppo_v3.log
"""
import sys, json, os, time
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scene_schema import SceneDistribution
from env_sim import make_sim_factory
from policy import MLP as BCMlp
from ppo import MLP, Adam, GaussianPolicy
from scenario_sampler import sample_spec, sample_tier, tier_dist, heldout_cells_tiered
from eval_scenarios import post_series
from parallel_rollout import parallel_episodes

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/v14_g13.manifest.json"
OBS_DIM, ACT_DIM, HID = 26, 4, 32
V2_DIM = 19
WP_IN, WP_HID = 7, 32  # v3.9 waypoint pathway (bias-free: zero input -> zero output)

def wp_forward(obs, wp1, wp2):
    h = np.maximum(obs[:, V2_DIM:] @ wp1, 0.0)  # no bias: wp=0 -> h=0 -> out=0
    return h @ wp2, h

def pack_actor(policy, wp1, wp2):
    return actor_to_bc_flat(policy.actor) + list(wp1.ravel()) + list(wp2.ravel())

def unpack_wp(actor_flat):
    n2 = WP_HID * ACT_DIM; n1 = WP_IN * WP_HID
    wp2 = np.array(actor_flat[-n2:], dtype=np.float64).reshape(WP_HID, ACT_DIM)
    wp1 = np.array(actor_flat[-n2 - n1:-n2], dtype=np.float64).reshape(WP_IN, WP_HID)
    return wp1, wp2
LR, CLIP, GAMMA, LAM = 3e-4, 0.2, 0.999, 0.95  # higher LR OK: only 224 probe params train
EPOCHS, MINIBATCH = 4, 2048
UPDATES, EPISODES_PER_UPDATE = 80, 32
# v3.9: soft curriculum - per-episode phase mixture, no hard switch
# (v3.7/v3.8 collapsed exactly at the phase transitions).
PHASES = [{"lat": 0.05, "dens": 0.0,  "corr": 4.0},   # A: clear gentle slalom
          {"lat": 0.10, "dens": 0.02, "corr": 4.0},   # B
          {"lat": 0.20, "dens": 0.05, "corr": 4.0},   # C
          {"lat": 0.35, "dens": 0.10, "corr": 2.0}]   # D: full T3
def phase_weights(u):
    if u <= 10: return [1.0, 0.0, 0.0, 0.0]
    if u <= 30: return [0.6, 0.4, 0.0, 0.0]
    if u <= 50: return [0.3, 0.5, 0.2, 0.0]
    if u <= 65: return [0.0, 0.3, 0.5, 0.2]
    return [0.0, 0.0, 0.4, 0.6]
CRITIC_WARMUP = 5  # updates 1..10: frozen actor+log_std, critic-only - the
# fresh 26-dim critic calibrates on the warm-start policy's own state
# distribution before any gradient touches the actor
LOG_STD_INIT = -3.0          # std 0.05: precision policy; bigger noise crashes instantly
# Outcome-dominant shaping (learning: episode-return fitness prefers dying).
SUCCESS_BONUS, CRASH_PENALTY, TIMEOUT_PENALTY = 50.0, 10.0, 10.0
ALIVE_BONUS = 0.02

DASHBOARD = os.environ.get("DASHBOARD_URL", "").rstrip("/")
TOKEN = os.environ.get("INGEST_TOKEN", "")

def post_status(doc):
    if not (DASHBOARD and TOKEN):
        return
    try:
        import urllib.request
        req = urllib.request.Request(DASHBOARD + "/api/training/status",
            data=json.dumps(doc).encode(),
            headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass

MIX = ("land", "land", "land", "hover_hold", "goto", "goto")  # ppo_v2's land-heavy mix: hover/land forgetting guard

def scene_for(seed, scenario, knobs, rng):
    """v3.7: T3-only goto on the phase curriculum - tier-3 base (waypoint
    count, goal distance) with offset/density/corridor phased in."""
    from scene_schema import SceneDistribution
    tier = 3
    scenario = "goto"
    base = tier_dist(seed, 3)
    dist = SceneDistribution.from_vector(base.to_vector())
    dist.waypoint_lat = knobs["lat"]
    dist.obstacle_density = knobs["dens"]
    dist.corridor_width = knobs["corr"]
    spec = sample_spec(seed, force_scenario="goto")
    return dist, spec, tier

def warm_start_params():
    """bc_ppo_v2_best (19-dim) -> 26-dim actor params, W1 zero-padded."""
    flat = json.load(open("/workspace/bc_ppo_v2_best.json"))
    bc = BCMlp(V2_DIM, ACT_DIM, hidden=HID, seed=0)
    bc.set_flat(np.array(flat, dtype=np.float64))
    w1 = np.zeros((V2_DIM + 7, HID))
    w1[:V2_DIM] = bc.W1
    return {"w1": w1, "b1": bc.b1.copy(), "w2": bc.W2.copy(),
            "b2": bc.b2.copy(), "w3": bc.W3.copy(), "b3": bc.b3.copy()}

def bc_to_actor_params(flat):
    bc = BCMlp(OBS_DIM, ACT_DIM, hidden=HID, seed=0)
    bc.set_flat(np.array(flat, dtype=np.float64))
    return {"w1": bc.W1.copy(), "b1": bc.b1.copy(), "w2": bc.W2.copy(),
            "b2": bc.b2.copy(), "w3": bc.W3.copy(), "b3": bc.b3.copy()}

def actor_to_bc_flat(actor):
    p = actor.params()
    return list(np.concatenate([p[k].ravel() for k in ("w1", "b1", "w2", "b2", "w3", "b3")]))

def rollout_one(actor_flat, log_std, critic_flat, seed, scenario, knobs):
    rng = np.random.default_rng(np.uint64(seed))
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM); actor.load(bc_to_actor_params(actor_flat))
    wp1, wp2 = unpack_wp(actor_flat)
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
    dist, spec, tier = scene_for(seed, scenario, knobs, rng)
    max_steps = 600 if tier == 3 else (400 if spec["scenario"] == "goto" else 700)
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    assert len(obs) == OBS_DIM, f"obs dim {len(obs)} != {OBS_DIM}"
    obs_b, act_b, rew_b, val_b, logp_b, done_b = [], [], [], [], [], []
    done = False
    prev_wp = 0
    while not done:
        mu, _ = actor.forward(obs[None, :])
        wpmu, _ = wp_forward(obs[None, :], wp1, wp2)
        mu = mu[0] + wpmu[0]
        a = mu + std * rng.standard_normal(ACT_DIM)
        logp = float(-0.5 * (((a - mu) / std) ** 2).sum()
                     - np.log(std).sum() - 0.5 * ACT_DIM * np.log(2 * np.pi))
        v, _ = critic.forward(obs[None, :])
        nxt, r, done = env.step(np.tanh(a))
        wp_now = int(env.last_info.get("wp", prev_wp))
        wp_passes = max(0, wp_now - prev_wp)
        prev_wp = wp_now
        # outcome-dominant + per-waypoint-pass signal (the only graded
        # progress signal T3 has; sim per-step progress reward stays ignored)
        r = ALIVE_BONUS + 10.0 * wp_passes
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

def eval_one(actor_flat, scenario, tier, seed):
    rng = np.random.default_rng(0)
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM); actor.load(bc_to_actor_params(actor_flat))
    wp1, wp2 = unpack_wp(actor_flat)
    dist = tier_dist(seed, tier)
    spec = sample_spec(seed, force_scenario=scenario)
    if scenario != "goto":
        dist.n_waypoints = 0.0
    max_steps = 600 if tier == 3 else (400 if scenario == "goto" else 700)
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    done = False
    while not done:
        mu, _ = actor.forward(obs[None, :])
        wpmu, _ = wp_forward(obs[None, :], wp1, wp2)
        obs, r, done = env.step(np.tanh(mu[0] + wpmu[0]))
    ok = bool(env.succeeded)
    env.close()
    return ok

def critic_flat_of(critic):
    p = critic.params()
    return list(np.concatenate([p[k].ravel() for k in ("w1", "b1", "w2", "b2", "w3", "b3")]))

def main():
    rng = np.random.default_rng(7)
    policy = GaussianPolicy(rng, OBS_DIM, ACT_DIM, HID, LOG_STD_INIT)
    policy.actor.load(warm_start_params())
    WP1 = rng.normal(0.0, 0.1, (WP_IN, WP_HID))   # random for symmetry breaking;
    WP2 = np.zeros((WP_HID, ACT_DIM))             # zero output: u0 == champion
    wp_opt = Adam({"wp1": WP1, "wp2": WP2}, lr=LR)
    critic = MLP(rng, OBS_DIM, HID, 1)
    actor_opt = Adam({**policy.actor.params(), "log_std": policy.log_std}, lr=LR)
    critic_opt = Adam(critic.params(), lr=1e-3)
    cells = heldout_cells_tiered()  # {(scenario, tier): [seeds]}
    cell_keys = sorted(cells.keys())

    def eval_all(flat):
        args = [(flat, sc, t, s) for (sc, t) in cell_keys for s in cells[(sc, t)]]
        res = parallel_episodes(eval_one, args)
        out, i = {}, 0
        for k in cell_keys:
            n = len(cells[k])
            out[k] = float(np.mean(res[i:i + n]))
            i += n
        return out

    cur_flat = pack_actor(policy, WP1, WP2)
    res0 = eval_all(cur_flat)
    print("PPOV39 u0 (zero-pad warm start from bc_ppo_v2_best): " +
          json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res0.items()}), flush=True)
    best_mean = float(np.mean(list(res0.values())))
    best_flat = cur_flat
    json.dump(best_flat, open("/workspace/bc_ppo_v39_u0.json", "w"))

    for u in range(1, UPDATES + 1):
        t0 = time.time()
        a_flat = pack_actor(policy, WP1, WP2)
        c_flat = critic_flat_of(critic)
        ls = list(policy.log_std)
        wts = phase_weights(u)
        scs = [MIX[int(rng.integers(0, len(MIX)))] for _ in range(EPISODES_PER_UPDATE)]
        args = [(a_flat, ls, c_flat, int(rng.integers(0, 2**31 - 1)), sc,
                 PHASES[int(rng.choice(4, p=wts))])
                for sc in scs]
        eps = parallel_episodes(rollout_one, args)

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
        # per-scenario advantage normalization (v3.4): mixed-scenario batches
        # share one outcome scale (+50/-10) but wildly different episode
        # lengths, so global normalization let long hover/land episodes
        # dominate the gradient and destabilized precision flight
        adv_b = np.array(adv_b)
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
                wpmu, hwp = wp_forward(mo, WP1, WP2)
                mu = mub + wpmu
                std = np.exp(policy.log_std)
                logp = (-0.5 * (((ma - mu) / std) ** 2).sum(-1)
                        - policy.log_std.sum() - 0.5 * ACT_DIM * np.log(2 * np.pi))
                ratio = np.exp(logp - mlogp)
                # d loss / d mu (minimize -w * advantage == maximize surrogate)
                coeff = -madv * np.where(
                    ((madv >= 0) & (ratio < 1 + CLIP)) | ((madv < 0) & (ratio > 1 - CLIP)),
                    ratio, 0.0) / len(mb)
                if u > CRITIC_WARMUP:  # actor frozen during critic warm-up
                    dmu = coeff[:, None] * (ma - mu) / (std ** 2)[None, :]
                    agrads = policy.actor.backward(dmu, cache)
                    # FROZEN TRUNK (v3.5): only w1 rows 19:26 (the 7 obs-v3
                    # waypoint channels) may move. v2 behavior is bit-exact
                    # preserved on T0-T2 because those channels read exact
                    # zeros there (obs v3.1 delta encoding).
                    agrads["w1"][:V2_DIM] = 0.0
                    for fk in ("b1", "w2", "b2", "w3", "b3"):
                        agrads[fk][...] = 0.0
                    agrads["log_std"] = np.zeros_like(policy.log_std)
                    allp = policy.actor.params(); allp["log_std"] = policy.log_std
                    tot = np.sqrt(sum((g ** 2).sum() for g in agrads.values()))
                    if tot > 0.5:
                        for k in agrads:
                            agrads[k] *= 0.5 / (tot + 1e-8)
                    actor_opt.step(allp, agrads)
                    policy.log_std = allp["log_std"]
                    # waypoint pathway grads (mu = trunk + wp path, dmu shared)
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

        cur_flat = pack_actor(policy, WP1, WP2)
        json.dump(cur_flat, open(f"/workspace/bc_ppo_v39_u{u}.json", "w"))
        res = eval_all(cur_flat)
        mean = float(np.mean(list(res.values())))
        floors_ok = res[("goto", 0)] >= 0.9 and res[("hover_hold", 0)] >= 0.6
        if mean > best_mean and floors_ok:
            best_mean = mean
            best_flat = cur_flat
            json.dump(best_flat, open("/workspace/bc_ppo_v39_best.json", "w"))
        post_series("ppo_v39_return", mean_ret, f"u{u}")
        for (sc, t), v in res.items():
            post_series(f"success_{sc}_t{t}", v, f"ppo-v3.9 u{u}")
        print(f"PPOV39 u{u}: ret={mean_ret:.2f} " +
              json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}) +
              f" best_mean={best_mean:.3f} wall={time.time()-t0:.0f}s", flush=True)
        post_status({
            "live_policy": {"name": "bc_flat.json",
                            "detail": "v1 policy - 15-dim obs, chassis_v1 dynamics",
                            "note": "flying on /watch; promotion candidate below"},
            "candidate": {"name": "bc_ppo_v2_best.json",
                          "detail": "obs v2 (19-dim) - v14-g13 + com dynamics; promotion awaiting user go"},
            "training": {"status": "running", "name": "ppo_v3.6", "iter": u, "iters": UPDATES,
                         "note": f"obs v3 tiered; goto_t0 {res[('goto',0)]:.2f} hover_t0 {res[('hover_hold',0)]:.2f} land_t0 {res[('land',0)]:.2f} goto_t3 {res[('goto',3)]:.2f}"},
            "queue": ["promotion flip of bc_ppo_v2_best: awaiting user go", "EE box #3 provisioning"],
        })

    json.dump(best_flat, open("/workspace/bc_ppo_v39_best.json", "w"))
    post_status({
        "training": {"status": "idle", "name": "ppo_v3.6",
                     "note": f"finished {UPDATES} updates; best mean {best_mean:.3f}"},
        "queue": ["promotion flip of bc_ppo_v2_best: awaiting user go", "EE box #3 provisioning"],
    })
    print(f"PPOV39_DONE best_mean={best_mean:.3f}", flush=True)

if __name__ == "__main__":
    main()
