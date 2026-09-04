"""PPO v3: tiered-scene training on obs v3 (26-dim), Phase 1+2 combined.

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
(floors on tier-0 goto/hover) -> /workspace/bc_ppo_v32_best.json. Shares NO
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
LR, CLIP, GAMMA, LAM = 1e-4, 0.2, 0.999, 0.95
EPOCHS, MINIBATCH = 4, 2048
UPDATES, EPISODES_PER_UPDATE = 50, 32
CRITIC_WARMUP = 10  # updates 1..10: frozen actor+log_std, critic-only - the
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

def scene_for(seed, scenario, t3_boost, rng):
    """(dist, spec, tier) for an episode: explicit scenario (MIX-weighted by
    the caller) crossed with tier sampling. T3 forces goto (waypoints live on
    nav legs); t3_boost flat-oversamples T3 to 20% early so the zero-init
    waypoint channels see gradient."""
    tier = 3 if (t3_boost and rng.random() < 0.2) else sample_tier(seed)
    if tier == 3:
        scenario = "goto"
    dist = tier_dist(seed, tier)
    spec = sample_spec(seed, force_scenario=scenario)
    if spec["scenario"] != "goto":
        dist.n_waypoints = 0.0  # waypoints live on goto nav legs only
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

def rollout_one(actor_flat, log_std, critic_flat, seed, scenario, t3_boost):
    rng = np.random.default_rng(np.uint64(seed))
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM); actor.load(bc_to_actor_params(actor_flat))
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
    dist, spec, tier = scene_for(seed, scenario, bool(t3_boost), rng)
    max_steps = 600 if tier == 3 else (400 if spec["scenario"] == "goto" else 700)
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    assert len(obs) == OBS_DIM, f"obs dim {len(obs)} != {OBS_DIM}"
    obs_b, act_b, rew_b, val_b, logp_b, done_b = [], [], [], [], [], []
    done = False
    while not done:
        mu, _ = actor.forward(obs[None, :])
        mu = mu[0]
        a = mu + std * rng.standard_normal(ACT_DIM)
        logp = float(-0.5 * (((a - mu) / std) ** 2).sum()
                     - np.log(std).sum() - 0.5 * ACT_DIM * np.log(2 * np.pi))
        v, _ = critic.forward(obs[None, :])
        nxt, r, done = env.step(np.tanh(a))
        r = ALIVE_BONUS  # ignore sim per-step reward: sign conventions differ per scenario
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
    return obs_b, act_b, rew_b, val_b, logp_b, done_b

def eval_one(actor_flat, scenario, tier, seed):
    rng = np.random.default_rng(0)
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM); actor.load(bc_to_actor_params(actor_flat))
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
        obs, r, done = env.step(np.tanh(mu[0]))
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

    cur_flat = actor_to_bc_flat(policy.actor)
    res0 = eval_all(cur_flat)
    print("PPOV32 u0 (zero-pad warm start from bc_ppo_v2_best): " +
          json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res0.items()}), flush=True)
    best_mean = float(np.mean(list(res0.values())))
    best_flat = cur_flat
    json.dump(best_flat, open("/workspace/bc_ppo_v32_u0.json", "w"))

    for u in range(1, UPDATES + 1):
        t0 = time.time()
        a_flat = actor_to_bc_flat(policy.actor)
        c_flat = critic_flat_of(critic)
        ls = list(policy.log_std)
        t3_boost = u <= CRITIC_WARMUP + 10
        scs = [MIX[int(rng.integers(0, len(MIX)))] for _ in range(EPISODES_PER_UPDATE)]
        args = [(a_flat, ls, c_flat, int(rng.integers(0, 2**31 - 1)), sc, t3_boost)
                for sc in scs]
        eps = parallel_episodes(rollout_one, args)

        obs_b, act_b, adv_b, ret_b, logp_b = [], [], [], [], []
        for ob, ab, rb, vb, lb, db in eps:
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
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
        mean_ret = float(np.mean([sum(e[2]) for e in eps]))

        n = len(adv_b); idx = np.arange(n)
        for _ in range(EPOCHS):
            rng.shuffle(idx)
            for start in range(0, n, MINIBATCH):
                mb = idx[start:start + MINIBATCH]
                mo, ma = obs_b[mb], act_b[mb]
                madv, mret, mlogp = adv_b[mb], ret_b[mb], logp_b[mb]
                mu, cache = policy.actor.forward(mo)
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
                    agrads["log_std"] = (coeff[:, None] * ((((ma - mu) ** 2) / (std ** 2)[None, :]) - 1.0)).sum(axis=0)
                    allp = policy.actor.params(); allp["log_std"] = policy.log_std
                    tot = np.sqrt(sum((g ** 2).sum() for g in agrads.values()))
                    if tot > 0.5:
                        for k in agrads:
                            agrads[k] *= 0.5 / (tot + 1e-8)
                    actor_opt.step(allp, agrads)
                    policy.log_std = allp["log_std"]
                v, vcache = critic.forward(mo)
                dv = 2.0 * (v[:, 0] - mret)[:, None] / len(mb)
                cgrads = critic.backward(dv, vcache)
                tot = np.sqrt(sum((g ** 2).sum() for g in cgrads.values()))
                if tot > 0.5:
                    for k in cgrads:
                        cgrads[k] *= 0.5 / (tot + 1e-8)
                critic_opt.step(critic.params(), cgrads)

        cur_flat = actor_to_bc_flat(policy.actor)
        json.dump(cur_flat, open(f"/workspace/bc_ppo_v32_u{u}.json", "w"))
        res = eval_all(cur_flat)
        mean = float(np.mean(list(res.values())))
        floors_ok = res[("goto", 0)] >= 0.9 and res[("hover_hold", 0)] >= 0.6
        if mean > best_mean and floors_ok:
            best_mean = mean
            best_flat = cur_flat
            json.dump(best_flat, open("/workspace/bc_ppo_v32_best.json", "w"))
        post_series("ppo_v32_return", mean_ret, f"u{u}")
        for (sc, t), v in res.items():
            post_series(f"success_{sc}_t{t}", v, f"ppo-v3.2 u{u}")
        print(f"PPOV32 u{u}: ret={mean_ret:.2f} " +
              json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}) +
              f" best_mean={best_mean:.3f} wall={time.time()-t0:.0f}s", flush=True)
        post_status({
            "live_policy": {"name": "bc_flat.json",
                            "detail": "v1 policy - 15-dim obs, chassis_v1 dynamics",
                            "note": "flying on /watch; promotion candidate below"},
            "candidate": {"name": "bc_ppo_v2_best.json",
                          "detail": "obs v2 (19-dim) - v14-g13 + com dynamics; promotion awaiting user go"},
            "training": {"status": "running", "name": "ppo_v3.2", "iter": u, "iters": UPDATES,
                         "note": f"obs v3 tiered; goto_t0 {res[('goto',0)]:.2f} hover_t0 {res[('hover_hold',0)]:.2f} land_t0 {res[('land',0)]:.2f} goto_t3 {res[('goto',3)]:.2f}"},
            "queue": ["promotion flip of bc_ppo_v2_best: awaiting user go", "EE box #3 provisioning"],
        })

    json.dump(best_flat, open("/workspace/bc_ppo_v32_best.json", "w"))
    post_status({
        "training": {"status": "idle", "name": "ppo_v3.2",
                     "note": f"finished {UPDATES} updates; best mean {best_mean:.3f}"},
        "queue": ["promotion flip of bc_ppo_v2_best: awaiting user go", "EE box #3 provisioning"],
    })
    print(f"PPOV32_DONE best_mean={best_mean:.3f}", flush=True)

if __name__ == "__main__":
    main()
