"""PPO fine-tune of the obs v2 scenario policy (concurrent RL track).

Warm-starts the actor from bc_flat_v2.json (DAgger track output, 19-dim obs,
v14-g13 + com dynamics), then PPO over scenario-sampled PARALLEL rollouts
(parallel_rollout.parallel_episodes, fork pool). Architecture match: the BC
net is tanh(tanh(tanh(x@W1+b1)@W2+b2)@W3+b3); the PPO actor is the same MLP
with a LINEAR output (mu) - so BC action = tanh(mu) exactly, and loading the
BC weights into the actor + small initial log_std is a true warm start.

Rules honored: fixed held-out eval seed blocks (same as DAgger track), own
dashboard series (ppo_v2_return + success_* labels "ppo-v2 uN"), checkpoint
every update (bc_ppo_v2_uN.json), best-with-floors -> bc_ppo_v2_best.json.
Shares NO state with the DAgger track (never touches bc_flat_v2.json).

Run:  AUTORESEARCH_OBS_V2=1 nohup python ppo_v2.py > /workspace/ppo_v2.log
"""
import sys, json, os, time
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
from scene_schema import SceneDistribution
from env_sim import make_sim_factory
from policy import MLP as BCMlp
from ppo import MLP, Adam, GaussianPolicy
from scenario_sampler import sample_spec, heldout_cells
from eval_scenarios import cell_dist, post_series
from parallel_rollout import parallel_episodes

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/v14_g13.manifest.json"
OBS_DIM, ACT_DIM, HID = 19, 4, 32
LR, CLIP, GAMMA, LAM = 1e-4, 0.2, 0.999, 0.95
EPOCHS, MINIBATCH = 4, 2048
UPDATES, EPISODES_PER_UPDATE = 40, 32
LOG_STD_INIT = -3.0          # std 0.05: BC policy needs precision (land: 0.5m pad, 0.5 m/s touchdown); bigger noise crashes instantly
# Outcome-dominant shaping (learning: episode-return fitness prefers dying).
# Order must be success >> timeout > crash at every episode length:
#   success: 50 + 0.02*T | timeout: 0.02*T - 10 | crash: 0.02*t - 10
SUCCESS_BONUS, CRASH_PENALTY, TIMEOUT_PENALTY = 50.0, 10.0, 10.0
ALIVE_BONUS = 0.02
MIX = ("land", "land", "land", "hover_hold", "goto", "goto")  # match dagger_land

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

def train_dist(seed):
    r = np.random.default_rng(np.uint64(seed) ^ np.uint64(0xD1A9))
    gd = float(r.uniform(2.0, 25.0))
    dens = float(r.choice([0.0, 0.05, 0.1, 0.2]))
    return SceneDistribution(obstacle_density=dens, corridor_width=4.0,
        scene_extent=max(10.0, gd * 2), goal_distance=gd,
        light_direction_entropy=0.3, texture_variety=0.0, dynamics_noise=0.0)

def bc_to_actor_params(flat):
    bc = BCMlp(OBS_DIM, ACT_DIM, hidden=HID, seed=0)
    bc.set_flat(np.array(flat, dtype=np.float64))
    return {"w1": bc.W1.copy(), "b1": bc.b1.copy(), "w2": bc.W2.copy(),
            "b2": bc.b2.copy(), "w3": bc.W3.copy(), "b3": bc.b3.copy()}

def actor_to_bc_flat(actor):
    p = actor.params()
    return list(np.concatenate([p[k].ravel() for k in ("w1", "b1", "w2", "b2", "w3", "b3")]))

def rollout_one(actor_flat, log_std, critic_flat, seed, scenario):
    """One stochastic episode; returns per-step buffers (lists)."""
    rng = np.random.default_rng(np.uint64(seed))
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM); actor.load(bc_to_actor_params(actor_flat))
    critic = MLP(rng, OBS_DIM, HID, 1)  # shapes set by init; weights loaded from flat below
    return _rollout_body(actor, critic, actor_flat, critic_flat, log_std, rng, seed, scenario)

def _rollout_body(actor, critic, actor_flat, critic_flat, log_std, rng, seed, scenario):
    # reconstruct critic from flat (ordered w1,b1,w2,b2,w3,b3)
    def load_flat(net, flat):
        keys = ("w1", "b1", "w2", "b2", "w3", "b3")
        p, i = {}, 0
        for k in keys:
            ref = getattr(net, k)
            n = ref.size
            p[k] = np.array(flat[i:i + n], dtype=np.float64).reshape(ref.shape)
            i += n
        net.load(p)
    load_flat(critic, critic_flat)
    std = np.exp(np.array(log_std, dtype=np.float64))
    dist = train_dist(seed)
    spec = sample_spec(seed, force_scenario=scenario)
    max_steps = 400 if scenario == "goto" else 700
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
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

def eval_one(actor_flat, scenario, seed):
    rng = np.random.default_rng(0)
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM); actor.load(bc_to_actor_params(actor_flat))
    dist = cell_dist(seed)
    spec = sample_spec(seed, force_scenario=scenario)
    max_steps = 400 if scenario == "goto" else 700
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
    bc_flat = json.load(open("/workspace/bc_flat_v2.json"))
    policy = GaussianPolicy(rng, OBS_DIM, ACT_DIM, HID, LOG_STD_INIT)
    policy.actor.load(bc_to_actor_params(bc_flat))
    critic = MLP(rng, OBS_DIM, HID, 1)
    actor_opt = Adam({**policy.actor.params(), "log_std": policy.log_std}, lr=LR)
    critic_opt = Adam(critic.params(), lr=1e-3)
    cells = heldout_cells()

    def eval_all(flat):
        args = [(flat, sc, s) for sc, seeds in cells.items() for s in seeds]
        res = parallel_episodes(eval_one, args)
        out, i = {}, 0
        for sc, seeds in cells.items():
            out[sc] = float(np.mean(res[i:i + len(seeds)]))
            i += len(seeds)
        return out

    cur_flat = actor_to_bc_flat(policy.actor)
    res0 = eval_all(cur_flat)
    print("PPOV2 u0 (warm start): " + json.dumps({k: round(v, 3) for k, v in res0.items()}), flush=True)
    best_mean = float(np.mean(list(res0.values())))
    best_flat = cur_flat
    json.dump(best_flat, open("/workspace/bc_ppo_v2_u0.json", "w"))

    for u in range(1, UPDATES + 1):
        t0 = time.time()
        a_flat = actor_to_bc_flat(policy.actor)
        c_flat = critic_flat_of(critic)
        ls = list(policy.log_std)
        scs = [MIX[int(rng.integers(0, len(MIX)))] for _ in range(EPISODES_PER_UPDATE)]
        args = [(a_flat, ls, c_flat, int(rng.integers(0, 2**31 - 1)), sc) for sc in scs]
        eps = parallel_episodes(rollout_one, args)

        # GAE per episode, then concat
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
                # d(-surrogate)/dmu = -A*ratio*(a-mu)/std^2 (descent moves mu toward good actions)
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
        json.dump(cur_flat, open(f"/workspace/bc_ppo_v2_u{u}.json", "w"))
        res = eval_all(cur_flat)
        mean = float(np.mean(list(res.values())))
        floors_ok = res["goto"] >= 0.9 and res["hover_hold"] >= 0.6
        if mean > best_mean and floors_ok:
            best_mean = mean
            best_flat = cur_flat
            json.dump(best_flat, open("/workspace/bc_ppo_v2_best.json", "w"))
        post_series("ppo_v2_return", mean_ret, f"u{u}")
        for sc, v in res.items():
            post_series(f"success_{sc}", v, f"ppo-v2 u{u}")
        print(f"PPOV2 u{u}: ret={mean_ret:.2f} " +
              json.dumps({k: round(v, 3) for k, v in res.items()}) +
              f" best_mean={best_mean:.3f} wall={time.time()-t0:.0f}s", flush=True)
        post_status({
            "live_policy": {"name": "bc_flat.json",
                            "detail": "v1 policy - 15-dim obs, chassis_v1 dynamics",
                            "note": "flying on /watch; promotion candidate below"},
            "candidate": {"name": "bc_flat_v2.json",
                          "detail": "obs v2 (19-dim) - v14-g13 + com dynamics",
                          "goto": res["goto"], "hover_hold": res["hover_hold"], "land": res["land"],
                          "note": f"ppo-v2 u{u} eval (warm start from dagger_land best)"},
            "training": {"status": "running", "name": "ppo_v2", "iter": u, "iters": UPDATES,
                         "note": f"goto {res['goto']:.2f} hover {res['hover_hold']:.2f} land {res['land']:.2f}"},
            "queue": ["obs v3: rescale radius signal (dim18) for descend-phase learning"],
        })

    json.dump(best_flat, open("/workspace/bc_ppo_v2_best.json", "w"))
    post_status({
        "live_policy": {"name": "bc_flat.json",
                        "detail": "v1 policy - 15-dim obs, chassis_v1 dynamics",
                        "note": "flying on /watch; promotion candidate below"},
        "candidate": {"name": "bc_ppo_v2_best.json" if os.path.exists("/workspace/bc_ppo_v2_best.json") else "bc_flat_v2.json",
                      "detail": "obs v2 (19-dim) - v14-g13 + com dynamics",
                      "note": f"ppo_v2 finished; best mean {best_mean:.3f}"},
        "training": {"status": "idle", "name": "ppo_v2",
                     "note": f"finished {UPDATES} updates; best mean {best_mean:.3f}"},
        "queue": ["obs v3: rescale radius signal (dim18) for descend-phase learning"],
    })
    print(f"PPOV2_DONE best_mean={best_mean:.3f}", flush=True)

if __name__ == "__main__":
    main()
