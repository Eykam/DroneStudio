"""PPO fine-tune with GT->est obs ANNEALING (fallback after dagger-est transfer loss).

Fraction of est-obs episodes per update ramps 0.25 -> 1.0 over the run;
GT episodes keep the reward signal dense while the policy adapts to est noise.
Base: ppo_est.py (same estimator, rewards, selection + GT floor).

ORIGINAL DOCSTRING:
PPO fine-tune under ESTIMATED obs (estimator-in-the-loop training track).

Warm-starts from bc_ppo_v2_best.json (obs v2 19-dim champion, GT-trained).
Every rollout/eval episode runs EstEnv(estimated=True, vo_aided=True): headless
Bullet physics + fast_telemetry -> SimIMU @500Hz -> ESKF, mag+ToF @20Hz,
synthetic-VO position aiding (per-episode randomized draws). Selection on
est-obs heldout cells; GT-obs goto regression floor (u0 value - 5pts).

Run:  cd /workspace/DroneStudio/autoresearch &&
      setsid nohup /workspace/venv-vision/bin/python ppo_est.py > /workspace/ppo_est.log 2>&1 < /dev/null &
"""
import os, json, time
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
from scenario_sampler import sample_spec, heldout_cells
from eval_scenarios import cell_dist, post_series
from parallel_rollout import parallel_episodes
from policy import MLP as BCMlp
from ppo import MLP, Adam, GaussianPolicy
from ppo_v2 import (bc_to_actor_params, actor_to_bc_flat, critic_flat_of,
                    train_dist, post_status, MANIFEST, OBS_DIM, ACT_DIM, HID,
                    LR, CLIP, GAMMA, LAM, EPOCHS, MINIBATCH,
                    SUCCESS_BONUS, CRASH_PENALTY, TIMEOUT_PENALTY, ALIVE_BONUS, MIX)
from eval_estimated import EstEnv

START = os.environ.get("EST_START", "/workspace/bc_ppo_v2_best.json")
OUT = "/workspace/bc_ppo_est_anneal"
UPDATES = int(os.environ.get("EST_UPDATES", "40"))
EPISODES_PER_UPDATE = int(os.environ.get("EST_EPS", "32"))
LOG_STD_INIT = -3.0
GT_FLOOR_DROP = 0.05

def make_est_env(dist, max_steps, spec, seed):
    return EstEnv(dist, seed=seed, max_steps=max_steps, dynamics=MANIFEST,
                  scenario_spec=spec, estimated=True, obs_v2=True,
                  vo_aided=True, est_seed=int(seed) + 777)

def rollout_one(actor_flat, log_std, critic_flat, seed, scenario, use_est):
    rng = np.random.default_rng(0)
    actor = MLP(rng, OBS_DIM, HID, ACT_DIM); actor.load(bc_to_actor_params(actor_flat))
    critic = MLP(rng, OBS_DIM, HID, 1)
    cp, i = {}, 0
    for k in ("w1", "b1", "w2", "b2", "w3", "b3"):
        ref = getattr(critic, k)
        cp[k] = np.array(critic_flat[i:i + ref.size], dtype=np.float64).reshape(ref.shape)
        i += ref.size
    critic.load(cp)
    std = np.exp(np.array(log_std, dtype=np.float64))
    dist = train_dist(seed)
    spec = sample_spec(seed, force_scenario=scenario)
    max_steps = 400 if scenario == "goto" else 700
    if use_est:
        env = make_est_env(dist, max_steps, spec, seed)
    else:
        from env_sim import make_sim_factory
        env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                               scenario_spec=spec)(seed)
    obs = env.reset()
    obs_b, act_b, rew_b, val_b, logp_b, done_b = [], [], [], [], [], []
    done = False
    while not done:
        mu, _ = actor.forward(obs[None, :]); mu = mu[0]
        a = mu + std * rng.standard_normal(ACT_DIM)
        logp = float(-0.5 * (((a - mu) / std) ** 2).sum()
                     - np.log(std).sum() - 0.5 * ACT_DIM * np.log(2 * np.pi))
        v, _ = critic.forward(obs[None, :])
        nxt, r, done = env.step(np.tanh(a))
        r = ALIVE_BONUS
        if done:
            if env.succeeded: r += SUCCESS_BONUS
            elif env.collided: r -= CRASH_PENALTY
            else: r -= TIMEOUT_PENALTY
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
    env = make_est_env(dist, max_steps, spec, seed)
    obs = env.reset()
    done = False
    while not done:
        mu, _ = actor.forward(obs[None, :])
        obs, r, done = env.step(np.tanh(mu[0]))
    ok = bool(env.succeeded); env.close()
    return ok

def eval_one_gt(actor_flat, scenario, seed):
    """GT-obs regression check (no estimator): same cells, binary GT obs."""
    from env_sim import make_sim_factory
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
    ok = bool(env.succeeded); env.close()
    return ok

def main():
    rng = np.random.default_rng(7)
    bc_flat = json.load(open(START))
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

    def eval_gt_goto(flat):
        args = [(flat, "goto", s) for s in cells["goto"]]
        return float(np.mean(parallel_episodes(eval_one_gt, args)))

    cur_flat = actor_to_bc_flat(policy.actor)
    res0 = eval_all(cur_flat)
    gt0 = eval_gt_goto(cur_flat)
    gt_floor = gt0 - GT_FLOOR_DROP
    print("PPOEST u0 (warm start, EST obs): " + json.dumps({k: round(v, 3) for k, v in res0.items()})
          + f" gt_goto={gt0:.3f} floor={gt_floor:.3f}", flush=True)
    best_mean = float(np.mean(list(res0.values())))
    best_flat = cur_flat
    json.dump(best_flat, open(OUT + "_u0.json", "w"))
    json.dump(best_flat, open(OUT + "_best.json", "w"))

    for u in range(1, UPDATES + 1):
        t0 = time.time()
        a_flat = actor_to_bc_flat(policy.actor)
        c_flat = critic_flat_of(critic)
        ls = list(policy.log_std)
        est_frac = min(1.0, 0.25 + 0.75 * (u - 1) / max(1, UPDATES - 1))
        scs = [MIX[int(rng.integers(0, len(MIX)))] for _ in range(EPISODES_PER_UPDATE)]
        args = [(a_flat, ls, c_flat, int(rng.integers(0, 2**31 - 1)), sc,
                 bool(rng.random() < est_frac)) for sc in scs]
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
                coeff = -madv * np.where(
                    ((madv >= 0) & (ratio < 1 + CLIP)) | ((madv < 0) & (ratio > 1 - CLIP)),
                    ratio, 0.0) / len(mb)
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
        json.dump(cur_flat, open(f"{OUT}_u{u}.json", "w"))
        res = eval_all(cur_flat)
        gtg = eval_gt_goto(cur_flat)
        mean = float(np.mean(list(res.values())))
        floors_ok = gtg >= gt_floor
        if mean > best_mean and floors_ok:
            best_mean = mean
            best_flat = cur_flat
            json.dump(best_flat, open(OUT + "_best.json", "w"))
        post_series("ppo_est_return", mean_ret, f"u{u}")
        for sc, v in res.items():
            post_series(f"est_success_{sc}", v, f"ppo-est u{u}")
        print(f"PPOEST u{u}: ret={mean_ret:.2f} " +
              json.dumps({k: round(v, 3) for k, v in res.items()}) +
              f" gt_goto={gtg:.3f} est_frac={est_frac:.2f} best_est={best_mean:.3f} wall={time.time()-t0:.0f}s", flush=True)
        post_status({
            "live_policy": {"name": "bc_ppo_v2_best.json",
                            "detail": "obs v2 (19-dim) GT-trained champion",
                            "note": "flying on /watch"},
            "candidate": {"name": "bc_ppo_est_anneal_best.json",
                          "detail": "estimator-in-the-loop PPO (synthetic-VO est obs)",
                          "goto": res["goto"], "hover_hold": res["hover_hold"], "land": res["land"],
                          "note": f"ppo-est u{u} EST-obs eval | GT-goto {gtg:.2f} (floor {gt_floor:.2f})"},
            "training": {"status": "running", "name": "ppo_est_anneal", "iter": u, "iters": UPDATES,
                         "note": f"EST goto {res['goto']:.2f} hover {res['hover_hold']:.2f} land {res['land']:.2f} | GT-goto {gtg:.2f}"},
            "queue": ["depth-net v1.1 retrain (vision track)"],
        })

    json.dump(best_flat, open(OUT + "_best.json", "w"))
    post_status({
        "training": {"status": "complete", "name": "ppo_est", "iter": UPDATES, "iters": UPDATES,
                     "note": f"best EST-obs mean {best_mean:.3f}"},
        "candidate": {"name": "bc_ppo_est_best.json",
                      "detail": "estimator-in-the-loop PPO (synthetic-VO est obs)",
                      "note": f"final best EST mean {best_mean:.3f}"},
    })
    print("PPOEST done. best_est_mean=", best_mean, flush=True)

if __name__ == "__main__":
    main()
