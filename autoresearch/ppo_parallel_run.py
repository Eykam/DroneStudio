#!/usr/bin/env python3
"""PPO at real budget (1M+ env steps) on sim+manifest dynamics, with rollout
collection parallelized across a fork pool. Update math identical to
ppo.train_ppo (GAE, clipped objective, advantage normalization, lr decay);
only the episode collection fan-out changes. Answers: does PPO learn where it
went 0/6 at 62k steps, and the success crossover vs CEM at equal wall-clock.
"""
import os, sys, json, time
import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scene_schema import SceneDistribution
from ppo import GaussianPolicy, MLP, Adam, RunningNorm, PPOConfig

DIST_PARAMS = {"obstacle_density": 0.018, "obstacle_size_mean": 0.75,
               "obstacle_size_spread": 0.08, "corridor_width": 10.0,
               "scene_extent": 22.0, "goal_distance": 5.0,
               "light_intensity": 0.92, "light_direction_entropy": 0.35,
               "texture_variety": 0.55, "dynamics_noise": 0.0}
DYNAMICS = os.environ.get("PPO_DYNAMICS",
    "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json")
MAX_STEPS = 250
TARGET_STEPS = int(os.environ.get("PPO_TARGET_STEPS", "1200000"))
JOBS = int(os.environ.get("PPO_JOBS", "12"))
OUT = os.environ.get("PPO_OUT", "/workspace/ppo_1m_report.json")
LOG = os.environ.get("PPO_LOG", "/workspace/ppo_1m.log")

_WORKER = {}

def _init(spec):
    backend, dist_json, max_steps, dynamics = spec
    from scene_schema import SceneDistribution as SD
    dist = SD.from_json(dist_json)
    if backend == "sim":
        from env_sim import SimBinaryEnv, make_sim_factory
        _WORKER["factory"] = make_sim_factory(dist, max_steps=max_steps, dynamics=dynamics)
        _WORKER["dims"] = (SimBinaryEnv.OBS_DIM, SimBinaryEnv.ACT_DIM)
    else:
        from env_quad import QuadNavEnv, make_quad_factory
        _WORKER["factory"] = make_quad_factory(dist, max_steps=max_steps)
        _WORKER["dims"] = (QuadNavEnv.OBS_DIM, QuadNavEnv.ACT_DIM)

def _episode(task):
    """One stochastic rollout; returns trajectory + sampled logp."""
    actor_params, log_std, mean, var, clip, seed, deterministic = task
    obs_dim, act_dim = _WORKER["dims"]
    rng = np.random.default_rng(seed)
    pol = GaussianPolicy(np.random.default_rng(0), obs_dim, act_dim, 128, -0.5)
    pol.actor.load(actor_params)
    pol.log_std = log_std.copy()
    env = _WORKER["factory"](seed)
    obs = env.reset()
    out = {"nobs": [], "a": [], "r": [], "done": [], "logp": [], "raw": [],
           "success": False, "steps": 0}
    done = False
    while not done:
        nobs = np.clip((obs - mean) / np.sqrt(var + 1e-8), -clip, clip)
        if deterministic:
            mu, _ = pol.actor.forward(nobs[None, :])
            a = mu[0]
            logp = 0.0
        else:
            mu, a2, lp, _ = pol.sample(nobs[None, :], rng)
            a, logp = a2[0], lp[0]
        nxt, r, done = env.step(np.tanh(a))
        out["nobs"].append(nobs); out["a"].append(a); out["r"].append(r)
        out["done"].append(done); out["logp"].append(logp); out["raw"].append(obs)
        obs = nxt
        out["steps"] += 1
    out["success"] = bool(env.succeeded)
    env.close()
    return out


def main():
    cfg = PPOConfig(hidden=128, lr=1e-4, lr_decay=True, gamma=0.99,
                    gae_lambda=0.95, clip=0.2, entropy_coef=1e-3,
                    value_coef=0.5, max_grad_norm=0.5, epochs=10,
                    minibatch=256, rollout_episodes=16, max_updates=100000,
                    init_log_std=-0.5, eval_every=25, eval_episodes=8,
                    obs_norm_clip=10.0, seed=0)
    names = set(SceneDistribution.__dataclass_fields__)
    dist = SceneDistribution(**{k: v for k, v in DIST_PARAMS.items() if k in names})
    spec = ("sim", dist.to_json(), MAX_STEPS, DYNAMICS)
    from env_sim import SimBinaryEnv
    obs_dim, act_dim = SimBinaryEnv.OBS_DIM, SimBinaryEnv.ACT_DIM

    rng = np.random.default_rng(cfg.seed)
    policy = GaussianPolicy(rng, obs_dim, act_dim, cfg.hidden, cfg.init_log_std)
    critic = MLP(rng, obs_dim, cfg.hidden, 1)
    norm = RunningNorm(obs_dim)
    actor_opt = Adam(policy.actor.params(), lr=cfg.lr)
    actor_opt.m["log_std"] = np.zeros_like(policy.log_std)
    actor_opt.v["log_std"] = np.zeros_like(policy.log_std)
    critic_opt = Adam(critic.params(), lr=cfg.lr)

    log = open(LOG, "a", buffering=1)
    def say(s):
        print(s, flush=True)
        log.write(s + "\n")

    eval_hist = []
    total_steps = 0
    update = 0
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(min(JOBS, cfg.rollout_episodes), initializer=_init, initargs=(spec,)) as pool:
        while total_steps < TARGET_STEPS:
            update += 1
            frac = max(0.0, 1.0 - total_steps / TARGET_STEPS)
            actor_opt.lr = cfg.lr * frac
            critic_opt.lr = cfg.lr * frac

            tasks = [(policy.actor.params(), policy.log_std, norm.mean, norm.var,
                      cfg.obs_norm_clip, int(rng.integers(0, 2**31 - 1)), False)
                     for _ in range(cfg.rollout_episodes)]
            eps = pool.map(_episode, tasks)

            obs_b = np.array([x for e in eps for x in e["nobs"]])
            act_b = np.array([x for e in eps for x in e["a"]])
            rew_b = np.array([x for e in eps for x in e["r"]], dtype=np.float64)
            done_b = np.array([x for e in eps for x in e["done"]], dtype=np.float64)
            logp_b = np.array([x for e in eps for x in e["logp"]])
            for e in eps:
                for o in e["raw"]:
                    norm.update(o)
            total_steps += sum(e["steps"] for e in eps)
            val_b = critic.forward(obs_b)[0][:, 0]

            # GAE
            adv = np.zeros_like(rew_b)
            lastgae = 0.0
            for t in reversed(range(len(rew_b))):
                nextval = val_b[t + 1] if t + 1 < len(rew_b) else 0.0
                nonterminal = 1.0 - done_b[t]
                delta = rew_b[t] + cfg.gamma * nextval * nonterminal - val_b[t]
                lastgae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * lastgae
                adv[t] = lastgae
            ret = adv + val_b
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            n = len(rew_b)
            idx = np.arange(n)
            for _ in range(cfg.epochs):
                rng.shuffle(idx)
                for start in range(0, n, cfg.minibatch):
                    mb = idx[start:start + cfg.minibatch]
                    mo, ma, madv, mret, mlogp = obs_b[mb], act_b[mb], adv[mb], ret[mb], logp_b[mb]
                    mu, cache = policy.actor.forward(mo)
                    std = np.exp(policy.log_std)
                    logp = policy._logp(ma, mu, std)
                    ratio = np.exp(logp - mlogp)
                    s1 = ratio * madv
                    s2 = np.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * madv
                    active = np.where((s1 <= s2), 1.0, 0.0)
                    clip_active = (ratio >= 1 - cfg.clip) & (ratio <= 1 + cfg.clip)
                    coef = -madv * active * clip_active
                    dlogp_dmu = (ma - mu) / (std ** 2)
                    dmu = dlogp_dmu * (-coef[:, None]) / len(mb)
                    dlogstd = (((ma - mu) ** 2 / (std ** 2)) - 1.0) * (-coef[:, None]) / len(mb)
                    dlogstd -= cfg.entropy_coef / len(mb)
                    agrads = policy.actor.backward(dmu, cache)
                    agrads["log_std"] = dlogstd.sum(axis=0)
                    allp = policy.actor.params(); allp["log_std"] = policy.log_std
                    tot = np.sqrt(sum((g ** 2).sum() for g in agrads.values()))
                    if tot > cfg.max_grad_norm:
                        for k in agrads:
                            agrads[k] *= cfg.max_grad_norm / (tot + 1e-8)
                    actor_opt.step(allp, agrads)
                    policy.log_std = allp["log_std"]
                    v, vcache = critic.forward(mo)
                    verr = v[:, 0] - mret
                    dv = (2.0 * verr / len(mb))[:, None]
                    cgrads = critic.backward(dv, vcache)
                    tot = np.sqrt(sum((g ** 2).sum() for g in cgrads.values()))
                    if tot > cfg.max_grad_norm:
                        for k in cgrads:
                            cgrads[k] *= cfg.max_grad_norm / (tot + 1e-8)
                    critic_opt.step(critic.params(), cgrads)

            if update % cfg.eval_every == 0 or total_steps >= TARGET_STEPS:
                etasks = [(policy.actor.params(), policy.log_std, norm.mean, norm.var,
                           cfg.obs_norm_clip, 20_000 + i, True)
                          for i in range(cfg.eval_episodes)]
                eeps = pool.map(_episode, etasks)
                succ = float(np.mean([e["success"] for e in eeps]))
                mean_ret = float(np.mean([sum(e["r"]) for e in eeps]))
                mean_len = float(np.mean([e["steps"] for e in eeps]))
                eval_hist.append({"update": update, "env_steps": total_steps,
                                  "eval_success": succ, "eval_mean_return": mean_ret,
                                  "eval_mean_steps": mean_len,
                                  "wall_s": round(time.time() - t0, 1)})
                say(json.dumps(eval_hist[-1]))
                if succ >= 0.99 and total_steps > 200_000:
                    break

    wall = time.time() - t0
    rep = {"config": {k: getattr(cfg, k) for k in ("lr", "hidden", "rollout_episodes",
            "epochs", "minibatch", "gamma", "gae_lambda", "clip")},
           "dynamics": DYNAMICS, "dist": DIST_PARAMS, "jobs": JOBS,
           "total_env_steps": total_steps, "updates": update,
           "wall_s": round(wall, 1), "eval_history": eval_hist,
           "final": eval_hist[-1] if eval_hist else None,
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2)
    say("PPO_RUN_COMPLETE " + json.dumps(rep["final"]))

if __name__ == "__main__":
    main()
