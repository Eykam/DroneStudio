"""Evaluator: the source of truth for a scene-distribution variant's fitness.

Evaluator-first design: the outer loop is only as good as this number. A
variant's score = held-out success rate of a policy TRAINED under that
variant, measured on fresh episodes it never trained on.

Backends: "quad" (env_quad.QuadNavEnv - DroneStudio flight model, default)
or "stub" (env.StubNavEnv - original analytic placeholder, kept for A/B).
"""
import numpy as np
from policy import cem_train

def _backend(name):
    if name == "quad":
        from env_quad import QuadNavEnv, make_quad_factory
        return QuadNavEnv, make_quad_factory
    if name == "sim":
        from env_sim import SimBinaryEnv, make_sim_factory
        return SimBinaryEnv, make_sim_factory
    from env import StubNavEnv, make_stub_factory
    return StubNavEnv, make_stub_factory

def _rollout_act_factory(trainer, trained, obs_dim, act_dim):
    """Uniform act(obs) interface over the two trainer output types."""
    if trainer == "ppo":
        import numpy as _np
        from ppo import MLP
        actor = MLP(_np.random.default_rng(0), obs_dim, 128, act_dim)
        actor.load({k.split(".", 1)[1]: v for k, v in trained.items() if k.startswith("actor.")})
        mean, var = trained["norm_mean"], trained["norm_var"]
        def act(obs):
            nobs = _np.clip((obs - mean) / _np.sqrt(var + 1e-8), -10, 10)
            mu, _ = actor.forward(nobs[None, :])
            return _np.tanh(mu[0])
        return act
    return trained.act


def evaluate_distribution(dist, train_seed=0, eval_seed=10_000, cem_iters=3,
                          cem_pop=8, train_episodes=4, eval_episodes=6,
                          max_steps=200, verbose=False, backend="quad",
                          trainer="cem", ppo_config=None, dynamics=None):
    EnvCls, make_factory = _backend(backend)
    if backend == "sim" and dynamics:
        factory = make_factory(dist, max_steps=max_steps, dynamics=dynamics)
    else:
        factory = make_factory(dist, max_steps=max_steps)
    if trainer == "ppo":
        from ppo import train_ppo, PPOConfig
        cfg = ppo_config or PPOConfig(seed=train_seed)
        res = train_ppo(factory, EnvCls.OBS_DIM, EnvCls.ACT_DIM, config=cfg)
        trained = {**{f"actor.{k}": v for k, v in res.best_params.get("actor", {}).items()},
                   "log_std": res.best_params.get("log_std"),
                   "norm_mean": res.best_params.get("norm_mean"),
                   "norm_var": res.best_params.get("norm_var")}
        train_ret = res.best_eval_mean
        act = _rollout_act_factory("ppo", trained, EnvCls.OBS_DIM, EnvCls.ACT_DIM)
    else:
        policy, train_ret = cem_train(factory, EnvCls.OBS_DIM, EnvCls.ACT_DIM,
                                      iters=cem_iters, pop=cem_pop,
                                      episodes_per_eval=train_episodes,
                                      seed=train_seed, verbose=verbose)
        act = policy.act
    returns, successes, steps = [], [], []
    for i in range(eval_episodes):
        env = factory(eval_seed + i)  # factory carries dynamics/max_steps
        obs = env.reset()
        total = 0.0
        for _ in range(max_steps):
            obs, r, done = env.step(act(obs))
            total += r
            if done:
                break
        returns.append(total)
        successes.append(1.0 if env.succeeded else 0.0)
        steps.append(env.steps)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_steps": float(np.mean(steps)),
        "train_best_return": float(train_ret),
        "backend": backend,
        "trainer": trainer,
        "dynamics": dynamics or "abstract",
    }
