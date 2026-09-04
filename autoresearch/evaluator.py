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
    from env import StubNavEnv, make_stub_factory
    return StubNavEnv, make_stub_factory

def evaluate_distribution(dist, train_seed=0, eval_seed=10_000, cem_iters=3,
                          cem_pop=8, train_episodes=4, eval_episodes=6,
                          max_steps=200, verbose=False, backend="quad"):
    EnvCls, make_factory = _backend(backend)
    factory = make_factory(dist, max_steps=max_steps)
    policy, train_ret = cem_train(factory, EnvCls.OBS_DIM, EnvCls.ACT_DIM,
                                  iters=cem_iters, pop=cem_pop,
                                  episodes_per_eval=train_episodes,
                                  seed=train_seed, verbose=verbose)
    returns, successes, steps = [], [], []
    for i in range(eval_episodes):
        env = EnvCls(dist, seed=eval_seed + i, max_steps=max_steps)
        obs = env.reset()
        total = 0.0
        for _ in range(max_steps):
            obs, r, done = env.step(policy.act(obs))
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
    }
