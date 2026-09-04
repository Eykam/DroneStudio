"""Tiny MLP navigation policy + CEM trainer (numpy only, CPU-scale).

CEM (cross-entropy method) is the skeleton's inner-loop trainer: zero deps,
robust at tiny scale, honest about what it is. Swap point for torch/PPO once
the loop is proven (see README next-steps).
"""
import os
import multiprocessing as mp
import numpy as np

class MLP:
    def __init__(self, obs_dim, act_dim, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        self.shapes = [(obs_dim, hidden), (hidden,), (hidden, hidden), (hidden,), (hidden, act_dim), (act_dim,)]
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self._init_random(rng)

    def _init_random(self, rng):
        self.W1 = rng.normal(0, 0.5, self.shapes[0]); self.b1 = np.zeros(self.shapes[1])
        self.W2 = rng.normal(0, 0.5, self.shapes[2]); self.b2 = np.zeros(self.shapes[3])
        self.W3 = rng.normal(0, 0.5, self.shapes[4]); self.b3 = np.zeros(self.shapes[5])

    def act(self, obs):
        h = np.tanh(obs @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        return np.tanh(h @ self.W3 + self.b3)

    def get_flat(self):
        return np.concatenate([p.ravel() for p in (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)])

    def set_flat(self, flat):
        out, i = [], 0
        for shp in self.shapes:
            n = int(np.prod(shp))
            out.append(flat[i:i+n].reshape(shp))
            i += n
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = out

    @classmethod
    def param_count(cls, obs_dim, act_dim, hidden=32):
        return obs_dim*hidden + hidden + hidden*hidden + hidden + hidden*act_dim + act_dim

def cem_train(env_factory, obs_dim, act_dim, iters=3, pop=8, elite_frac=0.375,
              episodes_per_eval=4, seed=0, verbose=False,
              init_mean=None, init_std=None, fitness=None):
    if fitness is not None:
        global FITNESS_MODE
        FITNESS_MODE = fitness
    """Train MLP weights by CEM. env_factory(seed) -> env with reset/step."""
    rng = np.random.default_rng(seed)
    n = MLP.param_count(obs_dim, act_dim)
    mean = np.array(init_mean, dtype=np.float64) if init_mean is not None else np.zeros(n)
    std = np.full(n, float(init_std)) if init_std is not None else np.ones(n) * 0.5
    n_elite = max(2, int(pop * elite_frac))
    policy = MLP(obs_dim, act_dim, seed=seed)
    best_ret, best_vec = -np.inf, mean.copy()
    for it in range(iters):
        cands = mean + std * rng.normal(0, 1, (pop, n))
        rets = []
        for c in cands:
            policy.set_flat(c)
            rets.append(np.mean([_run_episode(env_factory, policy, seed=int(rng.integers(1 << 30)))
                                 for _ in range(episodes_per_eval)]))
        rets = np.array(rets)
        elite = cands[np.argsort(rets)[-n_elite:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-3
        if rets.max() > best_ret:
            best_ret, best_vec = float(rets.max()), cands[int(rets.argmax())].copy()
        if verbose:
            print(f"  cem iter {it}: best={rets.max():.2f} mean={rets.mean():.2f}")
    policy.set_flat(best_vec)
    return policy, best_ret

# Fitness shaping for TRAINING only (the env's raw reward stays the reporting
# metric). "return": episode reward sum (original). "progress": fraction of
# start-distance closed + success bonus - does not reward early termination,
# which the raw return does (crash-fast beats hover-and-miss under -0.01/step).
FITNESS_MODE = "return"

def _episode_fitness(env, obs0, obs_end, total, succeeded):
    if FITNESS_MODE == "progress":
        import numpy as _np
        d0 = float(_np.linalg.norm(obs0[0:3]))
        d1 = float(_np.linalg.norm(obs_end[0:3]))
        frac = (d0 - d1) / max(d0, 1e-9)
        # charge-and-crash near the goal must not score like flying there
        crash = 1.0 if getattr(env, "collided", False) and not succeeded else 0.0
        return frac + (1.0 if succeeded else 0.0) - crash
    return total

def _run_episode(env_factory, policy, seed):
    env = env_factory(seed)
    try:
        obs = env.reset()
        obs0 = obs.copy()
        total = 0.0
        for _ in range(env.max_steps):
            obs, r, done = env.step(policy.act(obs))
            total += r
            if done:
                break
        return _episode_fitness(env, obs0, obs, total, getattr(env, "succeeded", False))
    finally:
        env.close()  # reaps the sim child; without this every episode leaks a zombie


# --- parallel rollouts -------------------------------------------------------
# CEM candidate evaluation parallelized across worker processes (fork). Each
# worker rebuilds the env factory from a small picklable spec, so closures
# never cross the process boundary. Physics untouched: same 500Hz binary,
# same episodes, just evaluated concurrently.

_WORKER = {}

def _rollout_worker_init(spec):
    backend, dist_json, max_steps, dynamics = spec
    from scene_schema import SceneDistribution
    dist = SceneDistribution.from_json(dist_json)
    if backend == "sim":
        from env_sim import SimBinaryEnv, make_sim_factory
        _WORKER["factory"] = make_sim_factory(dist, max_steps=max_steps, dynamics=dynamics)
        _WORKER["obs_dim"], _WORKER["act_dim"] = SimBinaryEnv.OBS_DIM, SimBinaryEnv.ACT_DIM
    elif backend == "quad":
        from env_quad import QuadNavEnv, make_quad_factory
        _WORKER["factory"] = make_quad_factory(dist, max_steps=max_steps)
        _WORKER["obs_dim"], _WORKER["act_dim"] = QuadNavEnv.OBS_DIM, QuadNavEnv.ACT_DIM
    else:
        from env import StubNavEnv, make_stub_factory
        _WORKER["factory"] = make_stub_factory(dist, max_steps=max_steps)
        _WORKER["obs_dim"], _WORKER["act_dim"] = StubNavEnv.OBS_DIM, StubNavEnv.ACT_DIM
    _WORKER["policy"] = MLP(_WORKER["obs_dim"], _WORKER["act_dim"])

def _rollout_eval_candidate(task):
    vec, seeds = task
    policy = _WORKER["policy"]
    policy.set_flat(vec)
    return float(np.mean([_run_episode(_WORKER["factory"], policy, seed=s) for s in seeds]))

def default_jobs():
    # Benchmarked on the 48-core Railway box (bench_parallel, 480 real
    # Zig-sim episodes, 10m/d0.1): 12w=82.7 eps/s, 24w=129.8, 32w=153.4,
    # 48w=180.7. 32 keeps ~85% of peak throughput and leaves cores for the
    # trainer + streamer; override with N_JOBS.
    return min(int(os.environ.get("N_JOBS", "32")), os.cpu_count() or 1)

def cem_train_parallel(spec, obs_dim, act_dim, iters=3, pop=8, elite_frac=0.375,
                       episodes_per_eval=4, seed=0, n_jobs=None, verbose=False,
              init_mean=None, init_std=None, fitness=None):
    if fitness is not None:
        global FITNESS_MODE
        FITNESS_MODE = fitness
    """cem_train with the population evaluated on a process pool.
    spec = (backend, dist_json, max_steps, dynamics); identical math/seed
    scheme to cem_train otherwise."""
    n_jobs = n_jobs or default_jobs()
    rng = np.random.default_rng(seed)
    n = MLP.param_count(obs_dim, act_dim)
    mean = np.array(init_mean, dtype=np.float64) if init_mean is not None else np.zeros(n)
    std = np.full(n, float(init_std)) if init_std is not None else np.ones(n) * 0.5
    n_elite = max(2, int(pop * elite_frac))
    policy = MLP(obs_dim, act_dim, seed=seed)
    best_ret, best_vec = -np.inf, mean.copy()
    ctx = mp.get_context("fork")
    with ctx.Pool(min(n_jobs, pop), initializer=_rollout_worker_init,
                  initargs=(spec,)) as pool:
        for it in range(iters):
            cands = mean + std * rng.normal(0, 1, (pop, n))
            tasks = [(cands[i], [int(rng.integers(1 << 30))
                                 for _ in range(episodes_per_eval)])
                     for i in range(pop)]
            rets = np.array(pool.map(_rollout_eval_candidate, tasks))
            elite = cands[np.argsort(rets)[-n_elite:]]
            mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-3
            if rets.max() > best_ret:
                best_ret, best_vec = float(rets.max()), cands[int(rets.argmax())].copy()
            if verbose:
                print(f"  cem iter {it}: best={rets.max():.2f} mean={rets.mean():.2f}", flush=True)
    policy.set_flat(best_vec)
    return policy, best_ret
