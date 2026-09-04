"""PPO trainer for the auto-researcher inner loop.

Replaces CEM with a proper policy-gradient method: actor-critic MLP in pure
numpy, clipped surrogate objective, GAE, minibatched epochs. Same interface
as the CEM trainer so the outer loop can swap trainers by config.

Design notes for Eyad:
- Why PPO over CEM: CEM is a zeroth-order black-box search over flat MLP
  weights. It works but wastes samples and cannot scale to larger policies
  (e.g. the transformer nav policy on the roadmap). PPO is first-order,
  on-policy, and the standard baseline for quadrotor navigation policies
  trained in simulation (cf. quad-rotor RL literature, e.g. Loquercio et al.
  2021 used a privileged-learning pipeline but PPO-class on-policy updates).
- Pure numpy keeps the box dependency-free (no torch). Adam implemented
  manually with numerical-stability guards. Backprop through a 2-hidden-layer
  tanh MLP is ~40 lines; keeping it explicit makes it auditable.
- GAE(lambda=0.95), clip 0.2, entropy bonus 1e-3, value loss 0.5, Adam lr
  3e-4 with linear decay, 10 epochs per rollout, minibatches of 256.
  Standard stable-baselines-flavored defaults, tuned lightly for the
  20Hz/500Hz QuadNavEnv horizon (15s episodes = 300 policy steps).
- Observation normalization: running mean/std (Welford), frozen during eval.
  Critical here because QuadNavEnv obs mix meters, rad/s, and [0,1] gates.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict

import numpy as np


# ---------------------------------------------------------------- utilities

class RunningNorm:
    """Welford running mean/std for observation normalization."""

    def __init__(self, dim: int, eps: float = 1e-8):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        n = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        m_a = self.var * self.count
        m_b = batch_var * n
        m2 = m_a + m_b + delta ** 2 * self.count * n / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype(np.float64)


class MLP:
    """2-hidden-layer tanh MLP with manual backprop (actor or critic)."""

    def __init__(self, rng: np.random.Generator, in_dim: int, hid: int, out_dim: int,
                 out_scale: float = 0.01):
        self.w1 = _xavier(rng, in_dim, hid)
        self.b1 = np.zeros(hid)
        self.w2 = _xavier(rng, hid, hid)
        self.b2 = np.zeros(hid)
        self.w3 = _xavier(rng, hid, out_dim) * out_scale
        self.b3 = np.zeros(out_dim)

    def forward(self, x: np.ndarray):
        z1 = x @ self.w1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.tanh(z2)
        out = a2 @ self.w3 + self.b3
        return out, (x, a1, a2)

    def backward(self, dout: np.ndarray, cache) -> dict:
        x, a1, a2 = cache
        dw3 = a2.T @ dout
        db3 = dout.sum(axis=0)
        da2 = dout @ self.w3.T
        dz2 = da2 * (1.0 - a2 ** 2)
        dw2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.w2.T
        dz1 = da1 * (1.0 - a1 ** 2)
        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2, "w3": dw3, "b3": db3}

    def params(self) -> dict:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2,
                "w3": self.w3, "b3": self.b3}

    def load(self, p: dict) -> None:
        self.w1, self.b1 = p["w1"], p["b1"]
        self.w2, self.b2 = p["w2"], p["b2"]
        self.w3, self.b3 = p["w3"], p["b3"]


class Adam:
    def __init__(self, params: dict, lr: float = 3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ------------------------------------------------------------------- PPO

@dataclass
class PPOConfig:
    hidden: int = 128
    lr: float = 3e-4
    lr_decay: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 10
    minibatch: int = 256
    rollout_episodes: int = 16          # episodes collected per update
    max_updates: int = 200              # total updates (stopping budget)
    init_log_std: float = -0.5
    target_reward: float | None = None  # early stop when eval mean exceeds
    eval_every: int = 10                # eval cadence (updates)
    eval_episodes: int = 8
    obs_norm_clip: float = 10.0
    seed: int = 0


@dataclass
class PPOResult:
    best_eval_mean: float = -np.inf
    best_update: int = -1
    total_env_steps: int = 0
    wall_seconds: float = 0.0
    eval_history: list = field(default_factory=list)
    best_params: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


class GaussianPolicy:
    """Tanh-squashed diagonal Gaussian over velocity-setpoint actions."""

    def __init__(self, rng, obs_dim, act_dim, hid, log_std_init):
        self.actor = MLP(rng, obs_dim, hid, act_dim)
        self.log_std = np.full(act_dim, log_std_init, dtype=np.float64)

    def sample(self, obs, rng):
        mu, cache = self.actor.forward(obs)
        std = np.exp(self.log_std)
        a = mu + std * rng.standard_normal(mu.shape)
        logp = self._logp(a, mu, std)
        return mu, a, logp, cache

    def _logp(self, a, mu, std):
        d = mu.shape[-1]
        return (-0.5 * (((a - mu) / std) ** 2).sum(-1)
                - self.log_std.sum() - 0.5 * d * np.log(2 * np.pi))


def train_ppo(env_factory, obs_dim: int, act_dim: int,
              config: PPOConfig | None = None, progress_cb=None) -> PPOResult:
    """Train a GaussianPolicy via PPO.

    Same contract as cem_train: env_factory(seed) -> fresh env, env.reset()
    -> obs, env.step(action) -> (obs, reward, done). Actions are clipped to
    [-1, 1] by the env; the policy's tanh squash already lands in range.
    """
    cfg = config or PPOConfig()
    rng = np.random.default_rng(cfg.seed)

    policy = GaussianPolicy(rng, obs_dim, act_dim, cfg.hidden, cfg.init_log_std)
    critic = MLP(rng, obs_dim, cfg.hidden, 1)
    norm = RunningNorm(obs_dim)

    actor_opt = Adam(policy.actor.params(), lr=cfg.lr)
    # log_std optimized with the actor params under its own key
    actor_opt.m["log_std"] = np.zeros_like(policy.log_std)
    actor_opt.v["log_std"] = np.zeros_like(policy.log_std)
    critic_opt = Adam(critic.params(), lr=cfg.lr)

    result = PPOResult(config=asdict(cfg))
    t0 = time.time()

    for update in range(cfg.max_updates):
        if cfg.lr_decay:
            frac = 1.0 - update / cfg.max_updates
            actor_opt.lr = cfg.lr * frac
            critic_opt.lr = cfg.lr * frac

        # ---------------- rollout: collect rollout_episodes trajectories
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
        for ep in range(cfg.rollout_episodes):
            env = env_factory(int(rng.integers(0, 2**31 - 1)))
            obs = env.reset()
            done = False
            while not done:
                nobs = np.clip(norm.normalize(obs), -cfg.obs_norm_clip, cfg.obs_norm_clip)
                mu, a, logp, _ = policy.sample(nobs[None, :], rng)
                v, _ = critic.forward(nobs[None, :])
                a_env = np.tanh(a[0])  # squash to env action range
                nxt, r, done = env.step(a_env)
                obs_buf.append(nobs); act_buf.append(a[0]); rew_buf.append(r)
                val_buf.append(v[0, 0]); logp_buf.append(logp[0]); done_buf.append(done)
                obs = nxt
                result.total_env_steps += 1
                norm.update(obs)

        obs_b = np.array(obs_buf); act_b = np.array(act_buf)
        rew_b = np.array(rew_buf); val_b = np.array(val_buf)
        logp_b = np.array(logp_buf); done_b = np.array(done_buf, dtype=np.float64)

        # ---------------- GAE
        adv = np.zeros_like(rew_b)
        lastgae = 0.0
        # terminal value 0: episodes always end (done) inside the buffer
        for t in reversed(range(len(rew_b))):
            nextval = val_b[t + 1] if t + 1 < len(rew_b) else 0.0
            nonterminal = 1.0 - done_b[t]
            delta = rew_b[t] + cfg.gamma * nextval * nonterminal - val_b[t]
            lastgae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * lastgae
            adv[t] = lastgae
        ret = adv + val_b
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---------------- PPO epochs
        n = len(rew_b)
        idx = np.arange(n)
        for _ in range(cfg.epochs):
            rng.shuffle(idx)
            for start in range(0, n, cfg.minibatch):
                mb = idx[start:start + cfg.minibatch]
                mo, ma, madv, mret, mlogp = obs_b[mb], act_b[mb], adv[mb], ret[mb], logp_b[mb]

                # actor grads (via score function on mu and log_std)
                mu, cache = policy.actor.forward(mo)
                std = np.exp(policy.log_std)
                logp = policy._logp(ma, mu, std)
                ratio = np.exp(logp - mlogp)
                s1 = ratio * madv
                s2 = np.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * madv
                # d(-min(s1,s2))/dlogp = -madv * [active branch]
                active = np.where((s1 <= s2), 1.0, 0.0)
                clip_active = (ratio >= 1 - cfg.clip) & (ratio <= 1 + cfg.clip)
                coef = -madv * active * clip_active  # (B,) - d(-surrogate)/dlogp
                dlogp_dmu = (ma - mu) / (std ** 2)   # (B,A)
                # d(-surrogate)/dparams = d(-surrogate)/dlogp * dlogp/dparams (sign fixed:
                # the original flipped mu and log_std to ascent, which anti-learned)
                dmu = dlogp_dmu * (coef[:, None]) / len(mb)
                dlogstd = (((ma - mu) ** 2 / (std ** 2)) - 1.0) * (coef[:, None]) / len(mb)
                # entropy bonus: dH/dlog_std = 1 per dim -> subtract coef
                dlogstd -= cfg.entropy_coef / len(mb)

                agrads = policy.actor.backward(dmu, cache)
                agrads["log_std"] = dlogstd.sum(axis=0)
                allp = policy.actor.params(); allp["log_std"] = policy.log_std
                # grad clip
                tot = np.sqrt(sum((g ** 2).sum() for g in agrads.values()))
                if tot > cfg.max_grad_norm:
                    for k in agrads:
                        agrads[k] *= cfg.max_grad_norm / (tot + 1e-8)
                actor_opt.step(allp, agrads)
                policy.log_std = allp["log_std"]

                # critic grads
                v, vcache = critic.forward(mo)
                dv = 2.0 * (v[:, 0] - mret)[:, None] * cfg.value_coef / len(mb)
                cgrads = critic.backward(dv, vcache)
                tot = np.sqrt(sum((g ** 2).sum() for g in cgrads.values()))
                if tot > cfg.max_grad_norm:
                    for k in cgrads:
                        cgrads[k] *= cfg.max_grad_norm / (tot + 1e-8)
                critic_opt.step(critic.params(), cgrads)

        # ---------------- eval
        if (update + 1) % cfg.eval_every == 0 or update == cfg.max_updates - 1:
            eval_rews = []
            for _ in range(cfg.eval_episodes):
                env = env_factory(int(rng.integers(0, 2**31 - 1)))
                obs = env.reset()
                done, ep_r = False, 0.0
                while not done:
                    nobs = np.clip(norm.normalize(obs), -cfg.obs_norm_clip, cfg.obs_norm_clip)
                    mu, _ = policy.actor.forward(nobs[None, :])
                    obs, r, done = env.step(np.tanh(mu[0]))
                    ep_r += r
                eval_rews.append(ep_r)
            em = float(np.mean(eval_rews))
            result.eval_history.append({"update": update + 1, "eval_mean": em,
                                        "eval_min": float(np.min(eval_rews)),
                                        "steps": result.total_env_steps})
            if em > result.best_eval_mean:
                result.best_eval_mean = em
                result.best_update = update + 1
                result.best_params = {
                    "actor": {k: v.copy() for k, v in policy.actor.params().items()},
                    "log_std": policy.log_std.copy(),
                    "norm_mean": norm.mean.copy(), "norm_var": norm.var.copy(),
                }
            if progress_cb:
                progress_cb(update + 1, em, result.best_eval_mean)
            if cfg.target_reward is not None and em >= cfg.target_reward:
                break

    result.wall_seconds = time.time() - t0
    return result


def save_policy(result: PPOResult, path: str) -> None:
    """Serialize best params (npz) + config/eval history (json sidecar)."""
    flat = {}
    for k, v in result.best_params.get("actor", {}).items():
        flat[f"actor.{k}"] = v
    flat["log_std"] = result.best_params.get("log_std", np.zeros(1))
    flat["norm_mean"] = result.best_params.get("norm_mean", np.zeros(1))
    flat["norm_var"] = result.best_params.get("norm_var", np.ones(1))
    np.savez(path, **flat)
    with open(path + ".json", "w") as f:
        json.dump({"best_eval_mean": result.best_eval_mean,
                   "best_update": result.best_update,
                   "total_env_steps": result.total_env_steps,
                   "wall_seconds": result.wall_seconds,
                   "eval_history": result.eval_history,
                   "config": result.config}, f, indent=2)
