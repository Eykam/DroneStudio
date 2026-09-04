"""Environment backends for the inner loop.

StubNavEnv: analytic 2D point-mass navigation with obstacles sampled from a
SceneDistribution. This is a PLACEHOLDER dynamics stand-in: it exists so the
outer loop -> mutation -> train -> evaluate -> archive cycle can close and be
tested end-to-end tonight. It proves the loop machinery, not sim fidelity.

SimBinaryEnv: the real target - drives the DroneStudio binary headless.
Blocked on a sim-side headless episode API (the app is a GUI OpenGL program
expecting UDP pose input from the Pi rig). Next milestone.
"""
import numpy as np
from scene_schema import SceneDistribution

class StubNavEnv:
    OBS_DIM = 10
    ACT_DIM = 2

    def __init__(self, distribution: SceneDistribution, seed=0, max_steps=200):
        self.dist = distribution
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self._sample_scene()

    def _sample_scene(self):
        d, rng = self.dist, self.rng
        extent = d.scene_extent
        self.spawn = np.array([0.0, 0.0])
        ang = rng.uniform(0, 2 * np.pi)
        self.goal = self.spawn + d.goal_distance * np.array([np.cos(ang), np.sin(ang)])
        self.goal = np.clip(self.goal, -extent / 2, extent / 2)
        n_cells = int(extent * extent / 25)
        n_obs = int(d.obstacle_density * n_cells)
        centers = rng.uniform(-extent / 2, extent / 2, (max(n_obs, 1), 2))
        sizes = np.abs(rng.normal(d.obstacle_size_mean,
                                  d.obstacle_size_mean * d.obstacle_size_spread,
                                  max(n_obs, 1)))
        keep = (np.linalg.norm(centers - self.spawn, axis=1) > 4) & \
               (np.linalg.norm(centers - self.goal, axis=1) > 4)
        self.obs_centers, self.obs_radii = centers[keep], np.maximum(sizes[keep] / 2, 0.25)

    def reset(self):
        self.pos = self.spawn.copy()
        self.vel = np.zeros(2)
        self.steps = 0
        self.prev_dist = np.linalg.norm(self.goal - self.pos)
        self.collided = False
        return self._obs()

    def _obs(self):
        rel_goal = (self.goal - self.pos) / max(self.dist.scene_extent, 1)
        v = self.vel / 5.0
        feats = [rel_goal[0], rel_goal[1], v[0], v[1]]
        for _ in range(3):
            if len(self.obs_centers):
                d_vec = self.obs_centers - self.pos
                idx = int(np.argmin(np.linalg.norm(d_vec, axis=1)))
                rel = d_vec[idx] / max(self.dist.scene_extent, 1)
                feats += [rel[0], rel[1]]
            else:
                feats += [0.0, 0.0]
        return np.array(feats, dtype=np.float64)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1) * 2.0
        noise = self.rng.normal(0, self.dist.dynamics_noise, 2)
        self.vel = 0.9 * self.vel + 0.1 * (a + noise * 10)
        self.pos = self.pos + self.vel * 0.1
        self.steps += 1
        dist = np.linalg.norm(self.goal - self.pos)
        reward = (self.prev_dist - dist) * 1.0 - 0.01
        self.prev_dist = dist
        done = False
        if len(self.obs_centers):
            hit = np.any(np.linalg.norm(self.obs_centers - self.pos, axis=1) < self.obs_radii + 0.3)
            if hit:
                reward -= 5.0
                self.collided = True
                done = True
        if dist < 2.0:
            reward += 10.0
            done = True
        if self.steps >= self.max_steps:
            done = True
        return self._obs(), reward, done

    @property
    def succeeded(self):
        return (not self.collided) and np.linalg.norm(self.goal - self.pos) < 2.0

def make_stub_factory(distribution, max_steps=200):
    def factory(seed):
        return StubNavEnv(distribution, seed=seed, max_steps=max_steps)
    return factory

class SimBinaryEnv:
    """Next milestone: launch DroneStudio headless (xvfb + EGL), inject pose
    over UDP (the 1kHz loop the Pi rig feeds today), and read RGB-D frames for
    observations. Requires a sim-side episode API: reset, deterministic scene
    seed, step clock. Not implemented - the GUI app has no headless mode yet."""
    def __init__(self, *a, **k):
        raise NotImplementedError(
            "Sim backend pending sim-side headless episode API - see autoresearch/README.md")
