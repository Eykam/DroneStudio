"""QuadNavEnv: 3D quadrotor navigation env built on DroneStudio's own flight
model, for CPU-scale inner-loop training until the sim exposes a headless
episode API.

Parameters lifted from the sim source (Studio/src):
- mass 1.5 kg, inertia from 0.3 m arms: Ixx=Iyy=0.040, Izz=0.047 kg m^2
  (core/ecs/prefabs/Drone.zig)
- Rate PID gains roll/pitch [0.1, 0.005, 0.001], yaw [0.05, 0.003, 0.0005],
  anti-windup clamped (core/ecs/components/FlightController.zig +
  core/flight/PIDController.zig)
- Low-pass filters tau=0.05 s on thrust and rates; max rates
  (10.47, 10.47, 5.24) rad/s; max collective thrust 40 N; gravity -9.81 Y
- Thrust along body +Y, torques about body axes (matches the Bullet force
  application in core/ecs/components/PhysicsThread.zig)

Two control levels, mirroring the target architecture (classical fast loop +
learned nav policy): the POLICY outputs desired body rates + collective
thrust at ~20 Hz; the FAST LOOP (his PID, 500 Hz) tracks them. Semi-implicit
Euler integration. Not Bullet-exact - documented approximation; the
sim-backend milestone replaces dynamics wholesale.
"""
import numpy as np
from scene_schema import SceneDistribution

# --- his parameters ---
MASS = 1.5
IXX = IYY = 0.040
IZZ = 0.047
MAX_RATES = np.array([10.47, 10.47, 5.24])
MAX_THRUST = 40.0
FILTER_TAU = 0.05
RATE_GAINS = [(0.1, 0.005, 0.001), (0.1, 0.005, 0.001), (0.05, 0.003, 0.0005)]
GRAVITY = np.array([0.0, -9.81, 0.0])
FAST_DT = 1.0 / 500.0
POLICY_HZ = 20
FAST_PER_POLICY = int(1.0 / (POLICY_HZ * FAST_DT))


class PID:
    """His PIDController.zig: anti-windup clamp, output clamp, derivative on error."""
    def __init__(self, kp, ki, kd, ilimit=5.0, olimit=2.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.imin, self.imax = -ilimit, ilimit
        self.omin, self.omax = -olimit, olimit
        self.reset()

    def reset(self):
        self.integrator = 0.0
        self.last_error = 0.0

    def step(self, err, dt):
        self.integrator = np.clip(self.integrator + err * dt, self.imin, self.imax)
        d = (err - self.last_error) / dt
        self.last_error = err
        return float(np.clip(self.kp * err + self.ki * self.integrator + self.kd * d,
                             self.omin, self.omax))


def quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz])

def quat_rotate(q, v):
    qv = np.array([v[0], v[1], v[2], 0.0])
    qc = np.array([-q[0], -q[1], -q[2], q[3]])
    return quat_mul(quat_mul(q, qv), qc)[:3]


class QuadNavEnv:
    OBS_DIM = 15
    ACT_DIM = 4

    def __init__(self, distribution: SceneDistribution, seed=0, max_steps=200):
        self.dist = distribution
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps  # policy steps
        self._sample_scene()

    def _sample_scene(self):
        d, rng = self.dist, self.rng
        ext = d.scene_extent
        self.spawn = np.array([0.0, 2.0, 0.0])  # his drone spawns translated +2 Y
        ang = rng.uniform(0, 2 * np.pi)
        self.goal = self.spawn + np.array([d.goal_distance * np.cos(ang),
                                           rng.uniform(-3, 5),
                                           d.goal_distance * np.sin(ang)])
        self.goal[1] = max(self.goal[1], 0.5)
        n_obs = int(d.obstacle_density * ext * ext / 25)
        centers = rng.uniform(-ext/2, ext/2, (max(n_obs, 1), 3))
        centers[:, 1] = np.abs(centers[:, 1]) * 0.3 + 0.5  # low altitude band
        sizes = np.abs(rng.normal(d.obstacle_size_mean,
                                  d.obstacle_size_mean * d.obstacle_size_spread,
                                  max(n_obs, 1)))
        keep = (np.linalg.norm(centers - self.spawn, axis=1) > 4) & \
               (np.linalg.norm(centers - self.goal, axis=1) > 4)
        self.obs_centers = centers[keep]
        self.obs_radii = np.maximum(sizes[keep] / 2, 0.25)

    def reset(self):
        self.pos = self.spawn.copy()
        self.vel = np.zeros(3)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.omega = np.zeros(3)
        self.pids = [PID(*g) for g in RATE_GAINS]
        self.filtered_rates = np.zeros(3)
        self.filtered_thrust = 0.0
        self.steps = 0
        self.prev_dist = np.linalg.norm(self.goal - self.pos)
        self.collided = False
        return self._obs()

    def _obs(self):
        ext = max(self.dist.scene_extent, 1)
        rel_goal = (self.goal - self.pos) / ext
        v = self.vel / 10.0
        g_body = quat_rotate(np.array([-self.quat[0], -self.quat[1], -self.quat[2], self.quat[3]]),
                             GRAVITY / 9.81)  # attitude cue, IMU-style
        rates = self.omega / 10.0
        if len(self.obs_centers):
            dvec = self.obs_centers - self.pos
            rel = dvec[int(np.argmin(np.linalg.norm(dvec, axis=1)))] / ext
        else:
            rel = np.zeros(3)
        return np.concatenate([rel_goal, v, g_body, rates, rel])

    def _fast_step(self, desired_rates, thrust_cmd):
        a = FAST_DT / (FILTER_TAU + FAST_DT)
        self.filtered_rates += a * (desired_rates - self.filtered_rates)
        self.filtered_thrust += a * (thrust_cmd - self.filtered_thrust)
        torque = np.array([pid.step(des - cur, FAST_DT)
                           for pid, des, cur in zip(self.pids, self.filtered_rates, self.omega)])
        inertia = np.array([IXX, IYY, IZZ])
        self.omega += FAST_DT * (torque - np.cross(self.omega, inertia * self.omega)) / inertia
        w = self.omega
        self.quat = self.quat + 0.5 * FAST_DT * quat_mul(self.quat, np.array([w[0], w[1], w[2], 0.0]))
        self.quat /= np.linalg.norm(self.quat)
        thrust_world = quat_rotate(self.quat, np.array([0.0, self.filtered_thrust, 0.0]))
        self.vel += FAST_DT * (thrust_world / MASS + GRAVITY)
        self.pos += FAST_DT * self.vel

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        desired_rates = a[:3] * MAX_RATES
        thrust_cmd = (a[3] + 1.0) / 2.0 * MAX_THRUST
        noise = self.rng.normal(0, self.dist.dynamics_noise, 3)
        for _ in range(FAST_PER_POLICY):
            self._fast_step(desired_rates + noise * 5.0, thrust_cmd)
        self.steps += 1
        dist = np.linalg.norm(self.goal - self.pos)
        reward = (self.prev_dist - dist) - 0.01
        self.prev_dist = dist
        done = False
        if self.pos[1] < 0.05 or (len(self.obs_centers) and
                np.any(np.linalg.norm(self.obs_centers - self.pos, axis=1) < self.obs_radii + 0.3)):
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


def make_quad_factory(distribution, max_steps=200):
    def factory(seed):
        return QuadNavEnv(distribution, seed=seed, max_steps=max_steps)
    return factory
