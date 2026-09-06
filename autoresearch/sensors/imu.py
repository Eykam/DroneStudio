"""IMU ideal model + datasheet error model. MPU-9250-class 6-axis."""
import numpy as np
from .base import SimSensor, Mount

def quat_rotate(q, v):
    u = np.array(q[:3]); w = q[3]
    return 2 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2 * w * np.cross(u, v)

class SimIMU(SimSensor):
    def __init__(self, spec, mount=None, seed=0, name=None):
        super().__init__(spec, mount, seed, name)
        rng = self.rng; p = spec
        self.gyro_scale = 1.0 + rng.uniform(-p["scale_tol"], p["scale_tol"], 3)
        self.accel_scale = 1.0 + rng.uniform(-p["scale_tol"], p["scale_tol"], 3)
        self.gyro_xa = rng.uniform(-p["cross_axis"], p["cross_axis"], (3, 3))
        np.fill_diagonal(self.gyro_xa, 0.0)
        self.accel_xa = rng.uniform(-p["cross_axis"], p["cross_axis"], (3, 3))
        np.fill_diagonal(self.accel_xa, 0.0)
        self.gyro_b0 = rng.uniform(-p["gyro_zro_init_dps"], p["gyro_zro_init_dps"], 3)
        self.accel_b0 = rng.uniform(-p["accel_zgo_init_mg"], p["accel_zgo_init_mg"], 3) * 9.81 / 1000.0
        self.gyro_b = np.zeros(3)
        self.accel_b = np.zeros(3)
        self._lp_g = None
        self._lp_a = None
        self.vib_ph_a = rng.uniform(0, 2 * np.pi, (3, 3))
        self.vib_ph_g = rng.uniform(0, 2 * np.pi, (3, 3))

    def ideal(self, ws, env):
        """world_state ws: omega, alpha, quat, thrust_world, mass."""
        r = self.mount.pos
        qinv = np.array([-ws["quat"][0], -ws["quat"][1], -ws["quat"][2], ws["quat"][3]])
        f_cg = quat_rotate(qinv, ws["thrust_world"] / ws["mass"])
        w, al = ws["omega"], ws["alpha"]
        f_imu = f_cg + np.cross(al, r) + np.cross(w, np.cross(w, r))
        # into sensor frame
        return {"gyro_dps": np.rad2deg(self.mount.rot @ w),
                "accel": self.mount.rot @ f_imu}

    def _gyro_tempco(self, T):
        dT = np.clip((T - 25.0) / 60.0, -1.0, 1.0)
        shape = dT + 0.35 * dT * abs(dT)
        return (self.spec["gyro_zro_temp_span_dps"] / 2.0) * shape

    def _vibration(self, t, throttle, dt=0.002):
        # motors near idle/off: no meaningful vibration energy
        if throttle < 0.05:
            return np.zeros(3), np.zeros(3)
        fc = self.spec["dlpf_hz"]
        motor_hz = 80.0 + 220.0 * throttle
        amp_a = 0.4 + 3.0 * throttle
        amp_g = 0.2 + 1.5 * throttle
        va = np.zeros(3); vg = np.zeros(3)
        for k in range(3):                        # fundamental, blade pass, 3rd
            f = motor_hz * (k + 1)
            # anti-alias reality: the DLPF attenuates each harmonic BEFORE the
            # ADC samples; only the filter residual aliases into the output band.
            g_antialias = 1.0 / np.sqrt(1.0 + (f / fc) ** 2)
            va += g_antialias * (amp_a / (k + 1)) * np.sin(2 * np.pi * f * t + self.vib_ph_a[k])
            vg += g_antialias * (amp_g / (k + 1)) * np.sin(2 * np.pi * f * t + self.vib_ph_g[k])
        va += (0.3 + 1.0 * throttle) * self.rng.normal(0, 1, 3)
        vg += (0.1 + 0.4 * throttle) * self.rng.normal(0, 1, 3)
        return va, vg

    def corrupt(self, ideal, env, t, dt, throttle=0.0):
        p = self.spec
        g_true = ideal["gyro_dps"]; a_true = ideal["accel"]
        g = (np.eye(3) + self.gyro_xa) @ (self.gyro_scale * g_true)
        a = (np.eye(3) + self.accel_xa) @ (self.accel_scale * a_true)
        bg_sig = p["gyro_ou_sigma_dps"]; bg_tau = p["gyro_ou_tau_s"]
        ba_sig = p["accel_ou_sigma_mg"] * 9.81 / 1000.0; ba_tau = p["accel_ou_tau_s"]
        self.gyro_b += -(self.gyro_b / bg_tau) * dt + bg_sig * np.sqrt(2 * dt / bg_tau) * self.rng.normal(0, 1, 3)
        self.accel_b += -(self.accel_b / ba_tau) * dt + ba_sig * np.sqrt(2 * dt / ba_tau) * self.rng.normal(0, 1, 3)
        T = env.temperature(t)
        b_g = self.gyro_b0 + self._gyro_tempco(T) + self.gyro_b
        b_a = self.accel_b0 + self.accel_b
        va, vg = self._vibration(t, throttle, dt)
        g = g + b_g + vg
        a = a + b_a + va
        al = 1.0 - np.exp(-2 * np.pi * p["dlpf_hz"] * dt)
        self._lp_g = g if self._lp_g is None else self._lp_g + al * (g - self._lp_g)
        self._lp_a = a if self._lp_a is None else self._lp_a + al * (a - self._lp_a)
        g = self._lp_g + self.rng.normal(0, p["gyro_rms_92hz"], 3)
        a = self._lp_a + self.rng.normal(0, p["accel_nd_ug"] * 1e-6 * 9.81 * np.sqrt(1.0 / (2 * dt)), 3)
        g_lsb = 2 * p["gyro_fs_dps"] / 2 ** p["bits"]
        a_lsb = 2 * p["accel_fs_g"] * 9.81 / 2 ** p["bits"]
        g = np.round(g / g_lsb) * g_lsb
        a = np.round(a / a_lsb) * a_lsb
        return {"gyro": np.deg2rad(g), "accel": a}

    def sample(self, t, dt, world_state, env, throttle=0.0):
        if not self.due(t):
            return None
        from .base import Measurement
        ideal = self.ideal(world_state, env)
        out = self.corrupt(ideal, env, t, dt, throttle)
        return Measurement(self.name, t + self.rng.normal(0, 1e-4),
                           self.spec["rate_hz"], out)

def self_test(spec=None, seconds=60, seed=1, verbose=True):
    """Datasheet anchor check: stationary hover; gyro noise RMS at the 92Hz
    DLPF setting should be ~0.1 dps + attenuated vibration (throttle 0)."""
    from .specs.mpu9250 import MPU9250_SPEC
    from .base import SimEnvironment
    spec = spec or MPU9250_SPEC
    imu = SimIMU(spec, seed=seed)
    env = SimEnvironment()
    m = 0.595
    ws = dict(omega=np.zeros(3), alpha=np.zeros(3),
              quat=np.array([0., 0., 0., 1.]),
              thrust_world=np.array([0., m * 9.81, 0.]), mass=m)
    dt = 1.0 / spec["rate_hz"]
    gs = []
    for i in range(int(seconds / dt)):
        meas = imu.sample(i * dt, dt, ws, env, throttle=0.0)
        if meas is not None:
            gs.append(meas.channels["gyro"])
    gs = np.rad2deg(np.array(gs))
    # successive-difference isolates sensor noise from drift/bias:
    # diff RMS / sqrt(2) recovers the white-noise sigma
    rms = float((np.diff(gs, axis=0).std(axis=0) / np.sqrt(2)).mean())
    ok = 0.06 < rms < 0.16
    if verbose:
        print("self_test: gyro noise RMS %.3f dps (anchor 0.1) %s" % (rms, "PASS" if ok else "FAIL"))
    return ok
