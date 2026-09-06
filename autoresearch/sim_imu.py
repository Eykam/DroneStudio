"""Datasheet-realistic MPU-9250 simulation (PS-MPU-9250A-01 v1.1).

Chain per axis per sample (output at 500Hz, physics-bound):
  true -> lever-arm specific force -> scale/cross-axis -> + bias(t)
       -> + vibration (sampled at output instants: >Nyquist aliases exactly)
       -> DLPF (92Hz class) -> + white noise (calibrated to datasheet RMS)
       -> 16-bit quantization.

Datasheet anchors: gyro 0.1 dps-rms @ DLPF 92Hz; gyro rate NSD 0.01 dps/
sqrt(Hz); gyro initial ZRO +/-5 dps; gyro ZRO temp variation +/-30 dps over
-40..85C; accel NSD 300 ug/sqrt(Hz); cross-axis +/-2%; FS 2000 dps / 16 g.
BIAS DRIFT rates are LITERATURE-CLASS (OU process) - not in the datasheet;
swap in Allan-dev measurements of the real MPU-9250 unit when available.
"""
import numpy as np

MPU9250 = dict(
    gyro_fs_dps=2000.0,
    accel_fs_g=16.0,
    gyro_rms_92hz=0.1,            # dps-rms @ DLPFCFG=2 (datasheet anchor)
    accel_nd_ug=300.0,            # ug/sqrt(Hz) class
    gyro_zro_init_dps=5.0,
    gyro_zro_temp_span_dps=30.0,  # -40..85C envelope, centered 25C
    accel_zgo_init_mg=60.0,       # class value; pin exact from PDF table
    cross_axis=0.02,
    scale_tol=0.01,               # per-axis scale-factor tolerance (class)
    dlpf_hz=92.0,
    bits=16,
    lever_arm_m=0.028,            # CAD-measured IMU lever arm (ChassisManifest)
    # OU bias drift (LITERATURE-CLASS)
    gyro_ou_sigma_dps=1.0, gyro_ou_tau_s=300.0,
    accel_ou_sigma_mg=0.5, accel_ou_tau_s=300.0,
)

def quat_rotate(q, v):
    # env_quad convention: q = (x,y,z,w)
    u = np.array(q[:3]); w = q[3]
    return 2 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2 * w * np.cross(u, v)

class SimIMU:
    def __init__(self, p=MPU9250, seed=0, temp_ambient_c=25.0):
        self.p = p
        self.rng = np.random.default_rng(seed)
        rng = self.rng
        self.gyro_scale = 1.0 + rng.uniform(-p["scale_tol"], p["scale_tol"], 3)
        self.accel_scale = 1.0 + rng.uniform(-p["scale_tol"], p["scale_tol"], 3)
        self.gyro_xa = rng.uniform(-p["cross_axis"], p["cross_axis"], (3, 3))
        np.fill_diagonal(self.gyro_xa, 0.0)
        self.accel_xa = rng.uniform(-p["cross_axis"], p["cross_axis"], (3, 3))
        np.fill_diagonal(self.accel_xa, 0.0)
        self.gyro_b0 = rng.uniform(-p["gyro_zro_init_dps"], p["gyro_zro_init_dps"], 3)
        self.accel_b0 = rng.uniform(-p["accel_zgo_init_mg"], p["accel_zgo_init_mg"], 3) * 9.81 / 1000.0
        # OU bias states
        self.gyro_b = np.zeros(3)
        self.accel_b = np.zeros(3)
        self.t0 = temp_ambient_c
        self.temp = temp_ambient_c
        self._lp_g = None
        self._lp_a = None
        # vibration phases: drawn ONCE (coherent sinusoids)
        self.vib_ph_a = rng.uniform(0, 2 * np.pi, (3, 3))  # 3 harmonics x 3 axes
        self.vib_ph_g = rng.uniform(0, 2 * np.pi, (3, 3))

    def _temperature(self, t):
        return self.t0 + 8.0 * (1.0 - np.exp(-t / 90.0)) + 1.5 * np.sin(2 * np.pi * t / 600.0)

    def _gyro_tempco(self, T):
        dT = np.clip((T - 25.0) / 60.0, -1.0, 1.0)
        shape = dT + 0.35 * dT * abs(dT)
        return (self.p["gyro_zro_temp_span_dps"] / 2.0) * shape  # dps, bounded

    def _vibration(self, t, throttle):
        motor_hz = 80.0 + 220.0 * throttle       # 4S 5in class
        amp_a = 0.4 + 3.0 * throttle             # m/s^2
        amp_g = 0.2 + 1.5 * throttle             # dps
        va = np.zeros(3); vg = np.zeros(3)
        for k in range(3):                        # fundamental, blade pass, 3rd
            f = motor_hz * (k + 1)
            va += (amp_a / (k + 1)) * np.sin(2 * np.pi * f * t + self.vib_ph_a[k])
            vg += (amp_g / (k + 1)) * np.sin(2 * np.pi * f * t + self.vib_ph_g[k])
        va += (0.3 + 1.0 * throttle) * self.rng.normal(0, 1, 3)
        vg += (0.1 + 0.4 * throttle) * self.rng.normal(0, 1, 3)
        return va, vg

    def sample(self, t, dt, omega_true, alpha_true, quat, thrust_world,
               mass, throttle):
        """omega_true rad/s (body), alpha_true rad/s^2 (body),
        quat (x,y,z,w) env convention, thrust_world N (world).
        Returns (gyro rad/s body, accel specific force m/s^2 body)."""
        p = self.p
        r = np.array([p["lever_arm_m"], 0.0, 0.0])
        qinv = np.array([-quat[0], -quat[1], -quat[2], quat[3]])
        f_cg = quat_rotate(qinv, thrust_world / mass)
        f_imu = f_cg + np.cross(alpha_true, r) + np.cross(omega_true, np.cross(omega_true, r))
        g_true = np.rad2deg(omega_true)          # work in dps for gyro chain

        g = (np.eye(3) + self.gyro_xa) @ (self.gyro_scale * g_true)
        a = (np.eye(3) + self.accel_xa) @ (self.accel_scale * f_imu)

        # OU bias update (bounded random walk)
        bg_sig = p["gyro_ou_sigma_dps"]; bg_tau = p["gyro_ou_tau_s"]
        ba_sig = p["accel_ou_sigma_mg"] * 9.81 / 1000.0; ba_tau = p["accel_ou_tau_s"]
        self.gyro_b += -(self.gyro_b / bg_tau) * dt + bg_sig * np.sqrt(2 * dt / bg_tau) * self.rng.normal(0, 1, 3)
        self.accel_b += -(self.accel_b / ba_tau) * dt + ba_sig * np.sqrt(2 * dt / ba_tau) * self.rng.normal(0, 1, 3)

        self.temp = self._temperature(t)
        b_g = self.gyro_b0 + self._gyro_tempco(self.temp) + self.gyro_b
        b_a = self.accel_b0 + self.accel_b

        va, vg = self._vibration(t, throttle)
        g = g + b_g + vg
        a = a + b_a + va

        # DLPF (first-order, 92Hz class)
        al = 1.0 - np.exp(-2 * np.pi * p["dlpf_hz"] * dt)
        self._lp_g = g if self._lp_g is None else self._lp_g + al * (g - self._lp_g)
        self._lp_a = a if self._lp_a is None else self._lp_a + al * (a - self._lp_a)
        g = self._lp_g
        a = self._lp_a

        # white noise: gyro calibrated to datasheet anchor; accel from NSD
        g = g + self.rng.normal(0, p["gyro_rms_92hz"], 3)
        a = a + self.rng.normal(0, p["accel_nd_ug"] * 1e-6 * 9.81 * np.sqrt(1.0 / (2 * dt)), 3)

        # 16-bit quantization
        g_lsb = 2 * p["gyro_fs_dps"] / 2 ** p["bits"]
        a_lsb = 2 * p["accel_fs_g"] * 9.81 / 2 ** p["bits"]
        g = np.round(g / g_lsb) * g_lsb
        a = np.round(a / a_lsb) * a_lsb
        return np.deg2rad(g), a
