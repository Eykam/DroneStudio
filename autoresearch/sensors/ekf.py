"""18-state error-state Kalman filter (ESKF) for IMU-driven estimation.

Nominal state: p (pos), v (vel), q (attitude quat x,y,z,w), b_g, b_a.
Error state (15): dp, dv, dth, db_g, db_a.
Predict from SIMULATED IMU only (never GT). Corrections via measurement
updates (VO position/pose, ToF ranges - wired in the fusion runner).

Convention: world frame = sim (x right, y up, z ... as env_quad); quat
(x,y,z,w) maps body->world via quat_rotate. Gravity g = (0,-9.81,0).
Formulation: Sola, "Quaternion kinematics for the error-state Kalman filter".
"""
import numpy as np

G = np.array([0.0, -9.81, 0.0])

def quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz])

def quat_rotate(q, v):
    u = np.array(q[:3]); w = q[3]
    return 2*np.dot(u, v)*u + (w*w - np.dot(u, u))*v + 2*w*np.cross(u, v)

def quat_from_rotvec(w):
    ang = np.linalg.norm(w)
    if ang < 1e-12:
        return np.array([0., 0., 0., 1.])
    ax = w / ang
    return np.array([ax[0]*np.sin(ang/2), ax[1]*np.sin(ang/2),
                     ax[2]*np.sin(ang/2), np.cos(ang/2)])

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def R_of(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

class ESKF:
    def __init__(self, noise):
        """noise: dict with gyro_nd (rad/s/sqrtHz), accel_nd (m/s2/sqrtHz),
        gyro_bias_rw (rad/s2/sqrtHz), accel_bias_rw (m/s3/sqrtHz)."""
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([0., 0., 0., 1.])
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        # GPS slowly-varying bias as filter states (parent 2026-09-07): single-
        # band GNSS error is dominated by a common-mode, time-correlated bias
        # (troposphere/ephemeris). Modelling it makes it observable during
        # motion (VO/IMU pin the relative trajectory; each fix measures p+b),
        # so the terminal phase can fly relative-nav precision.
        self.bgps = np.zeros(3)
        self.P = np.eye(18) * 1e-3
        self.P[9:12, 9:12] *= 10   # bias uncertainty higher
        self.P[12:15, 12:15] *= 10
        self.P[15:18, 15:18] = np.eye(3) * noise.get("gps_bias_std", 1.0) ** 2
        self.n = noise

    def predict(self, gyro_m, accel_m, dt):
        w = gyro_m - self.bg
        a = accel_m - self.ba
        R = R_of(self.q)
        # nominal integration
        self.p = self.p + self.v * dt + 0.5 * (R @ a + G) * dt * dt
        self.v = self.v + (R @ a + G) * dt
        self.q = quat_mul(self.q, quat_from_rotvec(w * dt))
        self.q /= np.linalg.norm(self.q)
        # error-state propagation
        F = np.eye(18)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R @ skew(a) * dt
        F[3:6, 12:15] = -R * dt
        F[6:9, 9:12] = -np.eye(3) * dt
        # GPS bias: OU mean-reversion, tau ~120s (navigation-message class)
        _tau = self.n.get("gps_bias_tau", 120.0)
        F[15:18, 15:18] = np.eye(3) * np.exp(-dt / _tau)
        # process noise (continuous -> discrete, simple form)
        Qd = np.zeros((18, 18))
        sa = self.n["accel_nd"] ** 2 * dt
        sg = self.n["gyro_nd"] ** 2 * dt
        sba = self.n["accel_bias_rw"] ** 2 * dt
        sbg = self.n["gyro_bias_rw"] ** 2 * dt
        Qd[3:6, 3:6] = np.eye(3) * sa
        Qd[6:9, 6:9] = np.eye(3) * sg
        Qd[9:12, 9:12] = np.eye(3) * sbg
        Qd[12:15, 12:15] = np.eye(3) * sba
        Qd[15:18, 15:18] = np.eye(3) * (self.n.get("gps_bias_std", 1.0) ** 2
                                        * (1.0 - np.exp(-2.0 * dt / _tau)))
        self.P = F @ self.P @ F.T + Qd

    def _inject(self, dx, K, H):
        dp, dv, dth, dbg, dba = dx[0:3], dx[3:6], dx[6:9], dx[9:12], dx[12:15]
        dbgps = dx[15:18]
        self.p += dp
        self.v += dv
        self.q = quat_mul(self.q, quat_from_rotvec(dth))
        self.q /= np.linalg.norm(self.q)
        self.bg += dbg
        self.ba += dba
        self.bgps += dbgps
        # covariance reset (simple form)
        self.P = (np.eye(18) - K @ H) @ self.P

    def update_position(self, z, R_meas):
        """z: measured position (3,), R_meas: 3x3 covariance."""
        H = np.zeros((3, 18))
        H[0:3, 0:3] = np.eye(3)
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        self._inject(K @ (z - self.p), K, H)
        return float(np.trace(S))

    def update_gps(self, z, R_meas):
        """GPS fix: z = p + bgps + noise. The bias-state split is what makes
        terminal relative-nav possible: VO/IMU constrain p between fixes, so
        the filter attributes the slow common-mode offset to bgps."""
        H = np.zeros((3, 18))
        H[0:3, 0:3] = np.eye(3)
        H[0:3, 15:18] = np.eye(3)
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        self._inject(K @ (z - (self.p + self.bgps)), K, H)
        return float(np.trace(S))

    def update_velocity(self, z, sigma):
        """z: measured velocity (3,), isotropic sigma. Used for ZUPT."""
        H = np.zeros((3, 18))
        H[0:3, 3:6] = np.eye(3)
        R = np.eye(3) * sigma ** 2
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self._inject(K @ (z - self.v), K, H)
        return float(np.trace(S))

    def update_attitude(self, q_meas, R_meas):
        """z: measured attitude quat; residual = rotvec of q_est^-1 * q_meas."""
        qe = self.q
        qe_inv = np.array([-qe[0], -qe[1], -qe[2], qe[3]])
        dq = quat_mul(qe_inv, q_meas)
        if dq[3] < 0:
            dq = -dq
        r = 2 * dq[:3]
        H = np.zeros((3, 18))
        H[0:3, 6:9] = np.eye(3)
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        self._inject(K @ r, K, H)


    def update_ground_range(self, r_meas, dir_body, sigma):
        """Rangefinder to ground plane y=0 (ToF altimeter). dir_body: unit ray
        direction in body frame. Linearized H on position and attitude."""
        d_w = R_of(self.q) @ dir_body
        if d_w[1] >= -0.2:
            return None                      # ray not downward enough
        r_pred = self.p[1] / (-d_w[1])
        if r_pred <= 0:
            return None
        H = np.zeros((1, 18))
        H[0, 1] = 1.0 / (-d_w[1])
        # attitude block: d(d_w)/d(theta) via -[d_w]x R (body-frame error state)
        J = -(self.p[1] / (d_w[1] ** 2)) * (skew(d_w) @ R_of(self.q))[1, :]
        H[0, 6:9] = J
        Rm = np.array([[sigma ** 2]])
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        self._inject(K @ np.array([r_meas - r_pred]), K, H)
        return float(r_pred)


    def update_mag(self, b_meas, b_world, sigma_ut):
        """Magnetometer: measured body-frame field vs known world field.
        b_meas (3,) body frame (uT), b_world (3,) world frame (uT).
        Observes the full attitude (dominantly yaw when the field is mostly
        horizontal). Linearized H on the attitude error only."""
        R = R_of(self.q)
        b_pred = R.T @ b_world
        # d(b_body)/d(theta_body) = [b_body]x  (body-frame error state)
        H = np.zeros((3, 18))
        H[0:3, 6:9] = skew(b_pred)
        Rm = np.eye(3) * sigma_ut ** 2
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        self._inject(K @ (b_meas - b_pred), K, H)
        return float(np.linalg.norm(b_meas - b_pred))

    def nees_position(self, p_true):
        e = self.p - p_true
        return float(e @ np.linalg.inv(self.P[0:3, 0:3]) @ e)
