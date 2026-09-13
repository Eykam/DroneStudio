"""Spec-driven IMU noise model + complementary attitude estimator (HiL item 3).

Noise parameters come from sensors/<part>.json (dronestudio.sensor/1) - the
same file the sim, CAD and EE read. Default part: icm42688p (user pick).
"""
import json, math, random

def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw)

def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])

def qrot(q, v):
    qv = (q[1], q[2], q[3])
    t = (2*(qv[1]*v[2]-qv[2]*v[1]), 2*(qv[2]*v[0]-qv[0]*v[2]), 2*(qv[0]*v[1]-qv[1]*v[0]))
    return (v[0] + q[0]*t[0] + (qv[1]*t[2]-qv[2]*t[1]),
            v[1] + q[0]*t[1] + (qv[2]*t[0]-qv[0]*t[2]),
            v[2] + q[0]*t[2] + (qv[0]*t[1]-qv[1]*t[0]))

class ImuNoise:
    """Per-sample white noise (ND * sqrt(fs/2)), random-walk bias, FSR quant."""
    def __init__(self, spec_path, seed=0):
        dyn = json.load(open(spec_path))["dynamics"]["imu"]
        g, a = dyn["gyro"], dyn["accel"]
        self.g_nd = g["noise_density_rad_per_s_rthz"]
        self.a_nd = a["noise_density_m_per_s2_rthz"]
        self.g_bw = g["bias_walk_rad_per_s_rts"]
        self.a_bw = a["bias_walk_m_per_s2_rts"]
        self.g_q = (g.get("range_dps", 2000) * math.pi / 180.0) / 32768.0
        self.a_q = (a.get("range_g", 16) * 9.80665) / 32768.0
        self.rng = random.Random(seed)
        self.bias_g = [0.0, 0.0, 0.0]
        self.bias_a = [0.0, 0.0, 0.0]

    def sample(self, omega, accel, dt):
        fs = 1.0 / max(dt, 1e-4)
        sg = self.g_nd * math.sqrt(fs / 2.0)
        sa = self.a_nd * math.sqrt(fs / 2.0)
        sq = math.sqrt(dt)
        g_out, a_out = [], []
        for i in range(3):
            self.bias_g[i] += self.rng.gauss(0.0, 1.0) * self.g_bw * sq
            self.bias_a[i] += self.rng.gauss(0.0, 1.0) * self.a_bw * sq
            gv = omega[i] + self.bias_g[i] + self.rng.gauss(0.0, sg)
            av = accel[i] + self.bias_a[i] + self.rng.gauss(0.0, sa)
            g_out.append(round(gv / self.g_q) * self.g_q)
            a_out.append(round(av / self.a_q) * self.a_q)
        return g_out, a_out

class AttitudeEstimator:
    """Complementary filter: gyro prediction + accel roll/pitch correction.
    Yaw is uncorrected (no mag feed yet - honest mag-less drift)."""
    def __init__(self, q0=(1.0, 0.0, 0.0, 0.0), k_acc=1.0):
        self.q = q0
        self.k = k_acc

    def update(self, gyro, accel, dt):
        dq = (1.0, gyro[0]*dt/2, gyro[1]*dt/2, gyro[2]*dt/2)
        n = math.sqrt(sum(x*x for x in dq)); dq = tuple(x/n for x in dq)
        self.q = qmul(self.q, dq)
        an = math.sqrt(sum(x*x for x in accel))
        if abs(an - 9.81) < 2.0:  # accel gate: only correct when |a| ~= 1g (rejects maneuver/vibration pollution)

            b = tuple(x/an for x in accel)
            be = qrot(qconj(self.q), (0.0, 1.0, 0.0))
            err = (be[1]*b[2]-be[2]*b[1], be[2]*b[0]-be[0]*b[2], be[0]*b[1]-be[1]*b[0])
            ca = [-self.k * dt * e for e in err]  # dt-scaled (1/s gain); body-frame correction enters q as q * r^-1
            cq = (1.0, ca[0]/2, ca[1]/2, ca[2]/2)
            n = math.sqrt(sum(x*x for x in cq)); cq = tuple(x/n for x in cq)
            self.q = qmul(self.q, cq)
        n = math.sqrt(sum(x*x for x in self.q)); self.q = tuple(x/n for x in self.q)
        return self.q

def att_err_deg(q_est, q_true):
    """Angle of q_err = q_est^-1 * q_true, degrees."""
    qi = qconj(q_est)
    e = qmul(qi, q_true)
    w = max(-1.0, min(1.0, abs(e[0])))
    return math.degrees(2.0 * math.acos(w))
