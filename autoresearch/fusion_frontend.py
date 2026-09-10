"""Phase 2e: ESKF + feature-frontend corrections vs ICP corrections.
Same harness as fusion_tof.py (chained loosely-coupled form - the sound
one): the VO source is the only variable. Frontend = vis_frontend
(Shi-Tomasi + LK + GT-depth Kabsch). Baseline = vis_vo2 icp_p2p.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame, cloud, icp_p2p, rot_from_yaw_pitch
from vis_frontend import gray, shi_tomasi, pyr_track, unproject, kabsch_trimmed
from sensors.ekf import ESKF, quat_mul, R_of
from sensors.imu import SimIMU
from sensors.tof import SimToF
from sensors.specs.mpu9250 import MPU9250_SPEC
from sensors.specs.vl53l9cx import VL53L9CX_SPEC
from sensors.base import SimEnvironment
from fusion_v0 import rot_to_quat, NOISE, G


class FrontendOdom:
    """Sequential feature-frontend odometry: T_rel per frame pair in the
    vis_vo2 chain convention (cur->prev), identity when unsolved."""
    def __init__(self, K, max_feats=300):
        self.f, self.cx, self.cy = K
        self.feats = None
        self.solved = 0
        self.total = 0

    def step(self, rgb_prev, rgb_cur, z_prev, z_cur):
        g0, g1 = gray(rgb_prev), gray(rgb_cur)
        if self.feats is None or len(self.feats) < 120:
            self.feats = shi_tomasi(g0)
        T = np.eye(4)
        self.total += 1
        if len(self.feats) >= 8:
            tr, ok = pyr_track(g0, g1, self.feats)
            y0 = np.clip(self.feats[:, 1].astype(int), 0, z_prev.shape[0] - 1)
            x0 = np.clip(self.feats[:, 0].astype(int), 0, z_prev.shape[1] - 1)
            z0 = z_prev[y0, x0].astype(np.float64)
            y1 = np.clip(np.round(tr[:, 1]).astype(int), 0, z_cur.shape[0] - 1)
            x1 = np.clip(np.round(tr[:, 0]).astype(int), 0, z_cur.shape[1] - 1)
            z1 = z_cur[y1, x1].astype(np.float64)
            good = ok & (z0 > 150) & (z0 < 60000) & (z1 > 150) & (z1 < 60000)
            if good.sum() >= 8:
                P = unproject(self.feats[good], z0[good], self.f, self.cx, self.cy)
                Q = unproject(tr[good], z1[good], self.f, self.cx, self.cy)
                R, t, _ = kabsch_trimmed(P, Q)
                T[:3, :3] = R.T
                T[:3, 3] = -R.T @ t
                self.solved += 1
            self.feats = tr[ok] if ok.sum() else None
        return T


def run(rgb, dep, meta, K, vo="frontend", use_tof=True, vo_every=1, dt=0.1,
        seed=0, tof_mode="lowtilt"):
    f, cx, cy = K
    n = len(dep)
    gt_p = meta[:, 1:4]
    gt_q = [rot_to_quat(rot_from_yaw_pitch(meta[i, 4], meta[i, 5])) for i in range(n)]
    kf = ESKF(NOISE); kf.p = gt_p[0].copy(); kf.q = gt_q[0].copy(); kf.v = (gt_p[1] - gt_p[0]) / dt
    imu = SimIMU(MPU9250_SPEC, seed=seed); env = SimEnvironment()
    tof = SimToF(VL53L9CX_SPEC, mode="room_mapping", seed=seed + 7)
    fe = FrontendOdom(K)
    def cast_ground(o, d):
        if d[1] >= -1e-6: return None
        tt = -o[1] / d[1]
        return float(tt) if tt > 0 else None
    est = [gt_p[0].copy()]
    T_vo = np.eye(4); T_vo[:3, :3] = rot_from_yaw_pitch(meta[0, 4], meta[0, 5]); T_vo[:3, 3] = gt_p[0]
    last_vo = 0
    for i in range(1, n):
        t = i * dt
        dp1 = (gt_p[i] - gt_p[i-1]) / dt
        dp0 = (gt_p[i-1] - gt_p[i-2]) / dt if i > 1 else dp1
        a_world = (dp1 - dp0) / dt
        qp, qc = gt_q[i-1], gt_q[i]
        dq = quat_mul(np.array([-qp[0], -qp[1], -qp[2], qp[3]]), qc)
        if dq[3] < 0: dq = -dq
        omega_body = 2 * dq[:3] / dt
        sf = a_world - G
        meas = imu.sample(t, dt, dict(omega=omega_body, alpha=np.zeros(3), quat=qc,
                                      thrust_world=sf * 0.595, mass=0.595), env, throttle=0.45)
        kf.predict(meas.channels["gyro"], meas.channels["accel"], dt)
        if use_tof:
            ws_t = dict(quat=gt_q[i], origin=gt_p[i])
            tm = tof.scan(t, ws_t, env, cast_ground)
            if tm is not None:
                rng = tm.channels["ranges"].ravel(); st = tm.channels["status"].ravel()
                dirs_b = tof._dirs_sensor
                okz = np.where(st == 0)[0]
                if len(okz):
                    j = okz[np.argmin([(R_of(kf.q) @ dirs_b[k])[1] for k in okz])]
                    d_w = R_of(kf.q) @ dirs_b[j]
                    tilt = np.arccos(np.clip(-d_w[1], 0, 1))
                    if not (tof_mode == "lowtilt" and tilt > np.deg2rad(15)):
                        kf.update_ground_range(rng[j], dirs_b[j], 0.01)
        if i % vo_every == 0:
            T_chain = np.eye(4)
            for j in range(last_vo + 1, i + 1):
                if vo == "frontend":
                    T_rel = fe.step(rgb[j-1], rgb[j], dep[j-1], dep[j])
                else:
                    zp = dep[j-1] / 1000.0; zc = dep[j] / 1000.0
                    Pp = cloud(zp, f, cx, cy); Pc = cloud(zc, f, cx, cy)
                    src = Pc[(zc > 0.15) & (zc < 60.0)][::2]
                    R_pc, t_pc = icp_p2p(src, zp, Pp, f, cx, cy)
                    T_rel = np.eye(4); T_rel[:3, :3] = R_pc; T_rel[:3, 3] = t_pc
                T_chain = T_chain @ T_rel
            T_vo = T_vo @ T_chain
            kf.update_position(T_vo[:3, 3].copy(), np.eye(3) * (0.25 ** 2 * i))
            kf.update_attitude(rot_to_quat(T_vo[:3, :3]), np.eye(3) * (0.02 ** 2 * i))
            last_vo = i
        est.append(kf.p.copy())
    est = np.array(est)
    rpe = float(np.sqrt(np.mean([np.sum(((est[j] - est[j-1]) - (gt_p[j] - gt_p[j-1])) ** 2)
                                 for j in range(1, n)])))
    ate = float(np.sqrt(np.mean(np.sum((est - gt_p) ** 2, 1))))
    frac = fe.solved / max(fe.total, 1) if vo == "frontend" else 1.0
    return ate, rpe, frac


if __name__ == "__main__":
    d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
    intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
    rgb, dep, meta = d["rgb"], d["depth"], d["meta"]
    for sid in [1000, 1001]:
        m = meta[:, 0] == sid
        for vo, use_tof, ve in [("icp", True, 1), ("frontend", True, 1),
                                ("frontend", False, 1), ("frontend", True, 10)]:
            ate, rpe, frac = run(rgb[m], dep[m], meta[m], K, vo=vo,
                                 use_tof=use_tof, vo_every=ve)
            print(f"scene {sid} vo={vo:8s} tof={use_tof} vo_every={ve:2d}  "
                  f"ATE {ate:7.3f} RPE {rpe:.3f} solved {frac:.2f}", flush=True)
