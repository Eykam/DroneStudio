"""Phase 2f: confidence-aware VO correction noise in fusion_frontend.
v0 inflated R blindly (0.25^2 * frame_idx) even when the frontend solved
NOTHING in the window (identity-padded chain) - fusion trusted drifted
zero-motion poses. Here R scales with actual solve quality per window:
  R_pos = base^2 * i * (1 + miss_w * unsolved_frac) * (1 + resid_w * med_resid)
Heuristic weights (miss_w=4, resid_w=20) - marked [model], not tuned
beyond sanity on the two eval scenes (overfit risk acknowledged).
"""
import sys
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import json
import numpy as np
from vis_vo2 import cam_frame, rot_from_yaw_pitch
from vis_frontend import gray, shi_tomasi, pyr_track, unproject, kabsch_trimmed
from fusion_frontend import run, FrontendOdom
import fusion_frontend as ff


class ConfFrontend(FrontendOdom):
    """Adds per-step solve metadata for confidence-aware R."""
    def __init__(self, K, max_feats=300):
        super().__init__(K, max_feats)
        self.last_solved = False
        self.last_resid = 1.0
        self.last_inl = 0

    def step(self, rgb_prev, rgb_cur, z_prev, z_cur):
        g0, g1 = gray(rgb_prev), gray(rgb_cur)
        if self.feats is None or len(self.feats) < 120:
            self.feats = shi_tomasi(g0)
        T = np.eye(4)
        self.total += 1
        self.last_solved = False
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
                R, t, inl = kabsch_trimmed(P, Q)
                resid = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
                self.last_resid = float(np.median(resid[inl])) if inl.any() else 1.0
                self.last_inl = int(inl.sum())
                T[:3, :3] = R.T
                T[:3, 3] = -R.T @ t
                self.solved += 1
                self.last_solved = True
            self.feats = tr[ok] if ok.sum() else None
        return T


def run_conf(rgb, dep, meta, K, miss_w=4.0, resid_w=20.0, dt=0.1, seed=0):
    """fusion_frontend.run with confidence-aware correction R."""
    from sensors.ekf import ESKF, quat_mul, R_of
    from sensors.imu import SimIMU
    from sensors.tof import SimToF
    from sensors.specs.mpu9250 import MPU9250_SPEC
    from sensors.specs.vl53l9cx import VL53L9CX_SPEC
    from sensors.base import SimEnvironment
    from fusion_v0 import rot_to_quat, NOISE, G
    f, cx, cy = K
    n = len(dep)
    gt_p = meta[:, 1:4]
    gt_q = [rot_to_quat(rot_from_yaw_pitch(meta[i, 4], meta[i, 5])) for i in range(n)]
    kf = ESKF(NOISE); kf.p = gt_p[0].copy(); kf.q = gt_q[0].copy(); kf.v = (gt_p[1] - gt_p[0]) / dt
    imu = SimIMU(MPU9250_SPEC, seed=seed); env = SimEnvironment()
    tof = SimToF(VL53L9CX_SPEC, mode="room_mapping", seed=seed + 7)
    fe = ConfFrontend(K)
    def cast_ground(o, d):
        if d[1] >= -1e-6: return None
        tt = -o[1] / d[1]
        return float(tt) if tt > 0 else None
    est = [gt_p[0].copy()]
    T_vo = np.eye(4); T_vo[:3, :3] = rot_from_yaw_pitch(meta[0, 4], meta[0, 5]); T_vo[:3, 3] = gt_p[0]
    last_vo = 0
    win_unsolved, win_resid = 0, []
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
                if tilt <= np.deg2rad(15):
                    kf.update_ground_range(rng[j], dirs_b[j], 0.01)
        # VO every frame in v0 (vo_every=1) - chain + confidence R
        T_chain = np.eye(4)
        T_rel = fe.step(rgb[i-1], rgb[i], dep[i-1], dep[i])
        T_chain = T_chain @ T_rel
        win_unsolved += 0 if fe.last_solved else 1
        if fe.last_solved: win_resid.append(fe.last_resid)
        T_vo = T_vo @ T_chain
        unsolved_frac = win_unsolved / max(i - last_vo, 1)
        med_res = float(np.median(win_resid)) if win_resid else 1.0
        scale = (1.0 + miss_w * unsolved_frac) * (1.0 + resid_w * med_res)
        kf.update_position(T_vo[:3, 3].copy(), np.eye(3) * (0.25 ** 2 * i * scale))
        kf.update_attitude(rot_to_quat(T_vo[:3, :3]), np.eye(3) * (0.02 ** 2 * i * scale))
        est.append(kf.p.copy())
    est = np.array(est)
    rpe = float(np.sqrt(np.mean([np.sum(((est[j] - est[j-1]) - (gt_p[j] - gt_p[j-1])) ** 2)
                                 for j in range(1, n)])))
    ate = float(np.sqrt(np.mean(np.sum((est - gt_p) ** 2, 1))))
    return ate, rpe, fe.solved / max(fe.total, 1)


if __name__ == "__main__":
    d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
    intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
    rgb, dep, meta = d["rgb"], d["depth"], d["meta"]
    for sid in [1000, 1001]:
        m = meta[:, 0] == sid
        ate0, rpe0, fr0 = run(rgb[m], dep[m], meta[m], K, vo="frontend", use_tof=True, vo_every=1)
        print(f"scene {sid} blind-R   ATE {ate0:7.3f} RPE {rpe0:.3f} solved {fr0:.2f}", flush=True)
        ate1, rpe1, fr1 = run_conf(rgb[m], dep[m], meta[m], K)
        print(f"scene {sid} conf-R    ATE {ate1:7.3f} RPE {rpe1:.3f} solved {fr1:.2f}", flush=True)
