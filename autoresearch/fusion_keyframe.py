"""Phase 2j: keyframe-anchored corrections (long-baseline v1).
Per-frame relatives compound noise; a few catastrophic segments dominate
ATE (Phase 2i). Here features track continuously and the ESKF is
corrected only at keyframes, using tracks that survived the whole
window: a longer-baseline, better-conditioned Kabsch that averages out
per-frame tracking noise. Fixed window K=8 frames [model v1], min 12
anchored tracks; IMU coasts between corrections (graceful per 2b).
Full bundle adjustment is the follow-on; this is the pose-graph's
averaging benefit without the optimizer.
"""
import sys, json
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame, rot_from_yaw_pitch
from vis_frontend import gray, shi_tomasi, pyr_track, unproject, kabsch_trimmed


class KeyframeOdom:
    def __init__(self, K, kwin=8, max_feats=300):
        self.f, self.cx, self.cy = K
        self.kwin = kwin
        self.max_feats = max_feats
        self.feats = None       # (N,2) current coords
        self.anchor_xy = None   # (N,2) coords at current keyframe (nan = unanchored)
        self.anchor_z = None    # (N,) depth at keyframe (nan)
        self.age = 0
        self.n_corr = 0

    def _reseed(self, g):
        self.feats = shi_tomasi(g, self.max_feats)
        n = len(self.feats)
        self.anchor_xy = np.full((n, 2), np.nan)
        self.anchor_z = np.full(n, np.nan)

    def _topup(self, g):
        # Add fresh Shi-Tomasi points as UNANCHORED; never wipe existing
        # tracks or their keyframe anchors (wipes killed every window solve).
        fresh = shi_tomasi(g, self.max_feats)
        if self.feats is None or len(self.feats) == 0:
            self.feats = fresh
            self.anchor_xy = np.full((len(fresh), 2), np.nan)
            self.anchor_z = np.full(len(fresh), np.nan)
            return
        keep = []
        occ = set(map(tuple, np.round(self.feats / 2).astype(int)))
        for p in fresh:
            if tuple(np.round(p / 2).astype(int)) not in occ:
                keep.append(p)
        if not keep:
            return
        keep = np.asarray(keep, dtype=self.feats.dtype)
        self.feats = np.vstack([self.feats, keep])
        self.anchor_xy = np.vstack([self.anchor_xy, np.full((len(keep), 2), np.nan)])
        self.anchor_z = np.concatenate([self.anchor_z, np.full(len(keep), np.nan)])

    def step(self, rgb_prev, rgb_cur, z_prev, z_cur):
        g0, g1 = gray(rgb_prev), gray(rgb_cur)
        if self.feats is None:
            self._reseed(g0)
        tr, ok = pyr_track(g0, g1, self.feats)
        self.feats = tr[ok] if ok.sum() else None
        self.anchor_xy = self.anchor_xy[ok] if ok.sum() else None
        self.anchor_z = self.anchor_z[ok] if ok.sum() else None
        self.age += 1
        out = None
        if self.feats is not None and self.age >= self.kwin:
            anchored = np.isfinite(self.anchor_z)
            y1 = np.clip(np.round(self.feats[:, 1]).astype(int), 0, z_cur.shape[0] - 1)
            x1 = np.clip(np.round(self.feats[:, 0]).astype(int), 0, z_cur.shape[1] - 1)
            zc = z_cur[y1, x1].astype(np.float64)
            valid = anchored & (self.anchor_z > 150) & (self.anchor_z < 60000) & (zc > 150) & (zc < 60000)
            if valid.sum() >= 12:
                P = unproject(self.anchor_xy[valid], self.anchor_z[valid], self.f, self.cx, self.cy)
                Q = unproject(self.feats[valid], zc[valid], self.f, self.cx, self.cy)
                R, t, inl = kabsch_trimmed(P, Q)
                # Kabsch maps keyframe->cur; chain needs cur->keyframe.
                out = np.eye(4)
                out[:3, :3] = R.T
                out[:3, 3] = -R.T @ t
                self.n_corr += 1
            # establish/re-establish the keyframe EVERY window, solve or not -
            # otherwise the first anchor can never exist (chicken-and-egg).
            self.anchor_xy = self.feats.copy()
            self.anchor_z = zc
            self.age = 0
        if self.feats is None or len(self.feats) < 120:
            self._topup(g1)
        return out


def run_kf(rgb, dep, meta, K, kwin=4, dt=0.1, seed=0):
    from sensors.ekf import ESKF, quat_mul, R_of
    from sensors.imu import SimIMU
    from sensors.tof import SimToF
    from sensors.specs.mpu9250 import MPU9250_SPEC
    from sensors.specs.vl53l9cx import VL53L9CX_SPEC
    from sensors.base import SimEnvironment
    from fusion_v0 import rot_to_quat, NOISE, G
    n = len(dep)
    gt_p = meta[:, 1:4]
    gt_q = [rot_to_quat(rot_from_yaw_pitch(meta[i, 4], meta[i, 5])) for i in range(n)]
    kf = ESKF(NOISE); kf.p = gt_p[0].copy(); kf.q = gt_q[0].copy(); kf.v = (gt_p[1] - gt_p[0]) / dt
    imu = SimIMU(MPU9250_SPEC, seed=seed); env = SimEnvironment()
    tof = SimToF(VL53L9CX_SPEC, mode="room_mapping", seed=seed + 7)
    kfo = KeyframeOdom(K, kwin)
    def cast_ground(o, d):
        if d[1] >= -1e-6: return None
        tt = -o[1] / d[1]
        return float(tt) if tt > 0 else None
    est = [gt_p[0].copy()]
    T_vo = np.eye(4); T_vo[:3, :3] = rot_from_yaw_pitch(meta[0, 4], meta[0, 5]); T_vo[:3, 3] = gt_p[0]
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
        tm = tof.scan(t, dict(quat=gt_q[i], origin=gt_p[i]), env, cast_ground)
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
        T_rel = kfo.step(rgb[i-1], rgb[i], dep[i-1], dep[i])
        if T_rel is None:
            T_rel = np.eye(4)
        T_vo = T_vo @ T_rel
        kf.update_position(T_vo[:3, 3].copy(), np.eye(3) * (0.25 ** 2 * i))
        kf.update_attitude(rot_to_quat(T_vo[:3, :3]), np.eye(3) * (0.02 ** 2 * i))
        est.append(kf.p.copy())
    est = np.array(est)
    rpe = float(np.sqrt(np.mean([np.sum(((est[j] - est[j-1]) - (gt_p[j] - gt_p[j-1])) ** 2)
                                 for j in range(1, n)])))
    ate = float(np.sqrt(np.mean(np.sum((est - gt_p) ** 2, 1))))
    return ate, rpe, kfo.n_corr


if __name__ == "__main__":
    for npz, scenes in [("/workspace/vision_model/traj/traj_s13_o2000.npz", None),
                        ("/workspace/vision_model/traj/traj_s12_o1000.npz", [1000, 1001])]:
        d = np.load(npz, allow_pickle=True)
        intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
        rgb, dep, meta = d["rgb"], d["depth"], d["meta"]
        sids = scenes or sorted(set(meta[:, 0].astype(int)))
        for sid in sids:
            m = meta[:, 0] == sid
            ate, rpe, nc = run_kf(rgb[m], dep[m], meta[m], K)
            print(f"{sid}: keyframe ATE {ate:7.3f} RPE {rpe:.3f} corrections {nc}", flush=True)
