"""Phase 2m: translation-scale gate (rotgate's sibling) + correction-noise cap.
Autopsy (2m): catastrophic tails on 3000/3005 are translation UNDER-SCALE in
degenerate segments (sky-facing far-content 3000: |Tt| 0.15-0.32 vs gtstep 0.41;
floor-planar forward flight 3005: 0.32-0.48 vs 0.51). Rotation stays sane
(dev 1-1.5 < 3 deg gate) - only scale collapses. IMU velocity at 10Hz is
accurate over one frame (same prior rotgate trusts): gate solves whose |Tt|
disagrees with the filter-implied step. Also: correction noise 0.25**2*i grows
with ABSOLUTE frame index -> filter goes deaf late, when drift needs pulling
back. V2 caps it.
"""
import sys, json
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame, rot_from_yaw_pitch
from fusion_frontend import FrontendOdom
from fusion_rotgate import rot_angle

def run_tg(rgb, dep, meta, K, gate_deg=3.0, dt=0.1, seed=0, tgate=True, noisecap=False):
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
    kf = ESKF(NOISE); kf.p = gt_p[0].copy(); kf.q = gt_q[0].copy(); kf.v = (gt_p[1]-gt_p[0])/dt
    imu = SimIMU(MPU9250_SPEC, seed=seed); env = SimEnvironment()
    tof = SimToF(VL53L9CX_SPEC, mode="room_mapping", seed=seed+7)
    fe = FrontendOdom(K)
    def cast_ground(o, d):
        if d[1] >= -1e-6: return None
        tt = -o[1]/d[1]; return float(tt) if tt > 0 else None
    est = [gt_p[0].copy()]
    T_vo = np.eye(4); T_vo[:3,:3] = rot_from_yaw_pitch(meta[0,4], meta[0,5]); T_vo[:3,3] = gt_p[0]
    q_prev = kf.q.copy(); n_rej = 0; n_trej = 0
    for i in range(1, n):
        t = i*dt
        dp1 = (gt_p[i]-gt_p[i-1])/dt
        dp0 = (gt_p[i-1]-gt_p[i-2])/dt if i > 1 else dp1
        a_world = (dp1-dp0)/dt
        qp, qc = gt_q[i-1], gt_q[i]
        dq = quat_mul(np.array([-qp[0],-qp[1],-qp[2],qp[3]]), qc)
        if dq[3] < 0: dq = -dq
        omega_body = 2*dq[:3]/dt
        sf = a_world - G
        meas = imu.sample(t, dt, dict(omega=omega_body, alpha=np.zeros(3), quat=qc,
                                      thrust_world=sf*0.595, mass=0.595), env, throttle=0.45)
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
        T_rel = fe.step(rgb[i-1], rgb[i], dep[i-1], dep[i])
        R_imu = R_of(q_prev).T @ R_of(kf.q)
        dev = rot_angle(T_rel[:3,:3].T @ R_imu)
        if fe.total > 0 and dev > gate_deg:
            T_rel = np.eye(4); n_rej += 1
        elif tgate and fe.total > 0:
            tmag = float(np.linalg.norm(T_rel[:3, 3]))
            estep = float(np.linalg.norm(kf.v)) * dt
            hi = max(2.5 * estep, 0.8)
            lo = 0.35 * estep if estep > 0.12 else 0.0
            if tmag > hi or tmag < lo:
                T_rel = np.eye(4); n_trej += 1
        T_vo = T_vo @ T_rel
        iidx = min(i, 40) if noisecap else i
        kf.update_position(T_vo[:3,3].copy(), np.eye(3)*(0.25**2*iidx))
        kf.update_attitude(rot_to_quat(T_vo[:3,:3]), np.eye(3)*(0.02**2*iidx))
        q_prev = kf.q.copy(); est.append(kf.p.copy())
    est = np.array(est)
    gt = gt_p
    rpe = float(np.sqrt(np.mean([np.sum(((est[j]-est[j-1])-(gt[j]-gt[j-1]))**2) for j in range(1, n)])))
    ate = float(np.sqrt(np.mean(np.sum((est-gt)**2, 1))))
    return ate, rpe, n_rej, n_trej

if __name__ == "__main__":
    d = np.load("/workspace/vision_model/traj/traj_s14_o3000.npz", allow_pickle=True)
    intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
    rgb, dep_gt, meta = d["rgb"], d["depth"], d["meta"]
    dep_v3 = np.load("/workspace/vision_model/traj/traj_s14_o3000_depv3.npy")
    sids = sorted(set(meta[:, 0].astype(int)))
    for dtag, dep in [("GT", dep_gt), ("V3", dep_v3)]:
        for vtag, tg, nc in [("V1-tgate", True, False), ("V2-tgate+noisecap", True, True)]:
            aa = []
            for sid in sids:
                m = meta[:, 0] == sid
                ate, rpe, nr, ntr = run_tg(rgb[m], dep[m], meta[m], K, tgate=tg, noisecap=nc)
                aa.append(ate)
                print(f"{dtag} {vtag} {sid}: ATE {ate:7.3f} RPE {rpe:.3f} rotrej {nr} transrej {ntr}", flush=True)
            print(f"{dtag} {vtag} MEAN {np.mean(aa):.3f}", flush=True)
