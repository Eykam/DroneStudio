"""Phase 2 fusion v0: ESKF (simulated MPU-9250) + VO relative-pose corrections.

Data: trajectory npz from vis_gen_trajectory.py (GT poses per frame).
IMU truth from finite differences of GT poses at the frame rate (kinematic
data -> 10Hz-class IMU; the 500Hz SimIMU path runs inside the live env later).
VO: icp_p2p per consecutive rendered frames (GT depth) -> relative SE3.
EKF: predict from simulated IMU; corrections as frozen-previous-estimate
increment measurements (position + attitude), first-order loose coupling.

Metrics per scene: fused ATE/RPE vs GT; baselines: VO-alone (chained),
IMU-alone (dead reckoning).
"""
import argparse, json, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame, cloud, icp_p2p, rot_from_yaw_pitch
from sensors.ekf import ESKF, quat_mul, quat_from_rotvec, quat_rotate, R_of
from sensors.imu import SimIMU
from sensors.specs.mpu9250 import MPU9250_SPEC
from sensors.base import SimEnvironment

G = np.array([0.0, -9.81, 0.0])
NOISE = dict(gyro_nd=np.deg2rad(0.01), accel_nd=300e-6*9.81,
             gyro_bias_rw=np.deg2rad(1.0)/np.sqrt(300),
             accel_bias_rw=0.5e-3*9.81/np.sqrt(300))

def rot_to_quat(Rm):
    t = np.trace(Rm)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        return np.array([(Rm[2,1]-Rm[1,2])/s, (Rm[0,2]-Rm[2,0])/s,
                         (Rm[1,0]-Rm[0,1])/s, s/4])
    i = int(np.argmax([Rm[0,0], Rm[1,1], Rm[2,2]]))
    j, k = (i+1) % 3, (i+2) % 3
    s = np.sqrt(max(1e-12, Rm[i,i] - Rm[j,j] - Rm[k,k] + 1)) * 2
    q = np.zeros(4)
    q[i] = s / 4
    q[j] = (Rm[j,i] + Rm[i,j]) / s
    q[k] = (Rm[k,i] + Rm[i,k]) / s
    q[3] = (Rm[k,j] - Rm[j,k]) / s
    return q / np.linalg.norm(q)

def run_scene(dep, meta, K, dt=0.1, vo_sigma=0.30, vo_att_sigma=0.10, seed=0):
    f, cx, cy = K
    n = len(dep)
    gt_p = meta[:, 1:4]
    gt_R = [rot_from_yaw_pitch(meta[i, 4], meta[i, 5]) for i in range(n)]
    gt_q = [rot_to_quat(Rm) for Rm in gt_R]

    kf = ESKF(NOISE)
    kf.p = gt_p[0].copy()
    kf.q = gt_q[0].copy()
    kf.v = (gt_p[1] - gt_p[0]) / dt
    imu = SimIMU(MPU9250_SPEC, seed=seed)
    env = SimEnvironment()

    est = [gt_p[0].copy()]
    vo_est = [gt_p[0].copy()]
    T_vo = np.eye(4); T_vo[:3, :3] = gt_R[0]; T_vo[:3, 3] = gt_p[0]
    imu_est = [gt_p[0].copy()]
    kf_dead = ESKF(NOISE)
    kf_dead.p, kf_dead.q, kf_dead.v = gt_p[0].copy(), gt_q[0].copy(), kf.v.copy()
    rpe_f, rpe_v, rpe_i = [], [], []

    for i in range(1, n):
        t = i * dt
        q_im1 = kf.q.copy()          # attitude estimate at end of frame i-1 (pre-predict)
        # ideal IMU from finite differences of GT
        dp1 = (gt_p[i] - gt_p[i-1]) / dt
        dp0 = (gt_p[i-1] - gt_p[i-2]) / dt if i > 1 else dp1
        a_world = (dp1 - dp0) / dt
        q_prev, q_cur = gt_q[i-1], gt_q[i]
        dq = quat_mul(np.array([-q_prev[0], -q_prev[1], -q_prev[2], q_prev[3]]), q_cur)
        if dq[3] < 0: dq = -dq
        omega_body = 2 * dq[:3] / dt
        qinv = np.array([-q_cur[0], -q_cur[1], -q_cur[2], q_cur[3]])
        sf_world = a_world - G
        throttle = 0.45
        ws = dict(omega=omega_body, alpha=np.zeros(3), quat=q_cur,
                  thrust_world=sf_world * 0.595, mass=0.595)
        meas = imu.sample(t, dt, ws, env, throttle=throttle)
        gm, am = meas.channels["gyro"], meas.channels["accel"]
        kf.predict(gm, am, dt)
        kf_dead.predict(gm, am, dt)
        imu_est.append(kf_dead.p.copy())

        # VO increment
        zp = dep[i-1] / 1000.0; zc = dep[i] / 1000.0
        Pp = cloud(zp, f, cx, cy); Pc = cloud(zc, f, cx, cy)
        src = Pc[(zc > 0.15) & (zc < 60.0)][::2]
        R_pc, t_pc = icp_p2p(src, zp, Pp, f, cx, cy)
        T_rel = np.eye(4); T_rel[:3, :3] = R_pc; T_rel[:3, 3] = t_pc
        T_vo = T_vo @ T_rel
        vo_est.append(T_vo[:3, 3].copy())

        # frozen-previous increment measurements
        p_prev = est[-1].copy()
        z_pos = p_prev + R_of(q_im1) @ t_pc
        kf.update_position(z_pos, np.eye(3) * vo_sigma ** 2)
        dq_meas = rot_to_quat(R_pc)
        q_meas = quat_mul(q_im1, dq_meas)
        kf.update_attitude(q_meas, np.eye(3) * vo_att_sigma ** 2)
        est.append(kf.p.copy())

        gtd = gt_p[i] - gt_p[i-1]
        rpe_f.append(np.linalg.norm((est[-1] - est[-2]) - gtd))
        rpe_v.append(np.linalg.norm((vo_est[-1] - vo_est[-2]) - gtd))
        rpe_i.append(np.linalg.norm((imu_est[-1] - imu_est[-2]) - gtd))

    est = np.array(est); vo_est = np.array(vo_est); imu_est = np.array(imu_est)
    def ate(x): return float(np.sqrt(np.mean(np.sum((x - gt_p) ** 2, 1))))
    def rpe(x): return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    return {"ate_fused": ate(est), "ate_vo": ate(vo_est), "ate_imu": ate(imu_est),
            "rpe_fused": rpe(rpe_f), "rpe_vo": rpe(rpe_v), "rpe_imu": rpe(rpe_i),
            "len": float(np.sum(np.linalg.norm(np.diff(gt_p, axis=0), axis=1)))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--scenes", type=int, default=4)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    intr = json.loads(str(d["intrinsics"]))
    K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
    dep, meta = d["depth"], d["meta"]
    sids = sorted(set(meta[:, 0].astype(int)))[:a.scenes]
    out = []
    for j, sid in enumerate(sids):
        m = meta[:, 0] == sid
        r = run_scene(dep[m], meta[m], K, seed=j)
        r["scene"] = int(sid)
        out.append(r)
        print(f"scene {sid}: ATE fused {r['ate_fused']:.3f} | vo {r['ate_vo']:.3f} | imu {r['ate_imu']:.3f}   "
              f"RPE fused {r['rpe_fused']:.3f} | vo {r['rpe_vo']:.3f} | imu {r['rpe_imu']:.3f}  (len {r['len']:.0f}m)", flush=True)
    if out:
        print("MEAN  ATE fused %.3f | vo %.3f | imu %.3f   RPE fused %.3f | vo %.3f | imu %.3f" % (
            np.mean([o["ate_fused"] for o in out]), np.mean([o["ate_vo"] for o in out]),
            np.mean([o["ate_imu"] for o in out]), np.mean([o["rpe_fused"] for o in out]),
            np.mean([o["rpe_vo"] for o in out]), np.mean([o["rpe_imu"] for o in out])))
    with open(a.npz.replace(".npz", "_fusion_v0.json"), "w") as fh:
        json.dump(out, fh, indent=1)

if __name__ == "__main__":
    main()
