import json, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame, cloud, icp_p2p, rot_from_yaw_pitch
from sensors.ekf import ESKF, quat_mul, R_of
from sensors.imu import SimIMU
from sensors.tof import SimToF
from sensors.specs.mpu9250 import MPU9250_SPEC
from sensors.specs.vl53l9cx import VL53L9CX_SPEC
from sensors.base import SimEnvironment
from fusion_v0 import rot_to_quat, NOISE, G

def run(dep, meta, K, use_tof=True, use_mag=True, use_vo_att=True, dt=0.1, seed=0):
    f, cx, cy = K
    n = len(dep)
    gt_p = meta[:,1:4]
    gt_q = [rot_to_quat(rot_from_yaw_pitch(meta[i,4], meta[i,5])) for i in range(n)]
    kf = ESKF(NOISE); kf.p = gt_p[0].copy(); kf.q = gt_q[0].copy(); kf.v = (gt_p[1]-gt_p[0])/dt
    imu = SimIMU(MPU9250_SPEC, seed=seed); env = SimEnvironment()
    tof = SimToF(VL53L9CX_SPEC, mode="room_mapping", seed=seed+7)
    # altimeter mount (mirror of eval_estimated.py fix 4735402): boresight down,
    # and the nadir gate must use MOUNTED dirs - raw _dirs_sensor bypasses the
    # mount, which left ToF inert in every prior fusion scorecard.
    tof.mount.rot = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    def cast_ground(o, d):
        if d[1] >= -1e-6: return None
        tt = -o[1]/d[1]
        return float(tt) if tt > 0 else None
    est = [gt_p[0].copy()]
    att_errs = []
    T_vo = np.eye(4); T_vo[:3,:3] = rot_from_yaw_pitch(meta[0,4], meta[0,5]); T_vo[:3,3] = gt_p[0]
    last_good = 0
    q_at_lg = kf.q.copy()
    for i in range(1, n):
        t = i*dt
        dp1 = (gt_p[i]-gt_p[i-1])/dt
        dp0 = (gt_p[i-1]-gt_p[i-2])/dt if i>1 else dp1
        a_world = (dp1-dp0)/dt
        qp, qc = gt_q[i-1], gt_q[i]
        dq = quat_mul(np.array([-qp[0],-qp[1],-qp[2],qp[3]]), qc)
        if dq[3] < 0: dq = -dq
        omega_body = 2*dq[:3]/dt
        sf = a_world - G
        meas = imu.sample(t, dt, dict(omega=omega_body, alpha=np.zeros(3), quat=qc,
                                      thrust_world=sf*0.595, mass=0.595), env, throttle=0.45)
        kf.predict(meas.channels["gyro"], meas.channels["accel"], dt)
        if use_tof:
            tm = tof.scan(t, dict(quat=qc, origin=gt_p[i]), env, cast_ground)
            if tm is not None:
                rng = tm.channels["ranges"].ravel(); st = tm.channels["status"].ravel()
                dirs_b = np.stack([tof.mount.rot @ d for d in tof._dirs_sensor])
                ok = np.where(st == 0)[0]
                if len(ok):
                    Rk = R_of(kf.q)
                    j = ok[np.argmin([(Rk @ dirs_b[m_])[1] for m_ in ok])]
                    d_w = Rk @ dirs_b[j]
                    if np.arccos(np.clip(-d_w[1], 0, 1)) <= np.deg2rad(15):
                        r_true = rng[j]
                        sigma = tof.spec["sigma_base_mm"]/1000.0 + tof.spec["sigma_range2"]*r_true*r_true
                        kf.update_ground_range(r_true, dirs_b[j], max(sigma, 0.005))
        # ICP from last GOOD frame (skip-and-bridge bad frames so the chain
        # never ingests a corrupt increment)
        zl = dep[last_good]/1000.0; zc = dep[i]/1000.0
        Pl = cloud(zl, f, cx, cy); Pc = cloud(zc, f, cx, cy)
        src = Pc[(zc>0.15)&(zc<60.0)][::2]
        R_pc, t_pc, fit = icp_p2p(src, zl, Pl, f, cx, cy, ret_fit=True)
        # INNOVATION GATE: VO increment rotation vs gyro-integrated rotation
        # over the same interval (ICP fitness is blind to wrong rotations -
        # ground-plane degeneracy; corr(fit, roterr) = 0.04 measured)
        R_pred_incr = R_of(q_at_lg).T @ R_of(kf.q)
        R_inno = R_pc @ R_pred_incr.T
        inno_deg = np.rad2deg(np.arccos(np.clip((np.trace(R_inno)-1)/2, -1, 1)))
        n_fr = i - last_good
        if inno_deg < 25.0 * n_fr and np.linalg.norm(t_pc) < 2.0 * n_fr:
            T_rel = np.eye(4); T_rel[:3,:3] = R_pc; T_rel[:3,3] = t_pc
            T_vo = T_vo @ T_rel
            sp = 0.25 + 2.0*min(fit, 0.3); sa = max(0.02 + 0.5*min(fit, 0.3), 0.15)  # floor: mag must win the attitude tug-of-war
            kf.update_position(T_vo[:3,3].copy(), np.eye(3)*(sp**2 * i))
            if use_vo_att:
                # chain re-anchoring: the chain is an integrated reference and
                # leaks 2-5deg/frame under-rotation below any per-frame gate.
                # Measure chain-vs-filter disagreement; past 20deg, reset the
                # CHAIN attitude to the (mag-owned) filter estimate, then feed
                # the re-anchored chain attitude back as the update. The leak
                # can no longer integrate without bound.
                q_vo = rot_to_quat(T_vo[:3,:3])
                qe = quat_mul(np.array([-kf.q[0],-kf.q[1],-kf.q[2],kf.q[3]]), q_vo)
                inno = np.rad2deg(2*np.arccos(np.clip(abs(qe[3]),0,1)))
                if inno > 20.0:
                    T_vo[:3,:3] = R_of(kf.q)
                    q_vo = rot_to_quat(T_vo[:3,:3])
                kf.update_attitude(q_vo, np.eye(3)*(sa**2 * i))
            last_good = i
            q_at_lg = kf.q.copy()
        # else: reject entirely - chain holds, EKF coasts on IMU (+ToF)
        if use_mag:
            # simulated AK8963 (MPU-9250 mag on the ee-flight design):
            # body-frame sample of a known constant world field + noise.
            B_world = np.array([20.0, 40.0, 5.0])
            hard_iron = np.array([1.2, -0.8, 0.5])
            b_meas = R_of(qc).T @ B_world + hard_iron + np.random.normal(0, 0.5, 3)
            kf.update_mag(b_meas - hard_iron, B_world, 0.7)
        est.append(kf.p.copy())
        qe = quat_mul(np.array([-kf.q[0],-kf.q[1],-kf.q[2],kf.q[3]]), gt_q[i])
        att_errs.append(np.rad2deg(2*np.arccos(np.clip(abs(qe[3]),0,1))))
    est = np.array(est)
    rpe = np.sqrt(np.mean([np.sum(((est[j]-est[j-1])-(gt_p[j]-gt_p[j-1]))**2) for j in range(1,n)]))
    return (float(np.sqrt(np.mean(np.sum((est-gt_p)**2,1)))),
            float(np.sqrt(np.mean((est[:,1]-gt_p[:,1])**2))),
            float(np.mean(att_errs)), float(np.max(att_errs)), float(rpe))

if __name__ == "__main__":
    d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
    intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
    dep, meta = d["depth"], d["meta"]
    sids = sorted(set(meta[:,0].astype(int)))
    rows = []
    for k, sid in enumerate(sids):
        m = meta[:,0]==sid
        ate, ye, am_, ax_, rpe = run(dep[m], meta[m], K, use_tof=True, use_mag=True, use_vo_att=False, seed=k)
        rows.append(dict(scene=int(sid), ate=ate, yrmse=ye, att_mean=am_, att_max=ax_, rpe=rpe))
        print("scene %d  ATE %7.3f yRMSE %.3f att %.1f/%.1f RPE %.3f" % (sid, ate, ye, am_, ax_, rpe), flush=True)
    print("MEAN  ATE %.3f yRMSE %.3f att %.1f RPE %.3f  (max ATE %.3f)" % (
        np.mean([r["ate"] for r in rows]), np.mean([r["yrmse"] for r in rows]),
        np.mean([r["att_mean"] for r in rows]), np.mean([r["rpe"] for r in rows]),
        max(r["ate"] for r in rows)))
    json.dump(rows, open("/workspace/vision_model/traj/fusion_v1_scorecard.json", "w"), indent=1)
