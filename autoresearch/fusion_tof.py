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

def run(dep, meta, K, use_tof=True, vo_every=1, dt=0.1, seed=0,
        tof_mode="full"):
    f, cx, cy = K
    n = len(dep)
    gt_p = meta[:,1:4]
    gt_q = [rot_to_quat(rot_from_yaw_pitch(meta[i,4], meta[i,5])) for i in range(n)]
    kf = ESKF(NOISE); kf.p = gt_p[0].copy(); kf.q = gt_q[0].copy(); kf.v = (gt_p[1]-gt_p[0])/dt
    imu = SimIMU(MPU9250_SPEC, seed=seed); env = SimEnvironment()
    tof = SimToF(VL53L9CX_SPEC, mode="room_mapping", seed=seed+7)
    down_body = np.array([0.0, -1.0, 0.0])
    def cast_ground(o, d):
        if d[1] >= -1e-6: return None
        tt = -o[1]/d[1]
        return float(tt) if tt > 0 else None
    est = [gt_p[0].copy()]
    T_vo = np.eye(4); T_vo[:3,:3] = rot_from_yaw_pitch(meta[0,4], meta[0,5]); T_vo[:3,3] = gt_p[0]
    last_vo = 0
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
            # downward center-ish zone: pick the zone closest to body-down
            ws_t = dict(quat=gt_q[i], origin=gt_p[i])  # truth generates the measurement
            tm = tof.scan(t, ws_t, env, cast_ground)
            if tm is not None:
                rng = tm.channels["ranges"].ravel(); st = tm.channels["status"].ravel()
                dirs_b = tof._dirs_sensor
                ok = np.where(st == 0)[0]
                if len(ok):
                    # most-downward valid zone
                    j = ok[np.argmin([ (R_of(kf.q) @ dirs_b[k])[1] for k in ok])]
                    d_w = R_of(kf.q) @ dirs_b[j]
                    tilt = np.arccos(np.clip(-d_w[1], 0, 1))
                    if tof_mode == "lowtilt" and tilt > np.deg2rad(15):
                        pass
                    else:
                        r_true = rng[j]
                        sigma = (tof.spec["sigma_base_mm"]/1000.0 +
                                 tof.spec["sigma_range2"]*r_true*r_true*1000.0/1000.0)
                        if tof_mode == "posonly":
                            # classical altimeter: observe p_y only, no att coupling
                            r_pred = kf.p[1] / (-d_w[1])
                            H = np.zeros((1, 15)); H[0, 1] = 1.0 / (-d_w[1])
                            Rm = np.array([[max(sigma, 0.005) ** 2]])
                            S = H @ kf.P @ H.T + Rm
                            Kk = kf.P @ H.T @ np.linalg.inv(S)
                            kf._inject(Kk @ np.array([r_true - r_pred]), Kk, H)
                        else:
                            kf.update_ground_range(r_true, dirs_b[j], max(sigma, 0.005))
        if i % vo_every == 0:
            T_chain = np.eye(4)
            for j in range(last_vo+1, i+1):
                zp = dep[j-1]/1000.0; zc = dep[j]/1000.0
                Pp = cloud(zp, f, cx, cy); Pc = cloud(zc, f, cx, cy)
                src = Pc[(zc>0.15)&(zc<60.0)][::2]
                R_pc, t_pc = icp_p2p(src, zp, Pp, f, cx, cy)
                T_rel = np.eye(4); T_rel[:3,:3] = R_pc; T_rel[:3,3] = t_pc
                T_chain = T_chain @ T_rel
            T_vo = T_vo @ T_chain
            kf.update_position(T_vo[:3,3].copy(), np.eye(3)*(0.25**2 * i))
            kf.update_attitude(rot_to_quat(T_vo[:3,:3]), np.eye(3)*(0.02**2 * i))
            last_vo = i
        est.append(kf.p.copy())
    est = np.array(est)
    rpe = np.sqrt(np.mean([np.sum(((est[j]-est[j-1])-(gt_p[j]-gt_p[j-1]))**2) for j in range(1,n)]))
    y_err = float(np.sqrt(np.mean((est[:,1]-gt_p[:,1])**2)))
    return float(np.sqrt(np.mean(np.sum((est-gt_p)**2,1)))), float(rpe), y_err

d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
dep, meta = d["depth"], d["meta"]
for sid in [1000, 1001]:
    m = meta[:,0]==sid
    for use_tof, ve, tm in [(False,1,"-"), (True,1,"full"), (True,1,"posonly"),
                            (True,1,"lowtilt"), (True,5,"posonly"), (True,5,"lowtilt")]:
        ate, rpe, ye = run(dep[m], meta[m], K, use_tof=use_tof, vo_every=ve,
                           tof_mode="full" if tm=="-" else tm)
        print("scene %d tof=%-8s vo_every=%2d  ATE %7.3f RPE %.3f yRMSE %.3f" % (sid, tm, ve, ate, rpe, ye), flush=True)
