import json, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame, cloud, icp_p2p, rot_from_yaw_pitch
from sensors.ekf import ESKF, quat_mul, R_of
from sensors.imu import SimIMU
from sensors.specs.mpu9250 import MPU9250_SPEC
from sensors.base import SimEnvironment
from fusion_v0 import rot_to_quat, NOISE, G

def run(dep, meta, K, mode="chained", vo_sigma=0.25, att_sigma=0.02, dt=0.1, seed=0):
    f, cx, cy = K
    n = len(dep)
    gt_p = meta[:,1:4]
    gt_q = [rot_to_quat(rot_from_yaw_pitch(meta[i,4], meta[i,5])) for i in range(n)]
    kf = ESKF(NOISE); kf.p = gt_p[0].copy(); kf.q = gt_q[0].copy(); kf.v = (gt_p[1]-gt_p[0])/dt
    imu = SimIMU(MPU9250_SPEC, seed=seed); env = SimEnvironment()
    est = [gt_p[0].copy()]
    T_vo = np.eye(4); T_vo[:3,:3] = rot_from_yaw_pitch(meta[0,4], meta[0,5]); T_vo[:3,3] = gt_p[0]
    for i in range(1, n):
        t = i*dt; q_im1 = kf.q.copy(); p_im1 = est[-1].copy()
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
        zp = dep[i-1]/1000.0; zc = dep[i]/1000.0
        Pp = cloud(zp, f, cx, cy); Pc = cloud(zc, f, cx, cy)
        src = Pc[(zc>0.15)&(zc<60.0)][::2]
        R_pc, t_pc = icp_p2p(src, zp, Pp, f, cx, cy)
        T_rel = np.eye(4); T_rel[:3,:3] = R_pc; T_rel[:3,3] = t_pc
        T_vo = T_vo @ T_rel
        if mode == "chained":
            kf.update_position(T_vo[:3,3].copy(), np.eye(3)*(vo_sigma**2 * i))
            kf.update_attitude(rot_to_quat(T_vo[:3,:3]), np.eye(3)*(att_sigma**2 * i))
        else:
            kf.update_position(p_im1 + R_of(q_im1) @ t_pc, np.eye(3)*vo_sigma**2)
            kf.update_attitude(quat_mul(q_im1, rot_to_quat(R_pc)), np.eye(3)*att_sigma**2)
        est.append(kf.p.copy())
    est = np.array(est)
    rpe = np.sqrt(np.mean([np.sum(((est[j]-est[j-1])-(gt_p[j]-gt_p[j-1]))**2) for j in range(1,n)]))
    return float(np.sqrt(np.mean(np.sum((est-gt_p)**2,1)))), float(rpe)

d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
dep, meta = d["depth"], d["meta"]
for sid in [1000, 1001]:
    m = meta[:,0]==sid
    for mode, vs, asg in [("chained",0.25,0.02), ("increment",0.30,0.02), ("increment",0.30,0.005)]:
        ate, rpe = run(dep[m], meta[m], K, mode=mode, vo_sigma=vs, att_sigma=asg)
        print("scene %d %-9s vs=%.3f as=%.3f  ATE %.3f RPE %.3f" % (sid, mode, vs, asg, ate, rpe), flush=True)
