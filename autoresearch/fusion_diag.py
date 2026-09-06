import json, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame, cloud, icp_p2p, rot_from_yaw_pitch
from sensors.ekf import ESKF, quat_mul, R_of
from sensors.imu import SimIMU
from sensors.specs.mpu9250 import MPU9250_SPEC
from sensors.base import SimEnvironment
from fusion_v0 import rot_to_quat, NOISE, G

def run(dep, meta, K, ideal=False, att_update=True, dt=0.1, seed=0):
    f, cx, cy = K
    n = len(dep)
    gt_p = meta[:, 1:4]
    gt_q = [rot_to_quat(rot_from_yaw_pitch(meta[i,4], meta[i,5])) for i in range(n)]
    kf = ESKF(NOISE)
    kf.p = gt_p[0].copy(); kf.q = gt_q[0].copy(); kf.v = (gt_p[1]-gt_p[0])/dt
    imu = SimIMU(MPU9250_SPEC, seed=seed); env = SimEnvironment()
    est = [gt_p[0].copy()]
    for i in range(1, n):
        t = i*dt; q_im1 = kf.q.copy()
        dp1 = (gt_p[i]-gt_p[i-1])/dt
        dp0 = (gt_p[i-1]-gt_p[i-2])/dt if i>1 else dp1
        a_world = (dp1-dp0)/dt
        qp, qc = gt_q[i-1], gt_q[i]
        dq = quat_mul(np.array([-qp[0],-qp[1],-qp[2],qp[3]]), qc)
        if dq[3] < 0: dq = -dq
        omega_body = 2*dq[:3]/dt
        sf = a_world - G
        if ideal:
            gm, am = omega_body.copy(), sf.copy()
        else:
            meas = imu.sample(t, dt, dict(omega=omega_body, alpha=np.zeros(3), quat=qc,
                                          thrust_world=sf*0.595, mass=0.595), env, throttle=0.45)
            gm, am = meas.channels["gyro"], meas.channels["accel"]
        kf.predict(gm, am, dt)
        zp = dep[i-1]/1000.0; zc = dep[i]/1000.0
        Pp = cloud(zp, f, cx, cy); Pc = cloud(zc, f, cx, cy)
        src = Pc[(zc>0.15)&(zc<60.0)][::2]
        R_pc, t_pc = icp_p2p(src, zp, Pp, f, cx, cy)
        z_pos = est[-1] + R_of(q_im1) @ t_pc
        kf.update_position(z_pos, np.eye(3)*0.30**2)
        if att_update:
            kf.update_attitude(quat_mul(q_im1, rot_to_quat(R_pc)), np.eye(3)*0.10**2)
        est.append(kf.p.copy())
    est = np.array(est)
    return float(np.sqrt(np.mean(np.sum((est-gt_p)**2,1))))

d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
dep, meta = d["depth"], d["meta"]; m = meta[:,0]==1000
print("A ideal-IMU pos-only     ATE %.3f" % run(dep[m], meta[m], K, ideal=True, att_update=False))
print("B ideal-IMU pos+att      ATE %.3f" % run(dep[m], meta[m], K, ideal=True, att_update=True))
print("C corrupted pos-only     ATE %.3f" % run(dep[m], meta[m], K, ideal=False, att_update=False))
print("D corrupted pos+att      ATE %.3f" % run(dep[m], meta[m], K, ideal=False, att_update=True))
