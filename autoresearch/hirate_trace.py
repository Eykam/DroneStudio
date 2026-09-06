import json, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame, cloud, icp_p2p, rot_from_yaw_pitch
from sensors.ekf import ESKF, quat_mul, R_of
from sensors.imu import SimIMU
from sensors.specs.mpu9250 import MPU9250_SPEC
from sensors.base import SimEnvironment
from fusion_v0 import rot_to_quat, NOISE, G
from fusion_hirate import upsample

d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
f, cx, cy = K
dep, meta = d["depth"], d["meta"]; m = meta[:,0]==1001
dep, meta = dep[m], meta[m]
n = len(dep); dt_f = 0.1
gt_p = meta[:,1:4]
gt_q = [rot_to_quat(rot_from_yaw_pitch(meta[i,4], meta[i,5])) for i in range(n)]
for i in range(1, n):
    if np.dot(gt_q[i], gt_q[i-1]) < 0: gt_q[i] = -gt_q[i]
ts, ps, qs = upsample(gt_p, gt_q, dt_f, k=10)
dt = ts[1]-ts[0]
kf = ESKF(NOISE); kf.p = ps[0].copy(); kf.q = qs[0].copy(); kf.v = (ps[1]-ps[0])/dt
imu = SimIMU(MPU9250_SPEC, seed=0); env = SimEnvironment()
T_vo = np.eye(4); T_vo[:3,:3] = rot_from_yaw_pitch(meta[0,4], meta[0,5]); T_vo[:3,3] = gt_p[0]
n_meas = 0
for i in range(1, len(ts)):
    t = ts[i]
    v1 = (ps[i]-ps[i-1])/dt
    v0 = (ps[i-1]-ps[i-2])/dt if i > 1 else v1
    a_world = (v1-v0)/dt
    dq = quat_mul(np.array([-qs[i-1][0],-qs[i-1][1],-qs[i-1][2],qs[i-1][3]]), qs[i])
    if dq[3] < 0: dq = -dq
    omega_body = 2*dq[:3]/dt
    sf = a_world - G
    meas = imu.sample(t, dt, dict(omega=omega_body, alpha=np.zeros(3), quat=qs[i],
                                  thrust_world=sf*0.595, mass=0.595), env, throttle=0.45)
    if meas is not None:
        n_meas += 1
        kf.predict(meas.channels["gyro"], meas.channels["accel"], dt)
    if i % 10 == 0:
        fi = i // 10
        zp = dep[fi-1]/1000.0; zc = dep[fi]/1000.0
        Pp = cloud(zp, f, cx, cy); Pc = cloud(zc, f, cx, cy)
        src = Pc[(zc>0.15)&(zc<60.0)][::2]
        R_pc, t_pc = icp_p2p(src, zp, Pp, f, cx, cy)
        T_rel = np.eye(4); T_rel[:3,:3] = R_pc; T_rel[:3,3] = t_pc
        T_vo = T_vo @ T_rel
        kf.update_position(T_vo[:3,3].copy(), np.eye(3)*(0.25**2 * fi))
        kf.update_attitude(rot_to_quat(T_vo[:3,:3]), np.eye(3)*(0.02**2 * fi))
    if i % 200 == 0:
        fi = min(i//10, n-1)
        qe = quat_mul(np.array([-kf.q[0],-kf.q[1],-kf.q[2],kf.q[3]]), gt_q[fi])
        att_err = np.rad2deg(2*np.arccos(np.clip(abs(qe[3]),0,1)))
        print("t=%4.1fs att_err=%6.1fdeg bg_est(dps)=%s gyro_b_true(dps)=%s n_meas=%d" % (
            t, att_err, np.round(np.rad2deg(kf.bg),2),
            np.round(np.rad2deg(np.linalg.norm(imu.gyro_b)),3), n_meas), flush=True)
