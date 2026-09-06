import json, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame, cloud, icp_p2p, rot_from_yaw_pitch
from sensors.ekf import ESKF, quat_mul, R_of
from sensors.imu import SimIMU
from sensors.specs.mpu9250 import MPU9250_SPEC
from sensors.base import SimEnvironment
from fusion_v0 import rot_to_quat, NOISE, G

d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
f, cx, cy = K
dep, meta = d["depth"], d["meta"]; m = meta[:,0]==1000
dep, meta = dep[m], meta[m]
n = len(dep); dt = 0.1
gt_p = meta[:,1:4]
gt_q = [rot_to_quat(rot_from_yaw_pitch(meta[i,4], meta[i,5])) for i in range(n)]
env = SimEnvironment()
print("env temp:", [round(env.temperature(t),1) for t in [0,15,30,45,60]], flush=True)
imu = SimIMU(MPU9250_SPEC, seed=0)
print("imu const biases: gyro_b0(dps)=", np.round(imu.gyro_b0,2), "accel_b0(m/s2)=", np.round(imu.accel_b0,3), flush=True)
print("gyro tempco at env temp:", np.round(imu._gyro_tempco(env.temperature(1.0)),2), "dps", flush=True)
kf = ESKF(NOISE); kf.p = gt_p[0].copy(); kf.q = gt_q[0].copy(); kf.v = (gt_p[1]-gt_p[0])/dt
est_prev = kf.p.copy()
for i in range(1, n):
    t = i*dt; q_im1 = kf.q.copy(); p_im1 = est_prev
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
    z_pos = p_im1 + R_of(q_im1) @ t_pc
    kf.update_position(z_pos, np.eye(3)*0.30**2)
    kf.update_attitude(quat_mul(q_im1, rot_to_quat(R_pc)), np.eye(3)*0.10**2)
    est_prev = kf.p.copy()
    if i % 50 == 0 or i == n-1:
        qe = quat_mul(np.array([-kf.q[0],-kf.q[1],-kf.q[2],kf.q[3]]), gt_q[i])
        att_err = np.rad2deg(2*np.arccos(np.clip(abs(qe[3]),0,1)))
        print("i=%3d pos_err=%6.2fm att_err=%5.1fdeg bg_est(dps)=%s ba_est=%s t_pc|=%.2f" % (
            i, np.linalg.norm(kf.p-gt_p[i]), att_err,
            np.round(np.rad2deg(kf.bg),2), np.round(kf.ba,2), np.linalg.norm(t_pc)), flush=True)
