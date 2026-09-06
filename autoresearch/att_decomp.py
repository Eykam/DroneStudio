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
dep, meta = d["depth"], d["meta"]; m = meta[:,0]==1001
dep, meta = dep[m], meta[m]
n = len(dep); dt = 0.1
gt_p = meta[:,1:4]
gt_q = [rot_to_quat(rot_from_yaw_pitch(meta[i,4], meta[i,5])) for i in range(n)]
kf = ESKF(NOISE); kf.p = gt_p[0].copy(); kf.q = gt_q[0].copy(); kf.v = (gt_p[1]-gt_p[0])/dt
imu = SimIMU(MPU9250_SPEC, seed=0); env = SimEnvironment()
T_vo = np.eye(4); T_vo[:3,:3] = rot_from_yaw_pitch(meta[0,4], meta[0,5]); T_vo[:3,3] = gt_p[0]
yawerrs, tilterrs = [], []
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
    zp = dep[i-1]/1000.0; zc = dep[i]/1000.0
    Pp = cloud(zp, f, cx, cy); Pc = cloud(zc, f, cx, cy)
    src = Pc[(zc>0.15)&(zc<60.0)][::2]
    R_pc, t_pc = icp_p2p(src, zp, Pp, f, cx, cy)
    T_rel = np.eye(4); T_rel[:3,:3] = R_pc; T_rel[:3,3] = t_pc
    T_vo = T_vo @ T_rel
    kf.update_position(T_vo[:3,3].copy(), np.eye(3)*(0.25**2 * i))
    kf.update_attitude(rot_to_quat(T_vo[:3,:3]), np.eye(3)*(0.02**2 * i))
    # decompose: yaw err = angle about world Y of (est fwd vs gt fwd projected to XZ)
    fwd_e = R_of(kf.q) @ np.array([1.,0,0]); fwd_g = R_of(gt_q[i]) @ np.array([1.,0,0])
    ye = np.rad2deg(np.arctan2(fwd_e[2], fwd_e[0]) - np.arctan2(fwd_g[2], fwd_g[0]))
    ye = (ye + 180) % 360 - 180
    up_e = R_of(kf.q) @ np.array([0.,1,0]); up_g = R_of(gt_q[i]) @ np.array([0.,1,0])
    te = np.rad2deg(np.arccos(np.clip(up_e @ up_g, -1, 1)))
    yawerrs.append(abs(ye)); tilterrs.append(te)
print("scene 1001 10Hz chained:")
print("yaw err deg:  mean %.1f p50 %.1f p90 %.1f max %.1f" % (np.mean(yawerrs), *np.percentile(yawerrs,[50,90]), np.max(yawerrs)))
print("tilt err deg: mean %.1f p50 %.1f p90 %.1f max %.1f" % (np.mean(tilterrs), *np.percentile(tilterrs,[50,90]), np.max(tilterrs)))
