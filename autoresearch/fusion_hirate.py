"""Fusion v2: IMU/EKF at 100Hz via Catmull-Rom + slerp upsampled kinematics.

Frame-rate (10Hz) finite differences starve the ESKF: bias absorption needs
many more integration steps than VO frames. Upsample GT poses 10x with C1
Catmull-Rom (positions) and slerp (attitudes), derive omega/specific-force at
100Hz, run SimIMU+ESKF at 100Hz, keep VO corrections at the true 10Hz frame
rate (chained form, R growing in frame index). ToF altimeter at its own mode
rate, low-tilt gated.
"""
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

def catmull_rom(P0, P1, P2, P3, u):
    return 0.5 * ((2*P1) + (-P0+P2)*u + (2*P0-5*P1+4*P2-P3)*u**2 + (-P0+3*P1-3*P2+P3)*u**3)

def slerp(q0, q1, u):
    d = np.dot(q0, q1)
    if d < 0: q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u*(q1-q0); return q/np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1-u)*th)*q0 + np.sin(u*th)*q1) / np.sin(th)

def upsample(gt_p, gt_q, dt, k=10):
    """10Hz knots -> k*dense samples. Returns dense t, p, q."""
    n = len(gt_p)
    ts, ps, qs = [], [], []
    P = np.vstack([gt_p[0], gt_p, gt_p[-1]])         # clamp ends
    Q = [gt_q[0]] + list(gt_q) + [gt_q[-1]]
    for i in range(n-1):
        for j in range(k):
            u = j / k
            ts.append((i + u) * dt)
            ps.append(catmull_rom(P[i], P[i+1], P[i+2], P[i+3], u))
            qs.append(slerp(Q[i+1], Q[i+2], u))
    ts.append((n-1)*dt); ps.append(gt_p[-1]); qs.append(gt_q[-1])
    # quat double-cover: enforce sign continuity along the dense sequence,
    # else interval joints produce 360deg omega spikes in finite differences
    for i in range(1, len(qs)):
        if np.dot(qs[i], qs[i-1]) < 0:
            qs[i] = -qs[i]
    return np.array(ts), np.array(ps), qs

def run(dep, meta, K, use_tof=True, vo_every=1, rate=100, seed=0):
    f, cx, cy = K
    n = len(dep)
    dt_f = 0.1
    gt_p = meta[:,1:4]
    gt_q = [rot_to_quat(rot_from_yaw_pitch(meta[i,4], meta[i,5])) for i in range(n)]
    ts, ps, qs = upsample(gt_p, gt_q, dt_f, k=rate//10)
    dt = ts[1] - ts[0]
    kf = ESKF(NOISE); kf.p = ps[0].copy(); kf.q = qs[0].copy()
    kf.v = (ps[1]-ps[0])/dt
    imu = SimIMU(MPU9250_SPEC, seed=seed); env = SimEnvironment()
    tof = SimToF(VL53L9CX_SPEC, mode="room_mapping", seed=seed+7)
    def cast_ground(o, d):
        if d[1] >= -1e-6: return None
        tt = -o[1]/d[1]
        return float(tt) if tt > 0 else None
    est_idx, est_pos = [0], [ps[0].copy()]
    att_errs = []
    T_vo = np.eye(4); T_vo[:3,:3] = rot_from_yaw_pitch(meta[0,4], meta[0,5]); T_vo[:3,3] = gt_p[0]
    vo_frame = 0; last_vo = 0
    frames_per_vo = (rate//10) * vo_every
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
        gm, am = (meas.channels["gyro"], meas.channels["accel"]) if meas is not None else (omega_body, sf*0)  # due() gating
        if meas is not None:
            kf.predict(gm, am, dt)
        else:
            kf.predict(omega_body, sf*0, dt)  # IMU between samples: ideal (shouldn't happen at 100Hz)
        # ToF at its own rate, GT pose generates the measurement
        if use_tof:
            tm = tof.scan(t, dict(quat=qs[i], origin=ps[i]), env, cast_ground)
            if tm is not None:
                rng = tm.channels["ranges"].ravel(); st = tm.channels["status"].ravel()
                dirs_b = tof._dirs_sensor
                ok = np.where(st == 0)[0]
                if len(ok):
                    Rk = R_of(kf.q)
                    j = ok[np.argmin([(Rk @ dirs_b[m_])[1] for m_ in ok])]
                    d_w = Rk @ dirs_b[j]
                    if np.arccos(np.clip(-d_w[1], 0, 1)) <= np.deg2rad(15):
                        r_true = rng[j]
                        sigma = tof.spec["sigma_base_mm"]/1000.0 + tof.spec["sigma_range2"]*r_true*r_true
                        kf.update_ground_range(r_true, dirs_b[j], max(sigma, 0.005))
        # VO at frame rate (chained)
        if i % (rate//10) == 0:
            fi = i // (rate//10)
            vo_frame += 1
            if vo_frame % vo_every == 0:
                T_chain = np.eye(4)
                fits = []
                for jj in range(last_vo+1, fi+1):
                    zp = dep[jj-1]/1000.0; zc = dep[jj]/1000.0
                    Pp = cloud(zp, f, cx, cy); Pc = cloud(zc, f, cx, cy)
                    src = Pc[(zc>0.15)&(zc<60.0)][::2]
                    R_pc, t_pc, fit = icp_p2p(src, zp, Pp, f, cx, cy, ret_fit=True)
                    fits.append(fit)
                    T_rel = np.eye(4); T_rel[:3,:3] = R_pc; T_rel[:3,3] = t_pc
                    T_chain = T_chain @ T_rel
                T_vo = T_vo @ T_chain
                # fitness-gated trust: ICP residual (median inlier m) inflates R;
                # hopeless frames (fit > 0.3m, fast/blurry maneuvers) get skipped
                fq = float(np.max(fits)) if fits else np.inf
                if fq < 0.30:
                    sp = 0.25 + 2.0 * fq
                    sa = 0.02 + 0.5 * fq
                    kf.update_position(T_vo[:3,3].copy(), np.eye(3)*(sp**2 * fi))
                    kf.update_attitude(rot_to_quat(T_vo[:3,:3]), np.eye(3)*(sa**2 * fi))
                last_vo = fi
            est_idx.append(fi); est_pos.append(kf.p.copy())
            qe = quat_mul(np.array([-kf.q[0],-kf.q[1],-kf.q[2],kf.q[3]]), gt_q[fi])
            att_errs.append(np.rad2deg(2*np.arccos(np.clip(abs(qe[3]),0,1))))
    est_pos = np.array(est_pos)
    gt_at = gt_p[est_idx]
    ate = float(np.sqrt(np.mean(np.sum((est_pos-gt_at)**2,1))))
    yrmse = float(np.sqrt(np.mean((est_pos[:,1]-gt_at[:,1])**2)))
    return ate, yrmse, float(np.mean(att_errs)), float(np.max(att_errs))

d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
dep, meta = d["depth"], d["meta"]
for sid in [1000, 1001]:
    m = meta[:,0]==sid
    for use_tof, ve in [(False,1), (True,1), (True,5)]:
        ate, ye, am_, ax_ = run(dep[m], meta[m], K, use_tof=use_tof, vo_every=ve)
        print("scene %d 100Hz tof=%d vo_every=%d  ATE %7.3f yRMSE %.3f att mean %.1fdeg max %.1fdeg"
              % (sid, use_tof, ve, ate, ye, am_, ax_), flush=True)
