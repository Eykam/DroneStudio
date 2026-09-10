import sys, json
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from vis_vo2 import cam_frame, rot_from_yaw_pitch
from vis_frontend import gray, shi_tomasi, pyr_track, unproject, kabsch_trimmed
from fusion_frontend import FrontendOdom
from fusion_rotgate import rot_angle

def run_trace(rgb, dep, meta, K, gate_deg=3.0, dt=0.1, seed=0):
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
    q_prev = kf.q.copy()
    log = []
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
        rej = 1 if (fe.total > 0 and dev > gate_deg) else 0
        if rej: T_rel = np.eye(4)
        T_vo = T_vo @ T_rel
        kf.update_position(T_vo[:3,3].copy(), np.eye(3)*(0.25**2*i))
        kf.update_attitude(rot_to_quat(T_vo[:3,:3]), np.eye(3)*(0.02**2*i))
        q_prev = kf.q.copy(); est.append(kf.p.copy())
        gt_dyaw = abs(meta[i,4]-meta[i-1,4]); gt_dpitch = abs(meta[i,5]-meta[i-1,5])
        gt_step = float(np.linalg.norm(gt_p[i]-gt_p[i-1]))
        nfeat = len(fe.feats) if fe.feats is not None else 0
        log.append([i, float(np.linalg.norm(est[-1]-gt_p[i])), dev, rej,
                    float(np.linalg.norm(T_rel[:3,3])), rot_angle(T_rel[:3,:3]),
                    gt_step, np.degrees(gt_dyaw), np.degrees(gt_dpitch), nfeat,
                    gt_p[i,0], gt_p[i,1], gt_p[i,2]])
    return np.array(log)

if __name__ == "__main__":
    d = np.load("/workspace/vision_model/traj/traj_s14_o3000.npz", allow_pickle=True)
    intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
    rgb, dep, meta = d["rgb"], d["depth"], d["meta"]
    for sid in [3000, 3005, 3002, 3007]:
        m = meta[:, 0] == sid
        L = run_trace(rgb[m], dep[m], meta[m], K)
        np.save(f"/tmp/trace_{sid}.npy", L)
        err = L[:, 1]
        jumps = np.argsort(np.diff(err))[::-1][:8] + 1
        print(f"=== {sid}: final err {err[-1]:.2f} max {err.max():.2f} at f{int(L[err.argmax(),0])}", flush=True)
        print(" top err-jump frames:", sorted(jumps.tolist()), flush=True)
        # per-frame stats around jumps
        jj = sorted(set(jumps.tolist()))
        hdr = "  f   err   dev  rej |Tt|  |Tr|  gtstep dyaw dpitch nfeat   gt_xyz"
        print(hdr, flush=True)
        for f in jj:
            w = L[(L[:,0] >= f-1) & (L[:,0] <= f+2)]
            for r in w:
                print(f"  {int(r[0]):3d} {r[1]:6.2f} {r[2]:5.2f} {int(r[3])} {r[4]:5.2f} {r[5]:5.2f} {r[6]:6.2f} {r[7]:5.1f} {r[8]:5.1f} {int(r[9]):4d}   ({r[10]:5.1f},{r[11]:4.1f},{r[12]:5.1f})", flush=True)
        # global corr: what do high-jump frames share?
        derr = np.diff(err)
        hi = derr > np.percentile(derr, 90)
        print(f"  p90-jump frames: mean dev {L[1:,2][hi].mean():.2f} rej% {L[1:,3][hi].mean()*100:.0f} dyaw {L[1:,7][hi].mean():.2f} gtstep {L[1:,6][hi].mean():.2f} nfeat {L[1:,9][hi].mean():.0f}", flush=True)
        print(f"  all frames:      mean dev {L[1:,2].mean():.2f} rej% {L[1:,3].mean()*100:.0f} dyaw {L[1:,7].mean():.2f} gtstep {L[1:,6].mean():.2f} nfeat {L[1:,9].mean():.0f}", flush=True)
