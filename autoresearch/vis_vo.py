"""Phase 2 VO v0: monocular visual odometry on depth (GT or learned).

Camera model (exact, from vision_raster.zig):
  dir_cam = (1, v, u), v=(cy-row)/f, u=(col-cx)/f, cx=w/2, cy=h/2,
  f=(w/2)/tan(hfov/2); depth[idx] = hit.t along NORMALIZED rd => RADIAL m.
  cam axes: +X forward (nose), +Y up, +Z right.
  world rotation: R = Ry(yaw) @ Rz(pitch) (standard right-handed).

Frame-to-frame point-to-point ICP on downsampled depth (pure numpy, no
cv2 on box). Pose chain: T_w_ci = T_w_c{i-1} @ T_prev_cur, initialized at
GT pose of frame 0 (standard VO init). Metrics: ATE + RPE (translation).
"""
import argparse, json, sys
import numpy as np

def cam_frame(w, h, hfov_deg):
    f = (w / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
    return f, w / 2.0, h / 2.0

def backproject(depth_m, f, cx, cy, step=4):
    h, w = depth_m.shape
    rows, cols = np.mgrid[0:h:step, 0:w:step].astype(np.float64)
    v = (cy - rows) / f
    u = (cols - cx) / f
    d = np.stack([np.ones_like(v), v, u], -1)          # (h',w',3) (1,v,u)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    z = depth_m[::step, ::step].astype(np.float64)
    pts = (d * z[..., None]).reshape(-1, 3)
    m = (z.ravel() > 0.15) & (z.ravel() < 60.0)
    return pts[m]

def rot_from_yaw_pitch(yaw_deg, pitch_deg):
    yaw = np.radians(yaw_deg); pitch = np.radians(pitch_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(pitch), -np.sin(pitch), 0],
                   [np.sin(pitch), np.cos(pitch), 0], [0, 0, 1]])
    return Ry @ Rz

def icp(src, dst, iters=25, max_dist=0.3):
    """R,t aligning src onto dst."""
    s = src.copy()
    R_tot, t_tot = np.eye(3), np.zeros(3)
    for _ in range(iters):
        idx = []
        for i in range(0, len(s), 512):
            d2 = ((s[i:i+512, None, :] - dst[None, :, :]) ** 2).sum(-1)
            idx.append(np.argmin(d2, 1))
        idx = np.concatenate(idx)
        pairs_d = np.linalg.norm(s - dst[idx], axis=1)
        m = pairs_d < max_dist
        if m.sum() < 10:
            break
        A, B = s[m], dst[idx][m]
        ca, cb = A.mean(0), B.mean(0)
        H = (A - ca).T @ (B - cb)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cb - R @ ca
        s = (R @ s.T).T + t
        R_tot = R @ R_tot
        t_tot = R @ t_tot + t
        if np.linalg.norm(t) < 1e-4:
            break
    return R_tot, t_tot

def run_scene(dep, meta, K):
    f, cx, cy = K
    n = len(dep)
    R0 = rot_from_yaw_pitch(meta[0, 4], meta[0, 5])
    T_w_c = np.eye(4)
    T_w_c[:3, :3] = R0
    T_w_c[:3, 3] = meta[0, 1:4]
    est_pos = [T_w_c[:3, 3].copy()]
    rpe = []
    for i in range(1, n):
        prev = backproject(dep[i-1] / 1000.0, f, cx, cy)
        cur = backproject(dep[i] / 1000.0, f, cx, cy)
        R_pc, t_pc = icp(cur, prev)              # maps cur cam -> prev cam
        T_prev_cur = np.eye(4)
        T_prev_cur[:3, :3] = R_pc
        T_prev_cur[:3, 3] = t_pc
        T_w_c = T_w_c @ T_prev_cur
        est_pos.append(T_w_c[:3, 3].copy())
        gt_dp = meta[i, 1:4] - meta[i-1, 1:4]
        est_dp = est_pos[-1] - est_pos[-2]
        rpe.append(np.linalg.norm(est_dp - gt_dp))
    est_pos = np.array(est_pos)
    gt_pos = meta[:n, 1:4]
    ate = float(np.sqrt(np.mean(np.sum((est_pos - gt_pos) ** 2, 1))))
    return {"ate": ate,
            "rpe": float(np.sqrt(np.mean(np.square(rpe)))) if rpe else 0.0,
            "traj_len_m": float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    intr = json.loads(str(d["intrinsics"]))
    K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
    dep, meta = d["depth"], d["meta"]
    out = []
    for sid in sorted(set(meta[:, 0].astype(int))):
        m = meta[:, 0] == sid
        r = run_scene(dep[m], meta[m], K)
        r["scene"] = int(sid)
        out.append(r)
        print(f"scene {sid}: ATE {r['ate']:.3f} m  RPE {r['rpe']:.3f} m  len {r['traj_len_m']:.1f} m", flush=True)
    if out:
        print(f"MEAN  ATE {np.mean([o['ate'] for o in out]):.3f} m  "
              f"RPE {np.mean([o['rpe'] for o in out]):.3f} m  over {len(out)} scenes")
    with open(a.npz.replace(".npz", "_vo_v0.json"), "w") as fh:
        json.dump(out, fh, indent=1)

if __name__ == "__main__":
    main()
