"""Phase 2 VO v0.1: depth VO via projective point-to-point ICP.

Camera model exact from vision_raster.zig: dir_cam=(1,v,u), v=(cy-row)/f,
u=(col-cx)/f, f=(w/2)/tan(hfov/2); depth = radial m (uint16 mm); sky=65535.
Cam axes +X fwd, +Y up, +Z right. R = Ry(yaw) @ Rz(pitch).

Estimator: projective data association (KinectFusion-style, no cv2) +
linearized point-to-point least squares, trim 0.8, 30 iters.
Pose chain: T_w_ci = T_w_c{i-1} @ T_prev_cur, init at GT frame 0.
Metrics: ATE (m), RPE (m/frame), per scene + MEAN line.
"""
import argparse, json
import numpy as np

def cam_frame(w, h, hfov_deg):
    return (w / 2.0) / np.tan(np.radians(hfov_deg / 2.0)), w / 2.0, h / 2.0

def cloud(z, f, cx, cy):
    h, w = z.shape
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float64)
    v = (cy - rows) / f
    u = (cols - cx) / f
    d = np.stack([np.ones_like(v), v, u], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d * z[..., None]

def project_idx(p, f, cx, cy, w, h):
    col = np.round(cx + f * p[:, 2] / p[:, 0]).astype(int)
    row = np.round(cy - f * p[:, 1] / p[:, 0]).astype(int)
    ok = (p[:, 0] > 0.2) & (row >= 0) & (row < h) & (col >= 0) & (col < w)
    return row, col, ok

def skew_batch(v):
    S = np.zeros((len(v), 3, 3))
    S[:, 0, 1], S[:, 0, 2] = -v[:, 2], v[:, 1]
    S[:, 1, 0], S[:, 1, 2] = v[:, 2], -v[:, 0]
    S[:, 2, 0], S[:, 2, 1] = -v[:, 1], v[:, 0]
    return S

def icp_p2p(srcP, dstZ, dstP, f, cx, cy, iters=30, trim=0.8, ret_fit=False):
    h, w = dstZ.shape
    s = srcP.copy()
    R_tot, t_tot = np.eye(3), np.zeros(3)
    fit = np.inf
    for _ in range(iters):
        row, col, ok = project_idx(s, f, cx, cy, w, h)
        zq = dstZ[row[ok], col[ok]]
        valid = (zq > 0.15) & (zq < 60.0)
        q = dstP[row[ok][valid], col[ok][valid]]
        s_ok = s[ok][valid]
        if len(s_ok) < 30:
            break
        r3 = s_ok - q
        rn = np.linalg.norm(r3, axis=1)
        keep = rn <= np.quantile(rn, trim)
        if keep.sum() < 30:
            break
        fit = float(np.median(rn[keep]))
        A = np.concatenate([-skew_batch(s_ok[keep]),
                            np.tile(np.eye(3), (keep.sum(), 1, 1))], 2).reshape(-1, 6)
        x, *_ = np.linalg.lstsq(A, -r3[keep].ravel(), rcond=None)
        wx, t = x[:3], x[3:]
        ang = np.linalg.norm(wx)
        if ang > 1e-9:
            ax = wx / ang
            Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R = np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx
        else:
            R = np.eye(3)
        s = (R @ s.T).T + t
        R_tot = R @ R_tot
        t_tot = R @ t_tot + t
        if np.linalg.norm(t) < 1e-5 and ang < 1e-5:
            break
    return (R_tot, t_tot, fit) if ret_fit else (R_tot, t_tot)

def rot_from_yaw_pitch(yaw_deg, pitch_deg):
    yaw = np.radians(yaw_deg); pitch = np.radians(pitch_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(pitch), -np.sin(pitch), 0],
                   [np.sin(pitch), np.cos(pitch), 0], [0, 0, 1]])
    return Ry @ Rz

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
        zprev = dep[i-1] / 1000.0
        zcur = dep[i] / 1000.0
        Pprev = cloud(zprev, f, cx, cy)
        Pcur = cloud(zcur, f, cx, cy)
        src = Pcur[(zcur > 0.15) & (zcur < 60.0)][::2]
        R_pc, t_pc = icp_p2p(src, zprev, Pprev, f, cx, cy)
        T_prev_cur = np.eye(4)
        T_prev_cur[:3, :3] = R_pc
        T_prev_cur[:3, 3] = t_pc
        T_w_c = T_w_c @ T_prev_cur
        est_pos.append(T_w_c[:3, 3].copy())
        gt_dp = meta[i, 1:4] - meta[i-1, 1:4]
        rpe.append(np.linalg.norm((est_pos[-1] - est_pos[-2]) - gt_dp))
    est_pos = np.array(est_pos)
    gt_pos = meta[:n, 1:4]
    ate = float(np.sqrt(np.mean(np.sum((est_pos - gt_pos) ** 2, 1))))
    return {"ate": ate,
            "rpe": float(np.sqrt(np.mean(np.square(rpe)))),
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
        print(f"scene {sid}: ATE {r['ate']:.3f} m  RPE {r['rpe']:.4f} m  len {r['traj_len_m']:.1f} m", flush=True)
    if out:
        print(f"MEAN  ATE {np.mean([o['ate'] for o in out]):.3f} m  "
              f"RPE {np.mean([o['rpe'] for o in out]):.4f} m  over {len(out)} scenes")
    with open(a.npz.replace(".npz", "_vo_v01.json"), "w") as fh:
        json.dump(out, fh, indent=1)

if __name__ == "__main__":
    main()
