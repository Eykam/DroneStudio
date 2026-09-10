"""Phase 2d: feature-VIO frontend v0 - RGB feature tracking to break the
depth-ICP aperture wall (VO_PHASE2.md: solver is exact with perfect
correspondences; planar scenes give it none along forward/tangent).

Pipeline per frame pair (pure numpy, deterministic):
  1. Shi-Tomasi corners on grayscale (min-eigenvalue map, NMS, min dist).
  2. Pyramidal Lucas-Kanade tracking prev->cur (2 levels, 9x9 win, 8 it),
     forward-backward consistency check.
  3. GT depth at both endpoints (development signal per NAV_STACK: depth
     from the rasterizer as ground truth) -> 3D-3D correspondences.
  4. Trimmed Kabsch SE3 (trim 0.8, 3 refits) -> relative motion.

Camera model exact from vision_raster.zig (same as vis_vo2): dir=(1,v,u),
v=(cy-row)/f, u=(col-cx)/f, f=(w/2)/tan(hfov/2); +X fwd, +Y up, +Z right;
depth = radial m (uint16 mm), sky=65535. R = Ry(yaw) @ Rz(pitch).
Metrics identical to vis_vo2: per-scene ATE / RPE (m/frame) + MEAN.
"""
import argparse, json
import numpy as np

def cam_frame(w, h, hfov_deg):
    return (w / 2.0) / np.tan(np.radians(hfov_deg / 2.0)), w / 2.0, h / 2.0

def gray(rgb):
    return rgb.astype(np.float64).mean(-1)

def box_blur(im, k=5):
    p = k // 2
    pad = np.pad(im, p, mode="edge")
    c = np.cumsum(np.cumsum(pad, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    out = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
    return out

def shi_tomasi(g, max_feats=300, min_dist=5, q=0.005):
    ix = np.zeros_like(g); iy = np.zeros_like(g)
    ix[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    iy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    a = box_blur(ix * ix); b = box_blur(ix * iy); c = box_blur(iy * iy)
    tr = (a + c) * 0.5
    dt = np.sqrt(np.maximum(tr * tr - (a * c - b * b), 0.0))
    lam = tr - dt  # min eigenvalue
    thr = lam.max() * q
    ys, xs = np.nonzero(lam > thr)
    vals = lam[ys, xs]
    order = np.argsort(-vals)
    H, W = g.shape
    occ = np.zeros((H, W), bool)
    feats = []
    for idx in order:
        y, x = ys[idx], xs[idx]
        if x < 5 or y < 5 or x >= W - 5 or y >= H - 5:
            continue
        if occ[max(0, y - min_dist):y + min_dist + 1,
               max(0, x - min_dist):x + min_dist + 1].any():
            continue
        occ[y, x] = True
        feats.append((float(x), float(y)))
        if len(feats) >= max_feats:
            break
    return np.array(feats) if feats else np.zeros((0, 2))

def sample(im, x, y):
    """bilinear sample at (x,y) arrays; invalid -> nan"""
    H, W = im.shape
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = x0 + 1; y1 = y0 + 1
    ok = (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    wx = x - x0; wy = y - y0
    v = (im[y0c, x0c] * (1 - wx) * (1 - wy) + im[y0c, x1c] * wx * (1 - wy)
         + im[y1c, x0c] * (1 - wx) * wy + im[y1c, x1c] * wx * wy)
    return np.where(ok, v, np.nan)

def lk_track(g0, g1, pts, win=9, iters=8, clamp=2.0):
    """single-level LK for pts (N,2) from g0 into g1; returns new pts + ok"""
    p = win // 2
    ix0 = np.zeros_like(g0); iy0 = np.zeros_like(g0)
    ix0[:, 1:-1] = (g0[:, 2:] - g0[:, :-2]) * 0.5
    iy0[1:-1, :] = (g0[2:, :] - g0[:-2, :]) * 0.5
    ix1 = np.zeros_like(g1); iy1 = np.zeros_like(g1)
    ix1[:, 1:-1] = (g1[:, 2:] - g1[:, :-2]) * 0.5
    iy1[1:-1, :] = (g1[2:, :] - g1[:-2, :]) * 0.5
    x = pts[:, 0].copy(); y = pts[:, 1].copy()
    ok = np.isfinite(sample(g1, x, y))
    offs = np.arange(-p, p + 1)
    ox, oy = np.meshgrid(offs, offs)
    ox = ox.ravel(); oy = oy.ravel()
    for _ in range(iters):
        xs = x[:, None] + ox[None, :]; ys = y[:, None] + oy[None, :]
        I1 = sample(g1, xs, ys)
        Ix = sample(0.5 * (ix0 + ix1), xs, ys)
        Iy = sample(0.5 * (iy0 + iy1), xs, ys)
        x0s = pts[:, 0:1] + ox[None, :]; y0s = pts[:, 1:2] + oy[None, :]
        I0 = sample(g0, x0s, y0s)
        dI = I1 - I0
        good = np.isfinite(dI).all(1) & np.isfinite(Ix).all(1)
        A11 = np.nansum(Ix * Ix, 1); A12 = np.nansum(Ix * Iy, 1); A22 = np.nansum(Iy * Iy, 1)
        b1 = -np.nansum(Ix * dI, 1); b2 = -np.nansum(Iy * dI, 1)
        det = A11 * A22 - A12 * A12
        solv = good & (np.abs(det) > 1e-6)
        dx = np.zeros_like(x); dy = np.zeros_like(y)
        dx[solv] = (A22[solv] * b1[solv] - A12[solv] * b2[solv]) / det[solv]
        dy[solv] = (A11[solv] * b2[solv] - A12[solv] * b1[solv]) / det[solv]
        step = np.hypot(dx, dy)
        x = x + np.clip(dx, -clamp, clamp); y = y + np.clip(dy, -clamp, clamp)
        ok &= solv
        ok &= np.isfinite(sample(g1, x, y))
        if step.max(initial=0) < 0.05:
            break
    return np.stack([x, y], 1), ok

def down2(g):
    """exact 2x2 mean downsample; pixel (i,j) covers fine (2i..2i+1, 2j..2j+1),
    so coordinates map x_coarse = (x_fine - 0.5)/2 and x_fine = 2*x_coarse + 0.5"""
    h, w = g.shape
    g = g[: h - h % 2, : w - w % 2]
    return 0.25 * (g[0::2, 0::2] + g[1::2, 0::2] + g[0::2, 1::2] + g[1::2, 1::2])

def clamp_pts(x, y, shape, win):
    p = win // 2 + 1
    H, W = shape
    return np.clip(x, p, W - p - 1), np.clip(y, p, H - p - 1)

def pyr_track(g0, g1, pts, win=9):
    # NOTE: pyramid removed (2026-09-09 debug): at 10Hz trajectory rates the
    # true flow is ~1-4px; coarse levels contributed noise, not capture range,
    # and pushed inits outside the narrow convergence basin (median residual
    # 0.96 DN from detection points vs 25.4 after coarse-init drift). Single
    # full-res LK + forward-backward check is the honest v0 on this data.
    p, ok = lk_track(g0, g1, pts, win=win, iters=10, clamp=3.0)
    pb, okb = lk_track(g1, g0, p, win=win, iters=10, clamp=3.0)
    fb = np.hypot(pb[:, 0] - pts[:, 0], pb[:, 1] - pts[:, 1])
    ok &= okb & (fb < 0.5)
    return p, ok

def unproject(pts, z_mm, f, cx, cy):
    z = z_mm / 1000.0
    v = (cy - pts[:, 1]) / f
    u = (pts[:, 0] - cx) / f
    d = np.stack([np.ones_like(v), v, u], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d * z[:, None]

def kabsch_trimmed(P, Q, trim=0.8, refits=3):
    """SE3 mapping P -> Q with iterative trimming."""
    keep = np.ones(len(P), bool)
    R = np.eye(3); t = np.zeros(3)
    for _ in range(refits):
        Pk, Qk = P[keep], Q[keep]
        cp, cq = Pk.mean(0), Qk.mean(0)
        H = (Pk - cp).T @ (Qk - cq)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cq - R @ cp
        resid = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        thr = np.quantile(resid, trim)
        keep = resid <= max(thr, 1e-9)
    return R, t, keep

def rot_from_yaw_pitch(yaw_deg, pitch_deg):
    yaw = np.radians(yaw_deg); pitch = np.radians(pitch_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(pitch), -np.sin(pitch), 0],
                   [np.sin(pitch), np.cos(pitch), 0], [0, 0, 1]])
    return Ry @ Rz

def run_scene(rgb, dep, meta, K, max_feats=200):
    f, cx, cy = K
    n = len(rgb)
    T_w_c = np.eye(4)
    T_w_c[:3, :3] = rot_from_yaw_pitch(meta[0, 4], meta[0, 5])
    T_w_c[:3, 3] = meta[0, 1:4]
    est_pos = [T_w_c[:3, 3].copy()]
    rpe = []
    feats = shi_tomasi(gray(rgb[0]), max_feats)
    nframes_used = 0
    for i in range(1, n):
        g0, g1 = gray(rgb[i - 1]), gray(rgb[i])
        if len(feats) >= 8:
            tr, ok = pyr_track(g0, g1, feats)
            z0 = dep[i - 1][np.clip(feats[:, 1].astype(int), 0, dep.shape[1] - 1),
                              np.clip(feats[:, 0].astype(int), 0, dep.shape[2] - 1)]
            z1 = dep[i][np.clip(np.round(tr[:, 1]).astype(int), 0, dep.shape[1] - 1),
                        np.clip(np.round(tr[:, 0]).astype(int), 0, dep.shape[2] - 1)]
            good = ok & (z0 > 150) & (z0 < 60000) & (z1 > 150) & (z1 < 60000)
            if good.sum() >= 8:
                P = unproject(feats[good], z0[good].astype(np.float64), f, cx, cy)
                Q = unproject(tr[good], z1[good].astype(np.float64), f, cx, cy)
                R, t, inl = kabsch_trimmed(P, Q)
                # ICP convention (vis_vo2): transform cur->prev maps cur cloud
                # into prev frame; Kabsch P->Q maps prev->cur, so invert.
                T_prev_cur = np.eye(4)
                T_prev_cur[:3, :3] = R.T
                T_prev_cur[:3, 3] = -R.T @ t
                T_w_c = T_w_c @ T_prev_cur
                nframes_used += 1
            feats = tr[ok]  # tracked pts live in g1 coords even if unsolved
        if len(feats) < 120:
            feats = shi_tomasi(g1, max_feats)
        est_pos.append(T_w_c[:3, 3].copy())
        gt_dp = meta[i, 1:4] - meta[i - 1, 1:4]
        rpe.append(np.linalg.norm((est_pos[-1] - est_pos[-2]) - gt_dp))
    est_pos = np.array(est_pos)
    gt_pos = meta[:n, 1:4]
    ate = float(np.sqrt(np.mean(np.sum((est_pos - gt_pos) ** 2, 1))))
    return {"ate": ate,
            "rpe": float(np.sqrt(np.mean(np.square(rpe)))),
            "frames_solved": nframes_used,
            "traj_len_m": float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    intr = json.loads(str(d["intrinsics"]))
    K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
    rgb, dep, meta = d["rgb"], d["depth"], d["meta"]
    out = []
    for sid in sorted(set(meta[:, 0].astype(int))):
        m = meta[:, 0] == sid
        r = run_scene(rgb[m], dep[m], meta[m], K)
        r["scene"] = int(sid)
        out.append(r)
        print(f"scene {sid}: ATE {r['ate']:.3f} m  RPE {r['rpe']:.4f} m  "
              f"solved {r['frames_solved']}/{int(m.sum())-1}  len {r['traj_len_m']:.1f} m", flush=True)
    if out:
        print(f"MEAN  ATE {np.mean([o['ate'] for o in out]):.3f} m  "
              f"RPE {np.mean([o['rpe'] for o in out]):.4f} m  over {len(out)} scenes")
    with open(a.out or a.npz.replace(".npz", "_frontend_v0.json"), "w") as fh:
        json.dump(out, fh, indent=1)

if __name__ == "__main__":
    main()
