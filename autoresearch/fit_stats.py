import json, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame, cloud, icp_p2p, rot_from_yaw_pitch
from fusion_v0 import rot_to_quat
from sensors.ekf import quat_mul

d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
f, cx, cy = K
dep, meta = d["depth"], d["meta"]; m = meta[:,0]==1001
dep, meta = dep[m], meta[m]
n = len(dep)
gt_R = [rot_from_yaw_pitch(meta[i,4], meta[i,5]) for i in range(n)]
fits, rots, roterrs = [], [], []
for i in range(1, n):
    zp = dep[i-1]/1000.0; zc = dep[i]/1000.0
    Pp = cloud(zp, f, cx, cy); Pc = cloud(zc, f, cx, cy)
    src = Pc[(zc>0.15)&(zc<60.0)][::2]
    R_pc, t_pc, fit = icp_p2p(src, zp, Pp, f, cx, cy, ret_fit=True)
    R_gt = gt_R[i-1].T @ gt_R[i]
    Re = R_pc @ R_gt.T
    ang = np.rad2deg(np.arccos(np.clip((np.trace(Re)-1)/2, -1, 1)))
    fits.append(fit); roterrs.append(ang)
fits = np.array(fits); roterrs = np.array(roterrs)
print("fit (m): mean %.3f p50 %.3f p90 %.3f p99 %.3f max %.3f" % (
    fits.mean(), *np.percentile(fits,[50,90,99]), fits.max()))
print("rot err (deg): mean %.2f p90 %.2f max %.2f" % (roterrs.mean(), np.percentile(roterrs,90), roterrs.max()))
print("corr(fit, roterr): %.2f" % np.corrcoef(fits, roterrs)[0,1])
bad = roterrs > 5
print("frames roterr>5deg: %d; their fit: mean %.3f max %.3f" % (bad.sum(), fits[bad].mean() if bad.sum() else 0, fits[bad].max() if bad.sum() else 0))
