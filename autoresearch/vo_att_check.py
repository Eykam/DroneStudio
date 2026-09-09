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
# gyro rate profile from GT
rates = []
for i in range(1, n):
    q0 = rot_to_quat(gt_R[i-1]); q1 = rot_to_quat(gt_R[i])
    if np.dot(q0,q1) < 0: q1 = -q1
    dq = quat_mul(np.array([-q0[0],-q0[1],-q0[2],q0[3]]), q1)
    if dq[3] < 0: dq = -dq
    rates.append(np.rad2deg(np.linalg.norm(2*dq[:3]/0.1)))
rates = np.array(rates)
print("GT angular rate dps: mean %.0f p50 %.0f p95 %.0f max %.0f" % (
    rates.mean(), np.percentile(rates,50), np.percentile(rates,95), rates.max()))
# VO per-frame rotation error vs GT
errs = []
for i in range(1, n):
    zp = dep[i-1]/1000.0; zc = dep[i]/1000.0
    Pp = cloud(zp, f, cx, cy); Pc = cloud(zc, f, cx, cy)
    src = Pc[(zc>0.15)&(zc<60.0)][::2]
    R_pc, t_pc = icp_p2p(src, zp, Pp, f, cx, cy)
    R_gt = gt_R[i-1].T @ gt_R[i]
    Re = R_pc @ R_gt.T
    ang = np.rad2deg(np.arccos(np.clip((np.trace(Re)-1)/2, -1, 1)))
    terr = np.linalg.norm(t_pc - gt_R[i-1].T @ (meta[i,1:4]-meta[i-1,1:4]))
    errs.append((ang, terr, rates[i-1]))
errs = np.array(errs)
print("VO rot err deg: mean %.2f p50 %.2f p90 %.2f p99 %.2f max %.2f" % (
    errs[:,0].mean(), *np.percentile(errs[:,0],[50,90,99]), errs[:,0].max()))
print("VO trans err m: mean %.3f p90 %.3f max %.3f" % (
    errs[:,1].mean(), np.percentile(errs[:,1],90), errs[:,1].max()))
bad = np.where(errs[:,0] > 5)[0]
print("frames with rot err > 5deg:", len(bad), "of", n-1, "; worst idx:", bad[np.argsort(-errs[bad,0])[:5]] if len(bad) else "-")
if len(bad):
    print("rot err vs GT rate corr: %.2f" % np.corrcoef(errs[:,0], errs[:,2])[0,1])
