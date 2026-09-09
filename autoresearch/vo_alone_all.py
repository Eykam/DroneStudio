import json, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame, cloud, icp_p2p, rot_from_yaw_pitch

d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
f, cx, cy = K
dep, meta = d["depth"], d["meta"]
rows = []
for sid in sorted(set(meta[:,0].astype(int))):
    m = meta[:,0]==sid
    dz, mz = dep[m], meta[m]
    gt_p = mz[:,1:4]
    T = np.eye(4); T[:3,:3] = rot_from_yaw_pitch(mz[0,4], mz[0,5]); T[:3,3] = gt_p[0]
    est = [gt_p[0].copy()]
    for i in range(1, len(dz)):
        zp = dz[i-1]/1000.0; zc = dz[i]/1000.0
        Pp = cloud(zp, f, cx, cy); Pc = cloud(zc, f, cx, cy)
        src = Pc[(zc>0.15)&(zc<60.0)][::2]
        R, t = icp_p2p(src, zp, Pp, f, cx, cy)
        Tr = np.eye(4); Tr[:3,:3] = R; Tr[:3,3] = t
        T = T @ Tr
        est.append(T[:3,3].copy())
    est = np.array(est)
    ate = float(np.sqrt(np.mean(np.sum((est-gt_p)**2,1))))
    rows.append(dict(scene=int(sid), ate=ate))
    print("scene %d  VO-alone ATE %7.3f" % (sid, ate), flush=True)
print("MEAN VO-alone ATE %.3f (max %.3f)" % (np.mean([r["ate"] for r in rows]), max(r["ate"] for r in rows)))
json.dump(rows, open("/workspace/vision_model/traj/vo_alone_scorecard.json", "w"), indent=1)
