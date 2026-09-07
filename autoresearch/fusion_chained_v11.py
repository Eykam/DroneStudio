"""Fusion scorecard with depth-net v1.1 predictions in the VO loop.

Identical pipeline to fusion_chained_g.py (ESKF + MPU-9250 + mag + VL53L9CX
ToF + chained ICP VO with innovation gate), but the VO point clouds come from
visnet v1.1 predicted depth instead of the npz GT depth. Same scenes, same
seeds, same metrics -> apples-to-apples vs fusion_v1_scorecard.json (GT depth).
"""
import json, sys, os
import numpy as np
import torch
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_train import VisNet
from fusion_chained_g import run
from vis_vo2 import cam_frame

CKPT = "/workspace/vision_model/visnet_v11_best.pt"
CACHE = "/workspace/vision_model/traj/traj_s12_o1000_depv11.npy"
OUT = "/workspace/vision_model/traj/fusion_v11_scorecard.json"

d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
rgb, meta = d["rgb"], d["meta"]

if os.path.exists(CACHE):
    dep = np.load(CACHE)
    print("loaded cached v11 depth", dep.shape, flush=True)
else:
    net = VisNet()
    net.load_state_dict(torch.load(CKPT, map_location="cpu"))
    net.eval(); torch.set_num_threads(32)
    r = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    outs = []
    with torch.no_grad():
        for i in range(0, len(r), 256):
            pd_, _ = net(r[i:i + 256])
            outs.append(pd_)
            print(f"depth {i + len(pd_)}/{len(r)}", flush=True)
    dep_m = torch.cat(outs)
    if dep_m.dim() == 4:
        dep_m = dep_m[:, 0]
    dep = (dep_m.exp().numpy() * 1000.0).clip(0, 65535).astype(np.uint16)
    np.save(CACHE, dep)
    print("predicted v11 depth", dep.shape, flush=True)

sids = sorted(set(meta[:, 0].astype(int)))
rows = []
for k, sid in enumerate(sids):
    m = meta[:, 0] == sid
    ate, ye, am_, ax_, rpe = run(dep[m], meta[m], K, use_tof=True, use_mag=True,
                                 use_vo_att=False, seed=k)
    rows.append(dict(scene=int(sid), ate=ate, yrmse=ye, att_mean=am_, att_max=ax_, rpe=rpe))
    print("scene %d  ATE %7.3f yRMSE %.3f att %.1f/%.1f RPE %.3f" % (sid, ate, ye, am_, ax_, rpe), flush=True)
print("V11-DEPTH MEAN  ATE %.3f yRMSE %.3f att %.1f RPE %.3f  (max ATE %.3f)" % (
    np.mean([r["ate"] for r in rows]), np.mean([r["yrmse"] for r in rows]),
    np.mean([r["att_mean"] for r in rows]), np.mean([r["rpe"] for r in rows]),
    max(r["ate"] for r in rows)))
gt = json.load(open("/workspace/vision_model/traj/fusion_v1_scorecard.json"))
print("GT-DEPTH  MEAN  ATE %.3f yRMSE %.3f att %.1f RPE %.3f  (max ATE %.3f)" % (
    np.mean([r["ate"] for r in gt]), np.mean([r["yrmse"] for r in gt]),
    np.mean([r["att_mean"] for r in gt]), np.mean([r["rpe"] for r in gt]),
    max(r["ate"] for r in gt)))
json.dump(rows, open(OUT, "w"), indent=1)
