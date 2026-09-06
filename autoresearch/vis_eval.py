#!/usr/bin/env python3
"""vis_eval.py - held-out TEST eval + latency benchmark for visnet (Phase 1c).
Test split identical to vis_train.py (seed-7 shuffle, uniq[:64] = test).
Latency: single-thread CPU, median of 100 forward passes at 128x96.
"""
import json, sys, time
import numpy as np
import torch

sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_train import VisNet, load_shards, metrics

CKPT = "/workspace/vision_model/visnet_v1_best.pt"
rgb, dep, seg, meta = load_shards("/workspace/vision_ds/shard_*.npz")
scene_ids = meta[:, 0].astype(int)
uniq = np.unique(scene_ids)
rng = np.random.default_rng(7)
rng.shuffle(uniq)
test_ids = set(uniq[:64].tolist())
te = np.isin(scene_ids, list(test_ids))
r = torch.from_numpy(rgb[te].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
d = torch.from_numpy(dep[te].astype(np.float32) / 1000.0)
s = torch.from_numpy(seg[te].astype(np.int64))
print(f"test frames: {len(r)}", flush=True)

net = VisNet()
net.load_state_dict(torch.load(CKPT, map_location="cpu"))
net.eval()

# full test eval in chunks
torch.set_num_threads(32)
PDS, PSS = [], []
with torch.no_grad():
    for i in range(0, len(r), 512):
        pd_, ps_ = net(r[i:i+512])
        PDS.append(pd_); PSS.append(ps_)
pd_ = torch.cat(PDS); ps_ = torch.cat(PSS)
logd = torch.log(d.clamp(0.3, 65.535))
mae, rmse, d125, ious = metrics(pd_, ps_, logd, s, d > 65.0)
miou = float(np.nanmean(list(ious.values())))

# single-core latency (Pi proxy, flagged as such)
torch.set_num_threads(1)
x = r[:1]
with torch.no_grad():
    for _ in range(10):
        net(x)
    ts = []
    for _ in range(100):
        t0 = time.perf_counter()
        net(x)
        ts.append((time.perf_counter() - t0) * 1000)
lat = float(np.median(ts))

out = {"checkpoint": CKPT, "test_frames": int(len(r)),
       "depth": {"mae_m": round(mae, 3), "rmse_m": round(rmse, 3), "delta125": round(d125, 4)},
       "seg": {"miou": round(miou, 4), "per_class_iou": {str(c): round(v, 4) for c, v in ious.items()}},
       "latency_ms_single_core_server": round(lat, 2),
       "note": "server CPU proxy - NOT a Pi 5 measurement"}
json.dump(out, open("/workspace/vision_model/visnet_v1_test.json", "w"), indent=2)
print(json.dumps(out, indent=2), flush=True)
