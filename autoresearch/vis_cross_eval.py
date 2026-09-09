import json, sys
import numpy as np
import torch
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_train import VisNet, load_shards, metrics

def run(ckpt, data_glob):
    rgb, dep, seg, meta = load_shards(data_glob)
    scene_ids = meta[:, 0].astype(int)
    uniq = np.unique(scene_ids)
    rng = np.random.default_rng(7)
    rng.shuffle(uniq)
    test_ids = set(uniq[:64].tolist())
    te = np.isin(scene_ids, list(test_ids))
    print(f"frames: {int(te.sum())}", flush=True)
    net = VisNet()
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    mae_l, rmse_l, d125_l, iou_l = [], [], [], []
    for i in range(0, int(te.sum()), 512):
        r = torch.from_numpy(rgb[te][i:i+512].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        d = torch.from_numpy(dep[te][i:i+512].astype(np.float32) / 1000.0)
        s = torch.from_numpy(seg[te][i:i+512].astype(np.int64))
        with torch.no_grad():
            pd_, ps_ = net(r)
            logd = torch.log(d.clamp(0.3, 65.535))
            mae, rmse, d125, ious = metrics(pd_, ps_, logd, s, d > 65.0)
        mae_l.append(mae); rmse_l.append(rmse); d125_l.append(d125); iou_l.append(ious)
    keys = sorted(iou_l[0].keys())
    return {
        "mIoU": float(np.nanmean([np.mean([x[k] for k in keys]) for x in iou_l])),
        "MAE_m": float(np.mean(mae_l)),
        "RMSE_m": float(np.mean(rmse_l)),
        "d1.25": float(np.mean(d125_l)),
        "ious": {str(k): float(np.mean([x[k] for x in iou_l])) for k in keys},
    }

ckpt, data = sys.argv[1], sys.argv[2]
res = run(ckpt, data)
res["ckpt"] = ckpt.split("/")[-1]
res["data"] = data
print(json.dumps(res))
