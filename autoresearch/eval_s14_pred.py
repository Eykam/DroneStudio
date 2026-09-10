import sys, json, time
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
import torch
from vis_train import VisNet, load_shards, metrics
from vis_vo2 import cam_frame
from fusion_rotgate import run_rotgate

torch.set_num_threads(32)
t0 = time.time()

def load_net(tag):
    net = VisNet()
    net.load_state_dict(torch.load(f"/workspace/vision_model/visnet_{tag}_best.pt", map_location="cpu"))
    net.eval()
    return net

# ---- 1. v3 test metrics on tex shards (same split protocol as vis_eval) ----
rgb, dep, seg, meta = load_shards("/workspace/vision_ds_tex/shard_*.npz")
scene_ids = meta[:, 0].astype(int)
uniq = np.unique(scene_ids)
rng = np.random.default_rng(7); rng.shuffle(uniq)
te = np.isin(scene_ids, uniq[:64].tolist())
r = torch.from_numpy(rgb[te].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
d = torch.from_numpy(dep[te].astype(np.float32) / 1000.0)
s = torch.from_numpy(seg[te].astype(np.int64))
del rgb, dep, seg
net3 = load_net("v3")
with torch.no_grad():
    PD, PS = [], []
    for i in range(0, len(r), 512):
        a, b = net3(r[i:i+512]); PD.append(a); PS.append(b)
pd_, ps_ = torch.cat(PD), torch.cat(PS)
logd = torch.log(d.clamp(0.3, 65.535))
mae, rmse, d125, ious = metrics(pd_, ps_, logd, s, d > 65.0)
miou = float(np.nanmean(list(ious.values())))
print(f"V3 TEST: mae {mae:.3f} rmse {rmse:.3f} d125 {d125:.3f} miou {miou:.3f}  (v2 was mae 3.26 rmse 5.68 d125 0.60 miou 0.933)", flush=True)
del r, d, s, pd_, ps_, PD, PS

# ---- 2. predict depth on s14 with v2 and v3 ----
d14 = np.load("/workspace/vision_model/traj/traj_s14_o3000.npz", allow_pickle=True)
intr = json.loads(str(d14["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
rgb14, dep_gt, meta = d14["rgb"], d14["depth"], d14["meta"]
r14 = torch.from_numpy(rgb14.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
def predict(net):
    outs = []
    with torch.no_grad():
        for i in range(0, len(r14), 512):
            a, _ = net(r14[i:i+512]); outs.append(a)
    m = torch.exp(torch.cat(outs)).numpy()  # log-depth -> meters
    return np.clip(m * 1000.0, 0, 65535).astype(np.uint16)
dep_v2 = predict(load_net("v2"))
np.save("/workspace/vision_model/traj/traj_s14_o3000_depv2.npy", dep_v2)
dep_v3 = predict(net3)
np.save("/workspace/vision_model/traj/traj_s14_o3000_depv3.npy", dep_v3)
print(f"predictions saved ({time.time()-t0:.0f}s)", flush=True)

# ---- 3. rotgate fusion on s14: GT vs v2-pred vs v3-pred ----
sids = sorted(set(meta[:, 0].astype(int)))
res = {"gt": [], "v2": [], "v3": []}
for sid in sids:
    m = meta[:, 0] == sid
    line = f"{sid}:"
    for tag, dx in [("gt", dep_gt), ("v2", dep_v2), ("v3", dep_v3)]:
        ate, rpe, frac, nrej = run_rotgate(rgb14[m], dx[m], meta[m], K)
        res[tag].append(ate)
        line += f"  {tag} ATE {ate:7.3f} (rpe {rpe:.3f} sol {frac:.2f})"
    print(line, flush=True)
print(f"MEAN: gt {np.mean(res['gt']):.3f}  v2pred {np.mean(res['v2']):.3f}  v3pred {np.mean(res['v3']):.3f}", flush=True)
print(f"ALL DONE ({time.time()-t0:.0f}s)", flush=True)
