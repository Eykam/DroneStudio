"""Fusion innovation-gate sweep with LIVE ToF (parent approval 2026-09-06 22:00).

Gates were tuned with inert ToF; live altimeter changed the update mix and
moved GT-depth ATE 30.2 -> 36.2. Coordinate sweep over the four gate params
around the current defaults, GT-depth scorecard (12 scenes), mean ATE/yRMSE.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame
from fusion_chained_g import run

d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
dep, meta = d["depth"], d["meta"]
sids = sorted(set(meta[:, 0].astype(int)))

CONFIGS = [
    ("base(inno25,t2,sa.15,re20)", {}),
    ("inno15", {"inno_deg_per_fr": 15.0}),
    ("inno40", {"inno_deg_per_fr": 40.0}),
    ("tgate1.0", {"t_gate_per_fr": 1.0}),
    ("tgate4.0", {"t_gate_per_fr": 4.0}),
    ("safloor.05", {"sa_floor": 0.05}),
    ("safloor.30", {"sa_floor": 0.30}),
    ("reanchor12", {"reanchor_deg": 12.0}),
    ("reanchor30", {"reanchor_deg": 30.0}),
]
rows = []
for name, kw in CONFIGS:
    ate, ye, atts, rpes = [], [], [], []
    for k, sid in enumerate(sids):
        m = meta[:, 0] == sid
        a, y, am_, ax_, r = run(dep[m], meta[m], K, use_tof=True, use_mag=True,
                                use_vo_att=False, seed=k, **kw)
        ate.append(a); ye.append(y); atts.append(am_); rpes.append(r)
    row = dict(config=name, ate=float(np.mean(ate)), max_ate=float(max(ate)),
               yrmse=float(np.mean(ye)), att=float(np.mean(atts)), rpe=float(np.mean(rpes)))
    rows.append(row)
    print("SWEEP %-26s ATE %7.3f (max %7.3f) yRMSE %6.3f att %5.1f RPE %.3f" % (
        name, row["ate"], row["max_ate"], row["yrmse"], row["att"], row["rpe"]), flush=True)
json.dump(rows, open("/workspace/vision_model/traj/fusion_gate_sweep.json", "w"), indent=1)
print("SWEEP_DONE", flush=True)
