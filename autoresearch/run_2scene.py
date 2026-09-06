import json, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from vis_vo2 import cam_frame
from fusion_chained_g import run
d = np.load("/workspace/vision_model/traj/traj_s12_o1000.npz", allow_pickle=True)
intr = json.loads(str(d["intrinsics"])); K = cam_frame(intr["w"], intr["h"], intr["hfov_deg"])
dep, meta = d["depth"], d["meta"]
for sid in [1000, 1001, 1003, 1009]:
    m = meta[:,0]==sid
    for va in [True, False]:
        ate, ye, am_, ax_, rpe = run(dep[m], meta[m], K, use_tof=True, use_mag=True, use_vo_att=va, seed=sid)
        print("scene %d vo_att=%d  ATE %7.3f yRMSE %.3f att %.1f/%.1f RPE %.3f" % (sid, va, ate, ye, am_, ax_, rpe), flush=True)
