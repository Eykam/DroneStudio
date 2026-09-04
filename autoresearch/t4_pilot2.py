"""T4 stage 1b: land demo enrichment. land_t2 was 6/48 in the first pass and
land is the weakest behavior at every stage (BC land_t0 0.06 from 47/48
demos). More land coverage, esp. T1/T2 where obstacles crowd the approach.

Output: /workspace/t4_demos_land.npz (X, A, meta).
"""
import sys, json, os, time
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scenario_sampler import sample_spec, tier_dist
from t4_pilot import teacher_act, run_one  # run_one(seed, dist, spec, max_steps)

def main():
    t0 = time.time()
    X, A, meta = [], [], {}
    for tier in (0, 1, 2):
        ks = 0; att = 0
        while ks < 96 and att < 500:
            seed = 70000 + tier * 10000 + att
            att += 1
            dist = tier_dist(seed, tier)
            spec = sample_spec(seed, force_scenario="land")
            dist.n_waypoints = 0.0
            tr = run_one(seed, dist, spec, 700)
            if tr:
                ks += 1
                X += [t[0] for t in tr]; A += [t[1] for t in tr]
        meta[f"land_t{tier}"] = {"kept": ks, "attempts": att}
        print(f"collect land_t{tier}: kept {ks}/{att} samples={len(X)}", flush=True)
    np.savez("/workspace/t4_demos_land.npz", X=np.array(X), A=np.array(A), meta=json.dumps(meta))
    print("T4PILOT2_DONE " + json.dumps(meta) + f" samples={len(X)} wall={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
