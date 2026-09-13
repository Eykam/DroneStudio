#!/usr/bin/env python3
"""Zero 2 W bench for the Osprey mono vision stack (CM0 = same BCM2837 silicon).

Measures, on YOUR hardware:
  1. VisNet depth+seg inference rate via onnxruntime (1 and 4 threads, 128x96)
  2. VIO frontend rate (Shi-Tomasi + LK from the repo's vis_frontend.py)
  3. Peak RAM (RSS) of the whole process
Everything synthetic/deterministic - timing is the deliverable, not accuracy.

Usage: python3 bench.py [--passes 200] [--viopairs 50]
"""
import argparse, json, os, platform, resource, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, AR)

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def bench_net(model, passes, warmup):
    import onnxruntime as ort
    out = {}
    x = np.random.default_rng(0).random((1, 3, 96, 128), dtype=np.float32)
    for threads in (1, 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        for _ in range(warmup):
            sess.run(None, {"rgb": x})
        ts = []
        for _ in range(passes):
            t0 = time.perf_counter()
            sess.run(None, {"rgb": x})
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.array(ts)
        out[f"threads{threads}"] = {
            "median_ms": round(float(np.median(ts)), 2),
            "p90_ms": round(float(np.percentile(ts, 90)), 2),
            "fps": round(1000.0 / float(np.median(ts)), 1),
        }
    return out

def bench_vio(pairs):
    import vis_frontend as vf
    rng = np.random.default_rng(7)
    # synthetic textured frame pair: random blobs + small shift (deterministic)
    base = rng.random((96, 128))
    for _ in range(40):
        r, c = rng.integers(0, 96), rng.integers(0, 128)
        base[max(0, r-2):r+2, max(0, c-2):c+2] += 1.0
    f0 = base
    f1 = np.roll(base, 2, axis=1)  # 2px shift
    # warmup + feature detect
    pts = vf.shi_tomasi(f0)
    ts = []
    for _ in range(pairs):
        t0 = time.perf_counter()
        p = vf.shi_tomasi(f0)
        vf.pyr_track(f0, f1, p)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return {
        "features": int(len(pts)),
        "median_ms": round(float(np.median(ts)), 2),
        "p90_ms": round(float(np.percentile(ts, 90)), 2),
        "fps": round(1000.0 / float(np.median(ts)), 1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "visnet_v11_128x96.onnx"))
    ap.add_argument("--passes", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--viopairs", type=int, default=50)
    a = ap.parse_args()

    cpuinfo = ""
    try:
        cpuinfo = [l for l in open("/proc/cpuinfo") if "Model" in l or "model name" in l][0].strip()
    except Exception:
        pass
    memtotal = ""
    try:
        memtotal = [l for l in open("/proc/meminfo") if "MemTotal" in l][0].strip()
    except Exception:
        pass

    print("== machine ==")
    print(cpuinfo or platform.platform())
    print(memtotal)
    print("python:", sys.version.split()[0])

    net = bench_net(a.model, a.passes, a.warmup)
    print("\n== VisNet 128x96 (onnxruntime) ==")
    print(json.dumps(net, indent=1))

    try:
        vio = bench_vio(a.viopairs)
        print("\n== VIO frontend (numpy KLT, 128x96) ==")
        print(json.dumps(vio, indent=1))
    except Exception as e:
        print("\n== VIO frontend == FAILED:", repr(e))
        vio = None

    print("\n== RAM ==")
    print(f"peak RSS: {rss_mb():.0f} MB")
    if vio:
        combined_ms = net["threads4"]["median_ms"] + vio["median_ms"]
        print(f"\ncombined net(4T)+VIO: {combined_ms:.0f} ms = {1000.0/combined_ms:.1f} Hz")

if __name__ == "__main__":
    main()
