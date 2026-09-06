#!/usr/bin/env python3
"""vis_dashboard_streamer.py - posts vision-training state to the dashboard.

Tails the training log for per-epoch metric lines; whenever the best
checkpoint changes, reloads it and renders GT-vs-prediction panels from a
fixed batch of VAL-scene frames (never test). DASHBOARD_URL + INGEST_TOKEN
come from /workspace/.dashboard_env via the launcher.
"""
import glob, json, os, re, subprocess, sys, time, urllib.request
import numpy as np

sys.path.insert(0, "/workspace/DroneStudio/autoresearch")

LOG = "/workspace/vision_model/train_v1.log"
CKPT = "/workspace/vision_model/visnet_v1_best.pt"
STATE = "/workspace/vision_model/poster_state.json"
DATA = "/workspace/vision_ds/shard_*.npz"
TAG = "v1"
N_FRAMES = 4

EPOCH_RE = re.compile(
    r"ep (\d+): loss ([\d.]+) \(d ([\d.]+) s ([\d.]+)\) \| val MAE ([\d.]+)m "
    r"RMSE ([\d.]+)m d1.25 ([\d.]+) mIoU ([\d.eE+-]+|nan) "
    r"c0:([\d.eE+-]+|nan) c1:([\d.eE+-]+|nan) c2:([\d.eE+-]+|nan) c3:([\d.eE+-]+|nan)")

DASHBOARD_URL = os.environ["DASHBOARD_URL"].rstrip("/")
INGEST_TOKEN = os.environ["INGEST_TOKEN"]


def post(payload):
    req = urllib.request.Request(
        DASHBOARD_URL + "/api/vision/ingest",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": "Bearer " + INGEST_TOKEN,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def fnum(s):
    try:
        v = float(s)
        return None if v != v else v  # nan -> None (json-clean)
    except ValueError:
        return None


def parse_log():
    epochs = []
    try:
        with open(LOG) as f:
            for line in f:
                m = EPOCH_RE.search(line)
                if m:
                    g = m.groups()
                    epochs.append({
                        "epoch": int(g[0]), "loss": fnum(g[1]),
                        "loss_d": fnum(g[2]), "loss_s": fnum(g[3]),
                        "mae": fnum(g[4]), "rmse": fnum(g[5]),
                        "d125": fnum(g[6]), "miou": fnum(g[7]),
                        "iou0": fnum(g[8]), "iou1": fnum(g[9]),
                        "iou2": fnum(g[10]), "iou3": fnum(g[11]),
                    })
    except FileNotFoundError:
        pass
    return epochs


def trainer_alive():
    r = subprocess.run(["pgrep", "-f", "vis_[t]rain.py"], capture_output=True)
    return r.returncode == 0


_val_cache = {}


def val_batch():
    """4 fixed frames from 4 different val scenes (split identical to
    vis_train.py: seed-7 shuffle, uniq[:64] test, uniq[64:128] val)."""
    if "batch" in _val_cache:
        return _val_cache["batch"]
    from vis_train import load_shards
    rgb, dep, seg, meta = load_shards(DATA)
    scene_ids = meta[:, 0].astype(int)
    uniq = np.unique(scene_ids)
    rng = np.random.default_rng(7)
    rng.shuffle(uniq)
    val_ids = uniq[64:128]
    picks = []
    for sid in val_ids[:N_FRAMES]:
        idxs = np.where(scene_ids == sid)[0]
        picks.append(idxs[len(idxs) // 2])
    batch = {k: v[picks] for k, v in
             (("rgb", rgb), ("dep", dep), ("seg", seg))}
    _val_cache["batch"] = batch
    return batch


def render_frames(epoch):
    import torch
    from vis_train import VisNet
    net = VisNet()
    net.load_state_dict(torch.load(CKPT, map_location="cpu"))
    net.eval()
    b = val_batch()
    r = torch.from_numpy(b["rgb"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    with torch.no_grad():
        pd, ps = net(r)
    pred_d = np.clip(pd.exp().numpy() * 1000.0, 0, 65535).astype(np.uint32)
    pred_s = ps.argmax(1).numpy().astype(np.uint8)
    frames = []
    for i in range(len(r)):
        rgb = b["rgb"][i].astype(np.uint32)
        packed = ((rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]).ravel().tolist()
        frames.append({
            "rgb": packed,
            "depth": b["dep"][i].astype(np.uint32).ravel().tolist(),
            "seg": b["seg"][i].astype(int).ravel().tolist(),
            "pred_depth": pred_d[i].ravel().tolist(),
            "pred_seg": pred_s[i].astype(int).ravel().tolist(),
        })
    h, w = b["dep"].shape[1], b["dep"].shape[2]
    return {"type": "train_frames", "w": w, "h": h, "epoch": epoch,
            "frames": frames}


def main():
    state = {"posted": [], "ckpt_mtime": 0.0}
    if os.path.exists(STATE):
        try:
            state.update(json.load(open(STATE)))
        except Exception:
            pass
    print("poster up", flush=True)
    while True:
        try:
            epochs = parse_log()
            new = [e for e in epochs if e["epoch"] not in set(state["posted"])]
            for e in new:
                post({"type": "train_epoch", **e})
                state["posted"].append(e["epoch"])
            post({"type": "train_status", "tag": TAG,
                  "status": "training" if trainer_alive() else "finished",
                  "epochs_target": 25,
                  "pid_alive": trainer_alive(),
                  "epochs_posted": len(state["posted"])})
            mt = os.path.getmtime(CKPT) if os.path.exists(CKPT) else 0.0
            if mt > state["ckpt_mtime"]:
                ep = epochs[-1]["epoch"] if epochs else None
                frames = render_frames(ep)
                post(frames)
                state["ckpt_mtime"] = mt
                print(f"posted frames @ epoch {ep}", flush=True)
            json.dump(state, open(STATE, "w"))
        except Exception as e:
            print("poster error:", repr(e), flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
