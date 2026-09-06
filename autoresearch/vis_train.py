#!/usr/bin/env python3
"""vis_train.py - learned depth+seg model (vision Phase 1b).

Small multi-task conv net trained on ray-caster GT (vis_gen_dataset.py):
shared encoder, metric depth head (log-space L1) + 4-class seg head
(0 sky, 1 ground, 2 obstacle, 3 pad). CPU torch. Split BY SCENE.

Sky handling: depth GT 65535mm = sky; trained as log-depth of 65.535m so
the model learns it as "far", metrics reported separately for sky.
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_shards(pattern):
    rgb, dep, seg, meta = [], [], [], []
    for f in sorted(glob.glob(pattern)):
        d = np.load(f)
        rgb.append(d["rgb"]); dep.append(d["depth"])
        seg.append(d["seg"]); meta.append(d["meta"])
    return (np.concatenate(rgb), np.concatenate(dep),
            np.concatenate(seg), np.concatenate(meta))


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.c(x)


class VisNet(nn.Module):
    """~1.6M params. 96x128 -> enc /16 -> dec to full res, two heads."""

    def __init__(self):
        super().__init__()
        ch = (24, 48, 96, 192)
        self.e0 = ConvBlock(3, ch[0])      # 96x128
        self.e1 = ConvBlock(ch[0], ch[1])  # 48x64
        self.e2 = ConvBlock(ch[1], ch[2])  # 24x32
        self.e3 = ConvBlock(ch[2], ch[3])  # 12x16
        self.pool = nn.MaxPool2d(2)
        self.d2 = ConvBlock(ch[3] + ch[2], ch[2])
        self.d1 = ConvBlock(ch[2] + ch[1], ch[1])
        self.d0 = ConvBlock(ch[1] + ch[0], ch[0])
        self.depth_head = nn.Conv2d(ch[0], 1, 1)
        self.seg_head = nn.Conv2d(ch[0], 4, 1)

    def forward(self, x):
        s0 = self.e0(x)
        s1 = self.e1(self.pool(s0))
        s2 = self.e2(self.pool(s1))
        b = self.e3(self.pool(s2))
        u = self.d2(torch.cat([F.interpolate(b, scale_factor=2, mode="nearest"), s2], 1))
        u = self.d1(torch.cat([F.interpolate(u, scale_factor=2, mode="nearest"), s1], 1))
        u = self.d0(torch.cat([F.interpolate(u, scale_factor=2, mode="nearest"), s0], 1))
        return self.depth_head(u).squeeze(1), self.seg_head(u)


def augment(rgb, dep, seg):
    # horizontal flip: exact for depth/seg
    if torch.rand(()) < 0.5:
        rgb = torch.flip(rgb, [2]); dep = torch.flip(dep, [1]); seg = torch.flip(seg, [1])
    # brightness jitter (sim2real headroom)
    rgb = torch.clamp(rgb * (0.8 + 0.4 * torch.rand(())), 0, 1)
    return rgb, dep, seg


def metrics(pred_logd, seg_logits, gt_logd, gt_seg, gt_sky):
    pred_d = pred_logd.exp()
    gt_d = gt_logd.exp()
    nonsky = ~gt_sky
    mae = (pred_d[nonsky] - gt_d[nonsky]).abs().mean().item() if nonsky.any() else 0.0
    rmse = ((pred_d[nonsky] - gt_d[nonsky]) ** 2).mean().sqrt().item() if nonsky.any() else 0.0
    ratio = torch.maximum(pred_d[nonsky] / gt_d[nonsky], gt_d[nonsky] / pred_d[nonsky])
    d125 = (ratio < 1.25).float().mean().item() if nonsky.any() else 0.0
    pred_seg = seg_logits.argmax(1)
    ious = {}
    for c in range(4):
        inter = ((pred_seg == c) & (gt_seg == c)).sum().item()
        union = ((pred_seg == c) | (gt_seg == c)).sum().item()
        ious[c] = inter / union if union else float("nan")
    return mae, rmse, d125, ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/vision_ds/shard_*.npz")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-scenes", type=int, default=64)
    ap.add_argument("--test-scenes", type=int, default=64)
    ap.add_argument("--out", default="/workspace/vision_model")
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(min(32, os.cpu_count()))
    os.makedirs(a.out, exist_ok=True)

    rgb, dep, seg, meta = load_shards(a.data)
    scene_ids = meta[:, 0].astype(int)
    uniq = np.unique(scene_ids)
    rng = np.random.default_rng(7)
    rng.shuffle(uniq)
    test_ids = set(uniq[:a.test_scenes].tolist())
    val_ids = set(uniq[a.test_scenes:a.test_scenes + a.val_scenes].tolist())
    tr = ~np.isin(scene_ids, list(test_ids | val_ids))
    va = np.isin(scene_ids, list(val_ids))
    print(f"scenes {len(uniq)} | train {tr.sum()} val {va.sum()} test {(~tr & ~va).sum()} frames", flush=True)

    def tensors(sel):
        r = torch.from_numpy(rgb[sel].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        d = torch.from_numpy(dep[sel].astype(np.float32) / 1000.0)  # meters
        s = torch.from_numpy(seg[sel].astype(np.int64))
        return r, d, s

    rtr, dtr, str_ = tensors(tr)
    rva, dva, sva = tensors(va)
    net = VisNet()
    nparams = sum(p.numel() for p in net.parameters())
    print(f"params: {nparams/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    # class weights: rough inverse-freq, capped (sky .23 ground .72 obs .04 pad .012)
    wseg = torch.tensor([1.0, 0.4, 4.0, 8.0])
    LOGSKY = float(np.log(65.535))

    best = None
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(rtr))
        tl = td = ts = 0.0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            idx = perm[i:i + a.bs]
            r, d, s = rtr[idx], dtr[idx], str_[idx]
            r, d, s = augment(r, d, s)
            logd = torch.log(d.clamp(0.3, 65.535))
            opt.zero_grad()
            pd, ps = net(r)
            loss_d = F.l1_loss(pd, logd)
            loss_s = F.cross_entropy(ps, s, weight=wseg)
            loss = loss_d + 0.5 * loss_s
            loss.backward()
            opt.step()
            tl += loss.item(); td += loss_d.item(); ts += loss_s.item()
        sched.step()
        net.eval()
        with torch.no_grad():
            pd, ps = net(rva[:2000])
            logd = torch.log(dva[:2000].clamp(0.3, 65.535))
            mae, rmse, d125, ious = metrics(pd, ps, logd, sva[:2000], dva[:2000] > 65.0)
        miou = float(np.nanmean(list(ious.values())))
        print(f"ep {ep}: loss {tl:.1f} (d {td:.1f} s {ts:.1f}) | val MAE {mae:.2f}m RMSE {rmse:.2f}m d1.25 {d125:.3f} mIoU {miou:.3f} " +
              " ".join(f"c{c}:{ious[c]:.2f}" for c in range(4)), flush=True)
        score = d125 + miou
        if best is None or score > best[0]:
            best = (score, ep)
            torch.save(net.state_dict(), os.path.join(a.out, f"visnet_{a.tag}_best.pt"))
            with open(os.path.join(a.out, f"visnet_{a.tag}_best.json"), "w") as f:
                json.dump({"epoch": ep, "mae": mae, "rmse": rmse, "d125": d125,
                           "miou": miou, "ious": ious, "params": nparams}, f)
    print(f"best: ep {best[1]} score {best[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
