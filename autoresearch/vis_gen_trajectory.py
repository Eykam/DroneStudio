"""Phase 2 VO data: SEQUENTIAL frames along smooth flight trajectories.

Phase 1's dataset is random poses per scene (classification-grade); VO
needs temporally ordered frames with GT poses. Per scene: sample 3-5
clearance-checked waypoints, Catmull-Rom interpolate at ~10 Hz, render
depth+rgb per pose (reset+render, same JSONL path as vis_gen_dataset),
record GT pose per frame.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from env_quad import QuadNavEnv
from scene_schema import SceneDistribution

BIN = "/workspace/zig-out/bin/dronestudio-headless"
W, H = 128, 96
HFOV_DEG = 75.0

class Headless:
    def __init__(self):
        self.p = subprocess.Popen([BIN], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
    def call(self, m):
        self.p.stdin.write(json.dumps(m) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("headless exited")
        return json.loads(line)
    def close(self):
        try: self.call({"cmd": "close"})
        except Exception: pass
        try: self.p.wait(timeout=5)
        except Exception: self.p.kill()

def catmull_rom(pts, n):
    """Uniform Catmull-Rom through pts -> n samples (endpoints clamped)."""
    P = np.vstack([pts[0], pts, pts[-1]]).astype(np.float64)
    out = []
    segs = len(P) - 3
    for i in range(segs):
        p0, p1, p2, p3 = P[i:i + 4]
        for t in np.linspace(0, 1, int(np.ceil(n / segs)), endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                              (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out[:n])

def clear(pos, obs_arr, margin):
    if not len(obs_arr):
        return True
    return (np.linalg.norm(obs_arr[:, :3] - pos, axis=1) - obs_arr[:, 3]).min() >= margin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--poses-per-scene", type=int, default=60)
    ap.add_argument("--scene-offset", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="/workspace/vision_model/traj")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    n_total = a.scenes * a.poses_per_scene
    RGB = np.zeros((n_total, H, W, 3), np.uint8)
    DEP = np.zeros((n_total, H, W), np.uint16)
    META = np.zeros((n_total, 9), np.float32)
    h = Headless()
    t0 = time.time()
    n = 0
    for si in range(a.scenes):
        scene_id = a.scene_offset + si
        env = QuadNavEnv(SceneDistribution(), seed=scene_id)
        obs_arr = np.concatenate(
            [env.obs_centers, env.obs_radii[:, None]], axis=1) \
            if len(env.obs_centers) else np.zeros((0, 4))
        ext = float(env.dist.scene_extent)
        # waypoints with clearance
        wps = []
        for _ in range(5):
            for _try in range(64):
                cand = np.array([rng.uniform(-ext / 3, ext / 3),
                                 rng.uniform(0.8, 5.0),
                                 rng.uniform(-ext / 3, ext / 3)])
                if clear(cand, obs_arr, 2.0):
                    wps.append(cand)
                    break
        if len(wps) < 3:
            continue
        path = catmull_rom(np.array(wps), a.poses_per_scene)
        # per-pose clearance: drop frames that clip an obstacle
        for fi, pos in enumerate(path):
            if not clear(pos, obs_arr, 0.8):
                continue
            if fi == 0:
                yaw = float(np.degrees(np.arctan2(-(path[1] - pos)[2], (path[1] - pos)[0])))
            else:
                d = pos - path[fi - 1]
                yaw = float(np.degrees(np.arctan2(-d[2], d[0]))) if np.hypot(d[0], d[2]) > 1e-6 else yaw
            pitch = float(np.clip(np.degrees(np.arctan2(-(pos[1] - 1.5), 4.0)), -45, 15))
            h.call({"cmd": "reset", "seed": int(rng.integers(1 << 30)),
                    "scene": {"spawn": [float(pos[0]), float(pos[1]), float(pos[2])],
                              "goal": [float(path[-1][0]), 0.0, float(path[-1][2])],
                              "obstacles": obs_arr.tolist(),
                              "extent": ext, "max_steps": 10}})
            f = h.call({"cmd": "render", "width": W, "height": H,
                        "yaw": float(np.radians(yaw)),
                        "pitch": float(np.radians(pitch)),
                        "hfov_deg": HFOV_DEG})
            rgb = np.array(f["rgb"], dtype=np.uint32)
            RGB[n, :, :, 0] = ((rgb >> 16) & 255).reshape(H, W)
            RGB[n, :, :, 1] = ((rgb >> 8) & 255).reshape(H, W)
            RGB[n, :, :, 2] = (rgb & 255).reshape(H, W)
            DEP[n] = np.array(f["depth"], dtype=np.uint16).reshape(H, W)
            META[n] = [scene_id, pos[0], pos[1], pos[2], yaw, pitch, fi, ext, len(obs_arr)]
            n += 1
        print(f"scene {scene_id} done ({n} frames, {time.time()-t0:.0f}s)", flush=True)
    h.close()
    out = os.path.join(a.out, f"traj_s{a.seed}_o{a.scene_offset}.npz")
    np.savez_compressed(out, rgb=RGB[:n], depth=DEP[:n], meta=META[:n],
                        intrinsics=json.dumps({"w": W, "h": H, "hfov_deg": HFOV_DEG}))
    print(f"saved {out}: {n} frames in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
