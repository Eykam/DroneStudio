#!/usr/bin/env python3
"""vis_gen_dataset.py - dataset generator for the learned depth+seg model.

Vision workstream Phase 1a. Drives dronestudio-headless over JSONL:
scenes sampled with the SAME generator nav training uses
(QuadNavEnv._sample_scene / SceneDistribution), camera poses re-reset per
sample, render emits RGB + metric depth (mm, 65535=sky) + seg classes
(0 ground, 1 sky, 2 obstacle, 3 pad - probe-verified 2026-09-05).

Camera aim convention (probe-verified): yaw = atan2(-dz, dx),
pitch = atan2(dy, hypot(dx,dz)). RENDER TAKES RADIANS - mount yaw/pitch
are passed straight into the rotation math (probe 2026-09-05: any
degree-magnitude value throws the view wildly, pitch=0 centers fine).
meta stores degrees for readability; the render call converts.

Split discipline: shards carry scene_id; train/val/test split BY SCENE,
never by frame. Roll augmentation happens in the training pipeline
(in-plane rotation is exact for depth/seg).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from env_quad import QuadNavEnv
from scene_schema import SceneDistribution
from vis_visual import sample_visual

BIN = "/workspace/zig-out/bin/dronestudio-headless"


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
        try:
            self.call({"cmd": "close"})
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def aim_yaw_pitch(pos, target):
    d = target - pos
    yaw = np.degrees(np.arctan2(-d[2], d[0]))
    pitch = np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2])))
    return float(yaw), float(pitch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, required=True)
    ap.add_argument("--poses-per-scene", type=int, default=48)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene-offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    h = Headless()
    os.makedirs(a.out, exist_ok=True)
    W, H = a.width, a.height
    RGB = np.zeros((a.scenes * a.poses_per_scene, H, W, 3), np.uint8)
    DEP = np.zeros((a.scenes * a.poses_per_scene, H, W), np.uint16)
    SEG = np.zeros((a.scenes * a.poses_per_scene, H, W), np.uint8)
    META = np.zeros((a.scenes * a.poses_per_scene, 9), np.float32)
    visuals = {}
    n = 0
    t0 = time.time()
    for si in range(a.scenes):
        scene_id = a.scene_offset + si
        env = QuadNavEnv(SceneDistribution(), seed=scene_id)
        goal = env.goal.copy()
        goal[1] = max(goal[1], 0.5)  # pad below ground breaks reset
        obs_arr = np.concatenate(
            [env.obs_centers, env.obs_radii[:, None]], axis=1) \
            if len(env.obs_centers) else np.zeros((0, 4))
        ext = float(env.dist.scene_extent)
        visuals[scene_id] = sample_visual(scene_id)  # seeded per scene: split-safe
        vis = visuals[scene_id]
        for pi in range(a.poses_per_scene):
            pos = None
            for _ in range(64):
                cand = np.array([rng.uniform(-ext / 2, ext / 2),
                                 rng.uniform(0.4, 10.0),
                                 rng.uniform(-ext / 2, ext / 2)])
                if len(obs_arr) and (np.linalg.norm(
                        obs_arr[:, :3] - cand, axis=1) - obs_arr[:, 3]).min() < 2.5:
                    continue
                pos = cand
                break
            if pos is None:
                continue
            if rng.random() < 0.45:  # aim at pad with jitter (class balance)
                # re-sample pos in a 4-18m annulus around the goal so the
                # 2m pad disc is comfortably resolvable at 128x96
                for _ in range(64):
                    ang = rng.uniform(0, 2 * np.pi)
                    rr = rng.uniform(4.0, 18.0)
                    cand = np.array([goal[0] + rr * np.cos(ang),
                                     rng.uniform(0.4, 10.0),
                                     goal[2] + rr * np.sin(ang)])
                    if abs(cand[0]) > ext / 2 or abs(cand[2]) > ext / 2:
                        continue
                    if len(obs_arr) and (np.linalg.norm(
                            obs_arr[:, :3] - cand, axis=1) - obs_arr[:, 3]).min() < 2.0:
                        continue
                    pos = cand
                    break
                yaw, pitch = aim_yaw_pitch(pos, goal)
                yaw += float(rng.normal(0, 8))
                pitch += float(rng.normal(0, 5))
            else:                     # free orientation, downward-biased
                yaw = float(rng.uniform(-180, 180))
                pitch = float(rng.uniform(-80, 15))
            pitch = float(np.clip(pitch, -85, 85))
            h.call({"cmd": "reset", "seed": int(rng.integers(1 << 30)),
                    "scene": {"spawn": [float(pos[0]), float(pos[1]), float(pos[2])],
                              "goal": [float(goal[0]), float(goal[1]), float(goal[2])],
                              "obstacles": obs_arr.tolist(),
                              "extent": ext, "max_steps": 10}})
            f = h.call({"cmd": "render", "width": W, "height": H,
                        "yaw": float(np.radians(yaw)),
                        "pitch": float(np.radians(pitch)),
                        "visual": vis})
            rgb = np.array(f["rgb"], dtype=np.uint32)
            RGB[n, :, :, 0] = ((rgb >> 16) & 255).reshape(H, W)
            RGB[n, :, :, 1] = ((rgb >> 8) & 255).reshape(H, W)
            RGB[n, :, :, 2] = (rgb & 255).reshape(H, W)
            DEP[n] = np.array(f["depth"], dtype=np.uint16).reshape(H, W)
            SEG[n] = np.array(f["seg"], dtype=np.uint8).reshape(H, W)
            META[n] = [scene_id, pos[0], pos[1], pos[2], yaw, pitch,
                       goal[0], goal[1], goal[2]]
            n += 1
        print(f"scene {scene_id} done ({n} frames, {time.time()-t0:.0f}s)",
              flush=True)
    h.close()
    out = os.path.join(a.out, f"shard_s{a.seed}_o{a.scene_offset}.npz")
    np.savez_compressed(out, rgb=RGB[:n], depth=DEP[:n], seg=SEG[:n],
                        meta=META[:n])
    vout = out.replace(".npz", ".visuals.json")
    json.dump(visuals, open(vout, "w"), indent=1)
    print(f"saved {out}: {n} frames in {time.time()-t0:.0f}s (+ {vout})", flush=True)


if __name__ == "__main__":
    main()
