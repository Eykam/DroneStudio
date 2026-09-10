#!/usr/bin/env python3
"""vis_visual.py - visual domain randomization sampler (rung-2).

Samples the `visual` block consumed by vision_raster.zig shade(). Seeded
per scene_id so every regeneration of a scene produces the SAME visual
parameters and train/val/test scene splits stay clean. Geometry
randomization stays in vis_gen_dataset / scene_schema (unchanged).
"""
import numpy as np

def sample_visual(scene_id: int, tex: bool = False) -> dict:
    rng = np.random.default_rng((int(scene_id) ^ 0x5E1A1) & 0x7FFFFFFF)
    # sun: azimuth anywhere, elevation 20-80 deg (y-up)
    az = rng.uniform(0, 2 * np.pi)
    el = np.radians(rng.uniform(20, 80))
    sun_dir = [float(np.cos(el) * np.cos(az)), float(np.sin(el)), float(np.cos(el) * np.sin(az))]
    tint = np.clip(rng.normal(1.0, 0.10, 3), 0.75, 1.3)          # per-channel color cast
    def vary(base, lo, hi, floor):
        return [float(np.clip(b * rng.uniform(lo, hi) * t, floor, 255.0)) for b, t in zip(base, tint)]
    sky_lo = vary([26.0, 34.0, 46.0], 0.5, 1.8, 8)               # dusk..bright horizon band
    sky_hi = vary([92.0, 108.0, 126.0], 0.6, 1.6, 25)
    fog_col = [(a + b) / 2 for a, b in zip(sky_lo, sky_hi)]      # fog tracks the sky
    d = {
        "sun_dir": sun_dir,
        "ambient": float(rng.uniform(0.18, 0.55)),
        "fog_scale": float(rng.uniform(20.0, 90.0)),             # denser..clearer
        "sky_lo": sky_lo,
        "sky_hi": sky_hi,
        "fog_col": fog_col,
        "floor_col": vary([150.0, 148.0, 142.0], 0.5, 1.15, 20),
        "obstacle_col": vary([92.0, 106.0, 124.0], 0.5, 1.25, 15),
        "goal_col": vary([30.0, 150.0, 84.0], 0.6, 1.2, 15),
        "checker_m": 0.0 if rng.random() < 0.6 else float(rng.uniform(0.5, 3.0)),
        "checker_gain": float(rng.uniform(0.7, 0.92)),
        "exposure": float(rng.uniform(0.7, 1.35)),
    }
    if tex:
        # procedural texture mode (Phase 2k): world-anchored noise everywhere +
        # floor planks on most scenes; checker still possible per base sampling.
        d["noise_gain"] = float(rng.uniform(0.18, 0.38))
        d["noise_scale_m"] = float(rng.uniform(0.12, 0.45))
        d["plank_m"] = 0.0 if rng.random() < 0.35 else float(rng.uniform(0.08, 0.18))
        d["plank_gain"] = float(rng.uniform(0.45, 0.7))
    return d
