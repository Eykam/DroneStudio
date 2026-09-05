# T5: sustained target-facing (user directive 2026-09-04)

User: facing should be rewarded DURING the whole trajectory, not just achieved
by the end (motivation: sensors that must face a given direction). Keep
outcome-dominant balance and the floors.

## Root cause of late facing

The scripted teacher (t4_pilot.py) refined yaw only when |yaw_err| < 2 rad and
turned yaw OFF at the atan2 antipode, so demos faced the target only near the
goal. BC/DAgger inherit whatever the demos do.

## Changes

1. t4_pilot.py teacher: far (>2*radius) + misaligned now commits to a
   fixed-sign turn at 0.45 * MAX_YAW (fixed sign dodges the antipode
   bang-bang; 0.6 destabilized the position loop in smoke). Refine-band clip
   0.04 -> 0.2. All 4 t4_smoke cells pass (goto t0 29 steps, was 49).
2. headless_main.zig: Scene.face_reward_w - per-step reward +=
   w * max(0, cos(yaw_err)) against the CURRENT target (waypoint while
   pending, else goal). Bounded by w * steps; +10/-5 outcome terms stay
   dominant. Verified wired: AUTORESEARCH_FACE_REWARD=0.01 pays +0.22 on the
   sanity episode vs 0. env_sim.py passes the env var into the scene.
3. t4_faceeval.py: en-route facing metric per eval cell - face20 = fraction
   of steps with |yaw_err| < 20 deg, plus mean |yaw_err| (from the yaw-frame
   rel vector in obs v3). Posts face20_{sc}_t{t} series.

## Results (heldout cells, chain rerun with new teacher, face reward 0.01)

selected: dag2_r2 -> /workspace/t4_best.json (pre-face best backed up at
/workspace/t4_best_pre_facefix.json)

| metric | pre-face dag2_r4 | face dag2_r2 |
|---|---|---|
| succ ALL | 0.633 | 0.672 |
| face20 ALL | 0.345 | 0.679 |
| mean yaw err ALL | 1.151 rad | 0.424 rad |
| goto_t0 | 0.938 | 1.0 |
| hover_t0 | 0.812 | 0.938 |
| land_t0 | 0.312 | 0.75 |
| land_t2 | 0.062 | 0.25 |

Floors held: goto_t0 1.0 >= 0.9, hover_t0 0.938 >= 0.6.
Known weak spot: goto_t3 face20 0.18 (hard tier w/ waypoints).
Champion (bc_ppo_v2_best) and promotion flip untouched.
