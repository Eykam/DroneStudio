# Outer navigation stack: vision policy over the trained inner loop

User direction 2026-09-04 (10:21 PM): next major build is the outer loop.
Stereo camera pair + ToF -> classical VIO/SLAM for state estimation ->
slower learned nav policy (1-10 Hz) emitting setpoints -> the t4 inner-loop
lineage executes them. Goal: full autonomous navigation. Staging (his
ordering): vision sim-to-sim first, HiL on the real FC second, real scenes
last. His gate: rendering realism for transfer.

## Architecture (agreed 10:21)

    cameras (stereo) --+
                       +--> VIO/SLAM (classical) --> state estimate --+
    ToF ---------------+                                              +--> nav policy (1-10 Hz)
    goal / mission --------------------------------------------------+      |
                                                                       setpoint (pos/yaw/vel)
                                                                            |
                                                              inner loop (t4 lineage, ~50Hz)
                                                                            |
                                                                        motor commands

- Inner loop stays as trained: given a setpoint, hover/goto/land without
  crashing. m2 caveat: only after dag7m2-class training closes the
  v1->m2 transfer gap (t4_best transfers at 0.083 under motor_v2 today).
- Estimation stays classical (VIO) so each piece is independently
  testable; only the nav policy is learned.
- Nav policy input: VIO state estimate (+covariance), local depth/occupancy
  summary, goal vector. Output: next setpoint for the inner loop.

## Sim gaps to close, in build order

### 1. Pixels on the CPU box (the hard gate today)

Headless box cannot render: the Studio renderer requires
GL_ARB_bindless_texture; no CPU rasterizer implements it
(RENDERER_FALLBACK.md, investigation complete). Options restated:

- **D - purpose-built software rasterizer (build FIRST).** Procedural
  scenes are analytic primitives (boxes/cylinders/floor): ~300 lines of
  Zig, zero GL, deterministic, produces depth + semantic segmentation at
  training rates inside the existing headless episode API. Unblocks
  vision-policy training on Railway without touching the editor renderer.
- **A - texture arrays (build SECOND).** Replaces bindless with
  sampler2DArray buckets; the smallest change that lets the REAL renderer
  run on llvmpipe. Needed for the photorealism rung (RGB, textures,
  lighting randomization) once depth+seg policies work.

### 2. Sensor models

- Stereo pair: two SensorCamera prefabs at a configurable baseline; add a
  camera model with intrinsics + noise, exposure variation, motion blur,
  rolling shutter (his list). On option D: ray-cast stereo is natural
  (two origins). Rolling shutter is a per-row time offset - cheap in a
  scanline/ray-caster, and it matters for VIO.
- ToF: ray-cast rangefinder array with a noise model (range-dependent
  sigma, dropout on grazing angles / low reflectance, max-range clamp).
  No ToF exists in the codebase today (only camera prefabs).
- IMU already exists for the fast loop; VIO consumes camera + IMU.

### 3. Procedural scenes + domain randomization

- Scenario sampler exists (scenario_sampler.py: extents, obstacles,
  tolerances, hold_s). Extend it to sample VISUAL parameters: texture /
  material properties, lighting direction/intensity/color, clutter
  density, floor/wall geometry families.
- Geometry randomization rides the existing sampler; visual
  randomization rides the new renderer path (option D: primitive
  material IDs -> segmentation + depth noise; option A: real materials).

### 4. VIO in the training loop

- Run VIO at camera rate inside episode rollouts (or precompute per
  frame and replay); the nav policy sees the ESTIMATE, not ground truth.
  This is where sim-to-real transfer is actually won: the policy learns
  to act under estimator drift/noise, not under oracle state.
- Ground-truth state stays available for reward/termination and for the
  inner loop in early experiments (sim-to-sim stage).

### 5. Nav policy training

- Slow loop at 1-10 Hz emitting setpoints; inner loop at 50 Hz (t4
  lineage, after m2-fidelity closes). Episodes: reach goal through
  clutter, no collision, time/energy budget.
- Start: behavior cloning from a classical planner (A*/RRT over the
  known map + VIO state) then RL fine-tune - mirrors the DAgger-first
  lesson from the inner loop (PPO-from-scratch destroyed precision;
  teacher-first worked).

## Staging (his ordering)

1. **Sim-to-sim vision policy**: option D pixels -> depth+seg -> VIO ->
   nav policy -> inner loop (t4 on m2). Success = navigates procedural
   scenes under estimator noise.
2. **Photorealism rung**: option A renderer + visual domain
   randomization; retrain/fine-tune; quantify the depth+seg->RGB gap.
3. **HiL on the real FC**: inner loop on the flight controller against
   sim state; nav policy offboard first.
4. **Real scenes**: field tests.

## Open questions for Eyad

- First vision policy: depth+segmentation (option D, fast, standard) is
  enough - RGB photorealism is rung 2. Confirm that's the split he wants,
  since his message emphasized realism. Our read: he wants the realism
  PATH built, not necessarily RGB-first; D->A delivers both in order.
- Stereo baseline / resolution / FOV targets from the hardware he has in
  mind (drives camera model + VIO choice).
- VIO stack preference: implement in-repo (deterministic, headless) vs
  wrap an existing one (OpenVINS/VINS-Fusion - heavier, but battle-tested).
