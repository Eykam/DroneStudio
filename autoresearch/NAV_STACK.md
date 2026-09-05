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

### 4. VIO in the training loop (in-repo implementation - decided 2026-09-05)

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

## Decisions (Eyad, 2026-09-05 12:32 PM)

- **Split: depth+segmentation first (option D), RGB photorealism on rung
  2 (option A).** His call: "figure it out" - default confirmed. The D->A
  order delivers both; the realism path stays the plan, not the first
  rung.
- **Camera targets: match the real hardware.** Stereo pair = 2x Raspberry
  Pi Camera Module 3 (IMX708) at the CAD manifest's 56mm baseline. Sim
  camera model targets: 640x480 @ 30Hz per camera (VIO operating point;
  native 4608x2592 is capture-side only), rolling shutter (IMX708 is
  rolling - per-row time offset in the rasterizer), ~75 deg horizontal
  FOV (standard lens variant; revisit if he mounts the 120 deg wide).
- **VIO: our own implementation, in-repo.** His call: "we should write
  our own better and faster implementation" - no OpenVINS/VINS-Fusion
  wrap. Design consequences: deterministic, headless, runs inside the
  training loop; tight IMU preintegration against the existing 500Hz
  sim IMU; stereo frontend (KLT features + depth from the rasterizer as
  ground truth for supervised signal during development); the estimator
  stays classical so it is testable against sim ground truth before the
  learned nav policy ever sees it.
