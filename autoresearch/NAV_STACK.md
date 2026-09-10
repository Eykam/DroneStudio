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
  LANDED 2026-09-09 (vision_raster.zig readTof/scanTof + headless
  tof/tof_scan cmds): the decided 8-sensor suite - 4 cardinal nav
  (0/90/180/270 deg) + 4 diagonal arm/proximity monitors (45-deg seats,
  measured 56% cone-blocked at the 40-140mm arm baseline, 2:1 weighted).
  Datasheet-true per TOF_SIM_SCOPE.md: VL53L9CX multizone dToF - 55x42
  deg FoV, <5cm-9m range, up to 100Hz, zone grids up to 54x42 (binned
  8x8 default). Noise v0: sigma = 5mm + 3mm/m^2 * range^2 (k ESTIMATE),
  grazing-incidence ramp, far-dropout ramp past 80% of max range,
  ambient-light factor, per-zone arm occlusion. Timing (update rate /
  integration latency) is a stub until estimator integration.
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

## Rung-2 kickoff decisions (agent engineering call, 2026-09-09)

- **RGB rasterizer path for rung 2: extend option D (vision_raster.zig),
  NOT option A.** The 2026-09-05 note mapped rung 2 to option A; parent
  re-opened A-vs-D as the builder's call at kickoff. Reasons: (1) D is
  built, deterministic, and already generates RGB+depth+seg at training
  rates (v1 dataset + live scenario streaming run on it); (2) rung 2's
  deliverable is visual domain randomization for depth+seg robustness,
  and shade() parameterization (sun dir/intensity/color, ambient split,
  per-class albedos, floor texture family, fog density/color, sky
  gradient) exercises exactly the invariance axes the model needs;
  (3) option A's prerequisite (bindless -> sampler2DArray surgery on the
  real renderer) plus llvmpipe throughput at stereo 640x480 training
  rates is unmeasured risk on the critical path. Option A remains the
  photorealism-rung plan; re-evaluate with raster_smoke.zig llvmpipe
  numbers before that rung commits.
- **Visual domain randomization design**: scene JSON carries a `visual`
  block (seeded per scene_id so train/val/test scene splits stay clean);
  scenario_sampler.py samples it alongside geometry; headless_main.zig
  passes it into RasterScene; shade() consumes it. Geometry
  randomization unchanged (existing sampler). Deterministic per seed.

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

## Control architecture note (2026-09-05): scripted precision-land module

The inner control loop is no longer pure-learned: terminal descent is a
SCRIPTED controller (t4_pilot teacher position-PD) composed with the learned
policy at the actuator level (autoresearch/t4_hybrid.py). Handoff boundary:
engage alt<=1.4m AND dxz<=max(0.75, 1.5*pad_radius); release alt>3.0 or
dxz>3r; cold module state at handoff. Implications for the vision stack:
the down-cam/ToF landing aid must serve the scripted module's state
estimate during terminal descent, not a learned policy - its inputs are
metric (pad-relative dxz, alt), which matches the depth/seg renderer's
metric output (1950mm from 2m verified). A vision-based pad detector can
later replace the simulator's privileged dxz at the same boundary without
changing the module's control law.

## Learned vision workstream (user-directed 2026-09-05 8:15 PM)

Direction (his words): learn depth + segmentation; end goal is SLAM
on-device, with SLAM issuing setpoints the trained nav policy executes.
Settles classical-vs-learned: learned, distilled from ray-caster GT.

Phase 1 - learned depth+seg (v1 COMPLETE 9:15 PM):
- Data: autoresearch/vis_gen_dataset.py - drives dronestudio-headless
  JSONL (reset scene spec + render per pose). 30,720 frames @128x96,
  640 scenes, split BY SCENE (512/64/64 train/val/test).
  Gotchas encoded in-code: render yaw/pitch are RADIANS; seg classes
  0 sky, 1 ground, 2 obstacle, 3 pad (probe-verified).
- Model: autoresearch/vis_train.py - 1.10M-param multi-task UNet,
  log-depth L1 + weighted 4-class CE, CPU torch (/workspace/venv-vision).
- Held-out TEST (64 scenes, autoresearch/vis_eval.py): depth MAE 0.598m,
  RMSE 1.05m, delta<1.25 0.968; seg mIoU 0.9971 (all classes >= 0.99).
  val ~= test: no overfit, scene split held.
- Latency: 11.8ms single-core server CPU (~85 fps). PROXY - not a Pi 5
  measurement; benchmark on-device before any real-time claim.
- Live view: dashboard /vision page (metrics curves + GT-vs-prediction
  panels), fed by autoresearch/vis_dashboard_streamer.py - a log tailer +
  checkpoint watcher that never touches the training process.
- Known issue: late-epoch val depth oscillation (ep 23-24 collapse to
  MAE ~6m while the ep-20 best sits at 0.6m). Next iteration: sky as a
  separate classification mask so regression fits finite depth only;
  lower final lr.
- Sim2real: trained on analytic primitives; will NOT transfer to real
  camera imagery without texture/lighting domain randomization
  (SceneDistribution has the knobs; the rasterizer does not implement
  them yet) plus real-data fine-tuning.

Phase 2 - SLAM (next): monocular VO on learned metric depth + IMU
fusion, evaluated vs sim GT trajectory (ATE/RPE); then mapping and
goal/setpoint generation.
Phase 3 - setpoint interface: SLAM pose + goal into the existing hybrid
policy (learned goto/hover + scripted land module unchanged); on-device
packaging last.
