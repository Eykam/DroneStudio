# Milestone 1: Headless Episode API

Status: IMPLEMENTED + PARITY-VERIFIED (2026-09-04, commit c79ea5e).
Proposed 2026-09-03, built and verified the same night per his mandate:
"we should definitely 100% be using the zig binary."

Verification (autoresearch/parity_report.json): identical scene + identical
action sequence through QuadNavEnv (numpy) and this binary:
max position divergence 3.1e-5 m over a 61-step episode, velocity divergence
~5e-6 m/s, episode returns match to 5 decimals. Verdict: track.
Smoke: autoresearch/sim_smoke.json (full CEM train+eval path on backend="sim").

Design calls made (delegated authority, rationale in commit c79ea5e):
1. Collider: analytic sphere r=0.3 for the drone (exact QuadNavEnv parity;
   the GLTF convex hull stays with the renderer path). Obstacles are
   analytic spheres.
2. Scenes: procedural, sampled Python-side, passed concrete on reset -
   parity lives in the dynamics, not the RNG.
3. Loop rates: 500 Hz fast loop / 20 Hz policy, matching QuadNavEnv.
4. Actuation v1: PID torque + collective thrust applied directly to the
   body. The motor mixer + motor-lag path is the v1.1 fidelity upgrade,
   deferred until its free parameters are verified against his hardware.

## Why

The auto-researcher's inner loop must run thousands of short sim episodes per
outer-loop generation. The GUI app cannot do this: it is window-first
(ECSManager.init -> GlobalsSystem creates the GLFW window, ResourceManager
compiles all shaders at startup), and its fragment shaders require
GL_ARB_bindless_texture, which no CPU software rasterizer (llvmpipe,
swiftshader) provides. Verified empirically 2026-09-03 on the Railway box:
binary launches under xvfb, GL context inits, vertex shaders pass after the
450 patch, fragment shaders hard-fail on the bindless extension.

## Decision: physics-only episodes first, pixels later

The navigation policy being trained consumes state/VIO-style observations,
not camera frames, in the near-term architecture (classical VIO fast loop +
MLP/transformer nav policy). So episode API v1 runs PHYSICS ONLY: no window,
no GL, no renderer. Vision-based policies become possible later on a GPU box
(bindless works on real NVIDIA drivers) without changing this API.

## Proposed shape

New build target, additive only - no changes to existing systems' behavior:

    zig build headless -Dscene=<path.json>

A new exe (`src/headless_main.zig`) that:

1. Constructs the physics world directly (Bullet via cbullet) WITHOUT
   ECSManager/GlobalsSystem/ResourceManager:
   - drone rigid body: box proxy collider (0.3 m arms, 1.5 kg, inertia
     Ixx=Iyy=0.040, Izz=0.047 - exact values from prefabs/Drone.zig) instead
     of the GLTF-derived convex hull (documented simplification; GLTF loading
     is entangled with the renderer's ResourceManager)
   - static colliders for obstacles from the scene JSON (spheres/boxes)
2. Reuses FlightController's RateController + PIDController verbatim for the
   fast loop at 500 Hz - same code path as the app, not a re-implementation.
   (QuadNavEnv currently re-implements these in numpy; the headless target
   makes that approximation obsolete.)
3. Exposes a stdio JSON-lines protocol, one message per line:
     <- {"cmd":"reset","scene":{...SceneDistribution...},"seed":N}
     -> {"obs":[...15 floats...],"info":{...}}
     <- {"cmd":"step","action":[roll_rate,pitch_rate,yaw_rate,thrust]}
     -> {"obs":[...],"reward":R,"done":false,"info":{...}}
     <- {"cmd":"close"}
   Action semantics identical to env_quad.QuadNavEnv so the inner loop can
   swap backends with a flag (evaluator.py already has `backend=`).
4. Fixed-step clock: dt=1/500, decoupled from wall time - deterministic given
   seed (episodes are reproducible; the archive stays meaningful).

## Work items (est. order)

1. [ ] build.zig: `headless` exe target (cimgui/glad/glfw/ffmpeg EXCLUDED -
      physics + bullet + zigimg only; this is the main build-graph surgery)
2. [ ] src/headless_main.zig: bullet world setup, drone body, scene loader
3. [ ] Extract the collider-from-model path's physics-only half, or accept
      box-proxy collider for v1 (recommend box proxy)
4. [ ] stdio protocol + obs layout parity with QuadNavEnv
5. [ ] SimBinaryEnv (python) driving the binary; evaluator backend="sim"
6. [ ] Parity test: same seed+actions through QuadNavEnv and headless binary
      -> trajectories should track within integration-error tolerance
7. [ ] (GPU box, later) renderer episodes for vision policies

## Open questions for Eyad

- Box-proxy drone collider OK for training episodes, or is the GLTF hull
  load-bearing for the behaviors you want?
- Scene JSON: should it name prefabs (HintzeHall etc.) or stay procedural
  primitives (spheres/boxes/planes)? Procedural primitives match the
  auto-researcher's mutation model; prefabs anchor to real spaces.
- 500 Hz fast loop / 20 Hz policy rates match QuadNavEnv - confirm against
  the 1 kHz pose loop the Pi rig uses.
