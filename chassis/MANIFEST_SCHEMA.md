# dronestudio.chassis manifest schema (CAD researcher -> sim contract)

Producer: CAD auto-researcher (box #2, this branch, `chassis/export_manifest.py`).
Consumer: DroneStudio sim (auto-researcher branch work: manifest-driven Drone prefab).
Zig loader draft: `chassis/sim/ChassisManifest.zig`; integration notes: `chassis/sim/INTEGRATION.md`.
If the shape changes, bump the `schema` field AND update this file in the same commit.

Current version: **dronestudio.chassis/1.2**

## Design intent

The manifest is a complete rigid-body description - the sim should NOT derive
mass properties itself. Everything the flight dynamics need is precomputed
from the CAD B-rep (exact) plus an explicit payload model.

## Top-level fields

- `schema`: "dronestudio.chassis/1.2"
- `name`: variant id, matches the git commit / dashboard record
- `geometry.file`: binary glTF (GLB), meters, +X forward, +Z up.
  - `geometry.sim_file` (1.2): sim-ready GLB - NO quantized/meshopt required
    extensions (the sim GLTF loader rejects those), Z-up -> Y-up root rotation
    baked in. Use this for visual load; geometry.file stays the dashboard one.
  NOTE the axis-convention open question in INTEGRATION.md (sim is Y-up
  OpenGL-side; verify on first load against assets/drone/scene.gltf).
- `material`: print material (PETG: density 1240 kg/m3, E 2100 MPa, yield 50 MPa)
- `dynamics` - the composed rigid body the sim uses AS-IS:
  - `total_mass_kg`: frame + 4 motors + payload (stack, battery, cameras)
  - `com_m`: center of mass, meters, in the GLB frame
  - `inertia_about_com_kgm2`: FULL 3x3 tensor about the CoM (ixx..iyz).
    AXES ARE IN THE GLB FRAME (+X forward, +Z up) - same frame as com_m and
    the geometry. Any frame conversion (e.g. to the sim's Y-up OpenGL side)
    happens in the loader, never in this file.
  - `composition[]`: per-part mass/CoM breakdown (frame, motors, payload items)
- `aero`:
  - `projected_area_m2 {x,y,z}`: cross-section per body axis (for drag)
  - `cd_flat_plate_estimate`: 1.1 (flat-plate Cd; refine later)
  - drag model hint: F = 0.5 * 1.225 * cd * area * v^2 per axis
- `collision`: `convex_hull` (default) or `vhacd` + `max_hulls` - matches the
  sim's existing ColliderComponent paths (VHACD is already vendored)
- `motors[]` (sim mixer order, quad-X: M1 FR CW, M2 FL CCW, M3 RL CW, M4 RR CCW):
  - `position_m`, `axis` (thrust direction, +Z), `direction`
  - `mass_kg`, `prop_diameter_m`
  - `max_thrust_n`, `time_constant_s` (first-order lag), `drag_ratio` (ktau/kT)
    - these map 1:1 onto FlightController.FlightControllerParams
- `imu` (1.2): REAL pose from components.py placement, no longer hardcoded:
  - `position_m`: IMU (mpu9250) site in the GLB frame
  - `rotation_quat_xyzw`: IMU orientation (identity when mounted axis-aligned)
  - `offset_from_com_m`: `imu.position_m - dynamics.com_m` - lever arm for
    correcting readings: a_com = a_imu - alpha x r - omega x (omega x r)
- `cameras[]` (1.2): per camera: `id`, `lens_origin_m`, `lens_axis` (+X forward),
  `hfov_deg`, `vfov_deg` (Pi Camera Module 3: 66.3h x 41.6v). Lenses point out the
  nose apertures; the CAD evaluator gates that the FOV pyramid clears the frame.
- `stack`: 30.5 mm pattern info for the FC/ESC mount

## What changes per design vs what is fixed

Per design: geometry, dynamics, aero, collision. Fixed by the vehicle spec
(5-inch quad): motor params (10 N max, 40 ms lag, 0.15 drag ratio), prop
diameter, motor masses, payload masses. The auto-researcher may mutate motor
params only if the vehicle spec changes (e.g. different motors chosen).
