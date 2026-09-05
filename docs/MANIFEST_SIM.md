# Consuming CAD chassis manifests in the sim (schema 1.2)

The CAD auto-researcher (chassis branch) publishes `dronestudio.chassis/1.2`
manifests (e.g. `chassis/snapshots/v22-g21a/manifest.json`). The sim consumes
them on the **auto-researcher** branch via `Studio/src/core/ChassisManifest.zig`.

## How to enable

```sh
DRONE_CHASSIS_MANIFEST=/path/to/manifest.json ./zig-out/bin/Studio
```

Unset or unloadable -> the Drone prefab falls back to the legacy hardcoded
mass (1.5 kg), point-mass inertia, CoM-centered IMU, and 75 mm stereo baseline.
Nothing about the training fixture path or the headless sim changes.

## What gets wired (prefabs/Drone.zig)

| Manifest field            | Sim consumer                                             |
|---------------------------|----------------------------------------------------------|
| dynamics.total_mass_kg    | RigidBodyComponent mass                                  |
| dynamics.inertia (diag)   | RigidBodyComponent.setInertia                            |
| imu.offset_from_com_m     | IMUSensorComponent.pos_body (lever-arm correction, below)|
| imu.rotation_quat_xyzw    | IMUSensorComponent.rot_body (accel/gyro die frame)       |
| cameras[] (pose + FOV)    | SensorCamera poses, CameraModule hfov, SLAM baseline     |

v22-g21a values: mass 0.526 kg, arm 0.150 m, IMU offset
(+25.7, -0.001, +18.1) mm, stereo pair Pi Cam 3 Standard (66.3 x 41.6 deg)
at (85, -28, 14.5) / (85, +28, 14.5) mm -> baseline 56 mm (was 75 mm).

## IMU lever-arm correction (components/IMUSensor.zig)

`generateSample` now translates specific force from the CoM to the mount
point when `pos_body != 0` (field `lever_arm_enabled`, default true):

    a_imu = a_com + alpha x r + omega x (omega x r)     (body frame)

alpha is finite-differenced from body-frame omega at the 1 kHz sample rate
(omega updates at the 500 Hz physics rate, so alpha is piecewise stepwise).
Matches the estimator-side formula in the CAD schema doc
(a_com = a_imu - alpha x r - omega x (omega x r)).
The AK8963 mag die is rotated in-package vs the accel/gyro die; the manifest
transform is the accel/gyro frame and the mag remap stays in the estimator.

## Frame conventions - VERIFY ON FIRST DESKTOP RUN

- CAD glTF: +X fwd, +Z up (verified: v22-g21a GLB has no root correction node;
  mesh bounding box is thin in Z).
- Sim scene: OpenGL, +Y up. Mapping used: (x, y, z)_glb -> (x, z, -y)_sim,
  quaternions (x, y, z, w) -> (x, z, -y, w).
- OPEN: the mapping assumes the drone scene model nose is +X in the sim
  scene. Confirm visually (camera frustum children make lens poses visible):
  lenses should sit at the nose, baseline across it, FOV forward.

## GLB compatibility (BLOCKER for loading the CAD model itself)

v22-g21a `chassis.glb` marks `KHR_mesh_quantization` and
`EXT_meshopt_compression` as **required** extensions. The sim loader
(core/GLTF.zig) supports neither, so the CAD model cannot be loaded yet.
Options: CAD exports an uncompressed GLB (also add a Z-up -> Y-up root
rotation), or the sim adds a meshopt decoder. Tracked with the CAD agent
via the parent.
