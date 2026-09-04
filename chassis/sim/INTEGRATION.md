# Sim integration: manifest-driven Drone prefab

`prefabs/Drone.zig` currently hardcodes: GLTF path, mass 1.5 kg, an estimated
inertia tensor, quad-X mixer arm length 0.15 m, 10 N/motor. The chassis-branch
change loads `*.manifest.json` (dronestudio.chassis/1) via ChassisManifest.zig
and replaces them:

1. `loadGLTFModelCached(alloc, manifest.geometry.file)` (resolve relative to the manifest dir)
2. `RigidBodyComponent.init(manifest.totalMassKg(), collider.bullet_shape.?)`
3. `rigid_body.setInertia(...manifest.diagonalInertia())` - replaces the arm_length estimate block
4. FlightController params: `motor_arm_length = manifest.armLengthM()`,
   `motor_max_thrust = manifest.maxThrustPerMotorN()`,
   `motor_time_constant = manifest.motorTimeConstantS()`, `motor_drag_ratio = manifest.motorDragRatio()`
5. Collision: manifest.collision.type picks ConvexHull vs VHACD decomposition (max_hulls)
6. IMU site from manifest.imu.position_m

Open question to verify on first round-trip: axis convention. CAD exports Z-up
glTF; the sim's OpenGL side is Y-up with an NED adapter in FlightController.
If the loaded model lies flat, apply a -90 deg X rotation at import (gltf
loaders often bake this) - check against assets/drone/scene.gltf orientation.

Build check on the CAD box: `zig build -Dcuda=false -Dpi=false` (CUDA and Pi
targets are off-by-default options, so a CPU-only container compiles the
desktop sim and its physics tests).
