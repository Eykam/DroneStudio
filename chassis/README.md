# Chassis auto-researcher

Parametric, 3D-printable 5-inch quad chassis optimized against DroneStudio's sim.

- `chassis.py` - parametric model (build123d / Open CASCADE). Constraints from the sim's FlightController (quad-X, 0.15 m arm, 10 N/motor).
- `evaluate.py` - evaluator: watertight/single-body, FDM overhang, wall thickness, prop clearance, mass + full inertia tensor, hover margin. Emits a scalar score.
- `export_manifest.py` - exports geometry (glTF, meters) + `*.manifest.json` (dronestudio.chassis/1): inertial, collision spec, motor mounts/axes/directions, IMU site, stack pattern.

Every candidate design is committed here with its evaluation result (see git history).
