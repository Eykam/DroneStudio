# Sensor simulation library - design (v0 landed 2026-09-06)

## Goal (user directive)
Every simulated sensor behind one extensible interface; per-part error models
anchored in the actual datasheet of the part the EE design uses; mount pose and
output rate as first-class config; config-level part swaps.

## Architecture (autoresearch/sensors/)

base.py
  SimSensor (ABC): name, spec (datasheet dict), mount, seeded RNG, rate
  gating via due(t). sample(t, dt, world_state, env) -> Measurement.
  Mount: pos (body-frame m) + rot (sensor-to-body 3x3). Sensors with lever
  arms get correct centrifugal/tangential terms (imu.ideal does this).
  Measurement: name, t (with jitter), rate_hz, channels dict.
  SimEnvironment: temperature(t) profile; ambient-light scalar for ToF.
  Convention: world = sim frame (Y up), quats [x,y,z,w] body->world.

specs/<part>.py
  One dict per real part, every value traced to the datasheet (citations in
  comments) or explicitly labeled [model]/LITERATURE-CLASS when the datasheet
  is silent. This is the swap point: new part = new spec file, zero code.

imu.py  - SimIMU: ideal(world_state) -> corrupt() chain:
  scale/cross-axis matrices -> OU bias drift + tempco -> vibration
  (motor-harmonic + broadband, DLPF-anti-alias-attenuated before injection) ->
  DLPF -> white noise -> quantization. All stages independently anchored.
ekf.py  - 15-state ESKF (p, v, q, b_g, b_a; Sola formulation). Predicts from
  the SIMULATED IMU only. Updates: position, attitude, ground_range
  (rangefinder w/ attitude Jacobian), mag (body-frame field vs world field).
tof.py  - SimToF: per-zone grid from spec FoV, backend-injected ray cast
  (analytic caster, rendered depth, or mesh - same sensor code), per-zone
  range+status, sigma = base + k*r^2 scaled by ambient+temp, dropouts past
  80% of mode max.
registry.py - build_suite(vehicle_config): config dict -> sensor objects at
  their mount poses; the config-level swap entry point.

## Parts currently modeled (anchored to the EE design)
- MPU-9250 (PS-MPU-9250A-01 v1.1): gyro 0.1dps-rms @92Hz DLPF (self-test
  matches: 0.101), NSD 0.01dps/sqrtHz, ZRO init +/-5dps, tempco +/-30dps
  envelope, cross-axis +/-2%, FS +/-2000dps/+/-16g, 16-bit. Includes AK8963
  mag path ([model]-class noise; 0.15uT/LSB 16-bit).
- VL53L9CX (DS14879 Rev 7): FoV 55x42 (Table 3), grids 54x42..12x10, precision
  <5cm-8.8m / ambient 45cm-8.5m, <=100Hz, Table 9 mode profiles, -30..70C.
  Noise params [model] (FlightSense family behavior; DS14879 has no curves).

## Verified behaviors
- ESKF: clean-circle 0.0199m/8s predict-only; corrupted+fused 0.087m final.
- Fusion v1 (fusion_chained_g.py): gyro+mag+gravity own attitude (yaw needs
  the mag - unobservable otherwise, 107deg mean without it, 11-15deg with),
  chained VO position with growing R, ToF altimeter gated <15deg tilt.
- Known limits: VO frontend rotation tail (p99 24deg) at >150dps; chained
  attitude leaks slowly on violent scenes (re-anchoring = future work).

## Extension points (next parts)
- Barometer (ee-flight part TBD), optical flow, GPS - each = spec file +
  one sensor class + optional EKF update method.
- ToF ring: 8x VL53L9CX at CAD carrier poses (ring harness contract).
