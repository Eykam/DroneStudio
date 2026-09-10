# Rung-3 HiL design: real FC against sim physics (software prep)

Scope (parent, 2026-09-10): HiL sim-in-loop against the modeled FC stack.
Nav policy stays offboard. Hardware purchases user-gated. 500Hz physics rate
untouched. This doc is the implementation contract for the prep increments.

## Key architectural finding (why this is cheap)

The FC (`Studio/src/MotorController.zig`) is ALREADY network-fed:
- UDP :5000 protocol (`Server` struct): `SetOrientation` (target quaternion,
  offboard setpoint), `UpdateOrientation` (current quaternion, state estimate
  input), `SetSpeed`/`ArmMotor`/`DisarmMotor`/`Battery`, heartbeat monitor.
- 1kHz PID thread: quaternion target/current -> Euler -> 3x PID ->
  `applyMotorMixing` -> `motor_outputs[4]` (throttle fractions, two mixer
  layouts) -> DShot bit-banged via Pi GPIO mmap (`Motor` struct).
- The FC never fuses raw IMU itself: orientation estimation is offboard by
  design (matches the NAV_STACK arch: VIO/ESKF offboard, FC = inner loop).
  IMU.zig (I2C MPU9250) is the onboard fallback/driver path, not the primary
  estimate source in the target architecture.

So HiL tests the PRODUCTION dataflow exactly:
  sim physics -> (offboard estimator or GT) --UDP UpdateOrientation--> FC PID
  @1kHz -> mixer -> [HiL seam: UDP, not GPIO] -> sim motor_v2 -> physics.

## Seams

1. Actuator (the only code seam needed in the FC): build option `-Dhil=true`
   swaps the GPIO `Motor` backend for a UDP sender that emits
   `motor_outputs[4]` (already-normalized throttle fractions, the exact PID
   output) to the sim's HiL port. PID, mixing, UDP :5000 server, heartbeat -
   all unchanged. x86_64-hil binary runs on box1; aarch64 hardware binary
   unchanged (compiles with hil=false, GPIO path).
2. Sensor/state: no FC change - the sim (or the offboard ESKF fed by sim
   sensors) sends `UpdateOrientation` quaternions to FC UDP :5000, at a
   configurable rate (hardware will see the ESKF's ~50-100Hz; GT mode sends
   sim-truth for controller isolation).

## Sim side (headless_main.zig)

New command surface:
- `{"cmd":"hil_listen", "port": N}` - bind UDP for motor packets
  `{"motors":[m0,m1,m2,m3], "seq":K}` (fractions 0..1, the FC's mixer output).
- `{"cmd":"hil_mode", "pace":"free"|"lockstep"}` -
  free: physics free-runs on wall clock (deployment-true; default).
  lockstep: physics advances exactly 2 ticks (2ms @500Hz) per motor packet
  (deterministic replay/debug; approximates the 1kHz FC loop).
- `{"cmd":"hil_state", "rate_hz":R, "dest":"ip:port", "source":"gt"|"eskf"}` -
  stream `UpdateOrientation` quaternions (+ gyro rates optional) to the FC.
- Motor packets map onto the EXISTING motor_v2 path (RPM command inversion,
  ESC latency, electrical, battery sag) - the same fidelity the sim inner
  loop validated against. Throttle fraction -> RPM command scaling matches
  the in-sim mixer's convention (document the mapping at implementation).

## Orchestration (autoresearch/hil_run.py)

Launch: headless sim (hil commands) + MotorController x86_64-hil. Scenario
runner sends `SetOrientation` step targets per script; logs sim GT state,
FC motor outputs, UpdateOrientation stream; scores:
- attitude step response: rise time, overshoot, settle, steady-state error
  (roll/pitch/yaw, +-10 deg steps)
- disturbance: 0.5 m/s lateral velocity kick recovery
- rate/health: FC loop overruns (its own counter), UpdateOrientation jitter,
  motor packet rate/gaps, heartbeat stability
- comparison baseline: same scenarios through the in-sim t4-lineage inner
  loop (existing eval harness) - the FC must be at least parity.

## Risks / honest notes

- Free-run pacing on a shared box1 (load 5+) will jitter; lockstep exists
  for deterministic bring-up before wall-clock realism matters.
- The FC's Euler conversion + PID dt uses wall-clock nanoTimestamp - under
  lockstep the FC still sees real time; PID dt stays wall-clock-true (the
  sim is what pauses). Documented; acceptable for bring-up.
- Mixer throttle-fraction -> sim RPM-command mapping must be validated
  against the in-sim mixer on a hover point before any step-response
  numbers are believed (explicit first milestone).
- UDP loss/reorder on localhost ~nil; heartbeat timeout (2s) already in FC.

## Increments

1. This doc + build-option skeleton (-Dhil compiles both backends; no
   behavior change when false).
2. FC UDP motor backend + sim hil_listen/hil_mode/hil_state + hover mapping
   validation (throttle fraction -> RPM -> measured thrust = hover).
3. hil_run.py attitude step-response suite + report vs t4 baseline.
4. ESKF-in-the-loop mode (source=eskf: sim sensors -> rung-2 ESKF -> FC)
   - the full rung-2 stack flying the FC.
