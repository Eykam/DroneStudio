# Rate Controller Assessment (firmware `RateController` / `PIDController`)

Scope: `Studio/src/core/ecs/components/FlightController.zig` (`RateController`),
`Studio/src/core/flight/PIDController.zig`, `Studio/src/core/ecs/prefabs/Drone.zig`.
Everything below is a **code read + sim reasoning**, not flight data. Nothing in
this document claims the controller is tuned or flight-ready.

## What is complete and sound

- Full 3-axis rate PID via `PosePIDController` (roll/pitch/yaw), gains
  roll/pitch `(kp,ki,kd) = (0.1, 0.005, 0.001)`, yaw `(0.05, 0.003, 0.0005)`.
- Anti-windup on the integrator (clamped) **and** output clamping.
- `setGains()` for runtime tuning, `reset()` for mode changes.
- Rate setpoint clamps exist (`max_roll/max_pitch = 10.47 rad/s`,
  `max_yaw = 5.24 rad/s`), though the clamp in `updateAttitudeControl` is
  currently commented out (`// ω_sp = ω_sp.clamp(...)`).

## Findings (ordered by importance)

### 1. Comment says "derivative on measurement"; code differentiates the error
`PIDController.step`:
```zig
// Derivative term (derivative on measurement to avoid derivative kick)
self.derivative = (err - self.last_error) / dt;
```
Derivative-on-measurement would differentiate `-measurement`, not `err`.
As written, every **setpoint** change produces a derivative impulse
("derivative kick") of `kd * Δsp / dt`. In our control split the policy
emits a new rate setpoint at 20 Hz and the PID runs at 500 Hz, so the
setpoint jumps every 25 PID ticks. With `kd = 0.001` and `dt = 0.002 s`,
a `Δsp = 1 rad/s` jump injects a `0.5 N·m`-scale torque impulse
(`0.001 * 1 / 0.002`) for one tick. Small with the current tiny `kd`, but
the code does not do what the comment says, and any future D-gain increase
amplifies the kick. Fix: differentiate measurement (sign-flipped) or filter
the D term; correct the comment either way.

### 2. Gains are unverified for this airframe (tuning status unknown)
With `Ixx = Iyy = 0.040 kg·m²` (Drone.zig), `kp = 0.1` gives a closed-loop
rate stiffness of `kp/I = 2.5 s⁻²`, i.e. `ωn ≈ 1.6 rad/s` with almost no
D damping (`kd/I = 0.025 s⁻¹`). That is **very soft** for a quadrotor rate
loop (typical target bandwidth 30-100 rad/s) and would look sluggish in
step response. This may be deliberate for the sim (or never tuned at all).
It is exactly what the sim step-response tests below will quantify.
Do not fly these gains assuming they are tuned.

### 3. `std.debug.print` spam in hot control paths
Unconditional prints inside per-tick paths:
- `updateAttitudeControl`: `Q_ERR / W_SP` dump every call (~line 363)
- `updateStateEstimate`: `DT / RAW_RATES / RAW_ACCEL / Q_GYRO` dump every
  call (~line 612)
- `updateMotorMixer`: `MIXER: rates -> torque` dump every call (~line 665)
At 500-1000 Hz on a dedicated flight thread this floods stdout, serializes
on a mutex, and can blow the loop budget. Gate behind a compile-time or
runtime debug flag. (The headless binary's inner loop uses
`RateController.step` directly and does not hit these paths, so they do not
contaminate sim episodes - but they will matter on hardware.)

### 4. Complementary-filter accelerometer gating is commented out
In `updateStateEstimate` the accel-correction block's gate (the standard
`|accel_mag - g| < threshold` check) is commented out - the structure shows
the `if` removed with a dangling `// }`. Result: the accelerometer
correction is applied **always**, including during linear acceleration,
which corrupts the gravity reference and makes the attitude estimate drift
under sustained maneuvering. This only affects state estimation, not the
rate loop itself, but it feeds the attitude P loop that generates rate
setpoints in stabilized mode.

### 5. `dt` is used as a divisor with no guard
`step()` computes `(err - last_error) / dt`. A zero (or pathological) `dt`
yields inf/NaN torque. Clamp or skip the D term when `dt <= eps`.

### 6. Minor: rate setpoint clamp commented out
`updateAttitudeControl` computes `w_sp` from attitude error but the clamp to
`max_rate` is commented out, so large attitude errors can request
unphysical rates (attitude P gains `Kp_roll = Kp_pitch = 0.05` limit this
in practice, but the safety bound should be explicit).

## Sim-based verification plan (now runnable)

The headless binary (`zig build headless`, JSON-lines protocol, 500 Hz inner
loop running the *actual* `RateController.step`) makes these cheap to run:

1. **Step response (per axis).** Hover, then step the rate setpoint
   0 -> 3 rad/s for 0.5 s. Measure rise time, overshoot, settling time,
   steady-state error from the binary's own state output. Expect finding 2
   to show up as a slow, underdamped response.
2. **D-kick quantification.** Step the setpoint and log the torque on the
   tick of the jump; compares the measured impulse to `kd * Δsp / dt`
   (finding 1).
3. **Chirp / sine sweep** on the rate setpoint (0.5-50 Hz) to estimate
   closed-loop bandwidth per axis.
4. **Windup/recovery.** Command a setpoint beyond the torque limit for 2 s,
   release, and measure recovery time (validates anti-windup).
5. **Candidate gain tuning in sim.** Grid search / relay autotune against
   rise-time and overshoot targets (e.g. rise < 50 ms, overshoot < 20%).
   Output is explicitly labeled **sim-tuned candidates, not flight-ready**:
   they must be re-validated on hardware with conservative first flights.

These tests belong in `autoresearch/rate_tuning.py`, driving the same
`SimBinaryEnv` transport as the parity check, with results committed as
`autoresearch/rate_tuning_report.json`.

## Bottom line

The controller structure is complete and the hygiene (anti-windup, clamps,
reset) is right. The two things that stand between this and trustable gains
are the derivative-on-error mismatch (finding 1) and the fact that the
gains have never been characterized against this inertia model (finding 2).
Both are now testable in the sim, and the tests above will turn
"tuning status unknown" into measured step-response numbers.
