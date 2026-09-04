# Motor v2: electromechanical motor/ESC/prop/battery model

Runtime-toggleable higher-fidelity motor path in `Studio/src/headless_main.zig`
(default OFF; v1 first-order lag path unchanged).

## Enable

- Headless protocol: `{"cmd":"motor_v2","on":true}` (after `set_dynamics`).
- Autoresearch: `AUTORESEARCH_MOTOR_V2=1` (env_sim sends the toggle per episode spawn).

## Model (per 500 Hz tick; the 500 Hz physics rate is untouched)

1. Commanded per-motor thrust (his v1 mixer, unchanged) inverted through
   `thrust = kf * omega^2` to an RPM command, with `kf = max_thrust / omega_max^2`.
2. ESC latency: configurable tick-delay queue on the RPM command (DShot-ish).
3. DC motor electrical: `I = clamp((duty * V_batt - ke * omega) / R_w, 0, I_max)`,
   `duty = omega_cmd / omega_max`.
4. Rotor dynamics: `d(omega)/dt = (ke * I - kd * omega^2) / J_r`,
   `kd = kf * (kd/kf ratio)`.
5. Thrust and yaw drag from ACTUAL `omega^2` (so spin-up lag is physical, and
   drag follows the true rotor state).
6. Battery: 4S LiPo, `V_oc = 12.0 + 4.8 * SoC`, bus `V = max(9, V_oc - I_total * R_int)`,
   SoC drains with integrated current (two-half-step solve so sag sees the
   current tick's current).
7. Rotor gyroscopic reaction on the body: `-omega_body x (0, L_r, 0)` with
   `L_r = sum(J_r * omega_i * yaw_sign_i)`.

## Constants (ALL ESTIMATES pending measured data)

omega_max 3560 rad/s (~2300KV * 14.8V), ke 0.00416 N*m/A, R_w 0.06 ohm,
J_r 9e-6 kg*m^2, I_max 40 A, R_int 0.024 ohm (4S pack), capacity ~1.3 Ah,
ESC delay 1 tick, kd/kf 0.015 m (typical 5in prop).

## Measured data needed to replace estimates

Motor model + KV (2207?), winding resistance, prop inertia, ESC protocol,
battery cell count/capacity/internal resistance. Best: a bench thrust-stand
log (throttle step -> thrust/RPM/V/I) to fit kf, ke, R_w, J_r directly.

## Validation (2026-09-04, seeds 77000+, n=12)

Same policies, v1 vs v2: student 5m d0.1 0.917/0.917, 10m d0.1 1.000/1.000;
teacher 5m/10m 1.000/1.000 both modes. v2 is calibrated so existing policies
transfer at short range; differences grow with aggressive throttle usage
(sag, current limiting) and fast attitude moves (gyroscopic term).
