# Motor v2: electromechanical motor/ESC/prop/battery model

Runtime-toggleable higher-fidelity motor path in `Studio/src/headless_main.zig`
(default OFF; v1 first-order lag path unchanged).

## Enable

- Headless protocol: `{"cmd":"motor_v2","on":true}` (after `set_dynamics`).
- Autoresearch: `AUTORESEARCH_MOTOR_V2=1` (env_sim sends the toggle per episode spawn).

## Calibrated to the actual motor: AKK RS2205 2300KV (user-supplied 2026-09-04)

AKK RS2205 2300KV (Amazon https://a.co/d/066TbARJ), a clone of the EMAX
RS2205-2300KV design (same stator class/KV), so EMAX's published spec and
third-party thrust-stand data apply:

| Parameter | Value | Source |
| --- | --- | --- |
| Stator | 22 x 5 mm (2205), 12N14P | emaxmodel.com RS2205 page |
| KV | 2300 rpm/V -> ke = 60/(2*pi*2300) = 0.004152 V*s/rad | listing + EMAX spec |
| Internal resistance | 65 mOhm | EMAX RS2205 spec table (rees52 listing) |
| No-load current | 0.6 A @ 10 V | EMAX spec table |
| Peak current (measured) | 28.6 A @ 100% throttle, HQ5045BN, 4S | oscarliang.com thrust stand |
| Peak thrust (measured) | 989 g (HQ5045BN), 1024-1155 g other 5045-class props, 16.2-16.6 V | oscarliang.com, emaxmodel.com |
| Voltage | 3-4S (12.6-16.8 V); AKK listing rates 16.8 V | listing + EMAX spec |
| Weight | 28.8 g (AKK listing) / ~30 g (EMAX with wires) | listing + EMAX spec |
| Recommended prop | 5 in (5045 class); ESC 30-60 A | EMAX spec table |

kf = 7.9e-7 N/(rad/s)^2 fits the thrust stand: model electrical equilibrium
gives ~3500 rad/s at full charge/full throttle -> kf*w^2 = 9.7 N vs measured
989 g = 9.7 N (HQ5045BN). omega_max = 3560 rad/s (2300KV x ~14.8 V loaded)
is the kf-consistent command cap.

## Model (per 500 Hz tick; the 500 Hz physics rate is untouched)

1. Commanded per-motor thrust (his v1 mixer, unchanged) inverted through
   `thrust = kf * omega^2` to an RPM command.
2. ESC latency: configurable tick-delay queue on the RPM command (DShot-ish).
3. RPM-governor duty feedforward (Betaflight-style RPM control):
   `duty = clamp((ke*w_cmd + R_w*kd*w_cmd^2/ke) / V_batt, 0, 1)` - the bus
   voltage needed to hold w_cmd against prop drag. Saturation + current
   limit produce the true equilibrium under load/sag.
4. DC motor electrical: `I = clamp((duty * V_batt - ke * omega) / R_w, 0, I_max)`.
5. Rotor dynamics: `d(omega)/dt = (ke * I - kd * omega^2) / J_r`,
   `kd = kf * 0.015` (physical 5 in prop torque/thrust ratio).
6. Thrust and yaw drag from ACTUAL `omega^2` (spin-up lag is physical).
7. Battery: 4S LiPo, `V_oc = 12.0 + 4.8 * SoC`, bus `V = max(9, V_oc - I_total * R_int)`,
   SoC drains with integrated current (two-pass solve so sag sees the
   current tick's current).
8. Rotor gyroscopic reaction on the body: `-omega_body x (0, L_r, 0)` with
   `L_r = sum(J_r * omega_i * yaw_sign_i)`.

## Still estimates (need from user)

- J_r = 9e-6 kg*m^2 rotor+prop inertia (no published data; needs a spin-up
  or pendulum measurement)
- Battery: capacity ~1.3 Ah and R_int = 24 mOhm assumed (need actual pack
  mAh/C-rating)
- ESC latency 1 tick @ 500 Hz (need ESC protocol: DShot300/600/1200?)
- Prop identity beyond "5 inch": kd/kf = 0.015 is the generic 5 in value;
  the exact prop (e.g. HQ5045BN vs Gemfan 5045x3) shifts it
- Best possible: one bench thrust-stand log (throttle steps -> thrust/RPM/V/I)
  to fit kf, ke, R_w, J_r directly.

## Known spec discrepancy (flagged to user)

Manifest `drag_ratio = 0.15` (yaw torque per N of thrust) is non-physical for
a 2205 + 5 in prop - it implies 1.5 N*m of prop drag at 10 N thrust vs the
motor's ~0.12 N*m torque ceiling; the real 5 in value is ~0.015 m. v1 and the
v2 mixer allocation still use the manifest value; v2's physical yaw drag uses
0.015. Recommend correcting the chassis spec if 0.15 was a placeholder.

Manifest motor mass 33.4 g vs measured 28.8-30 g (4 motors -> ~18 g total
mass + inertia impact). Not changed here - chassis manifest belongs to the
CAD side; flagged for coordination.

## Validation (2026-09-04, seeds 77000+, n=12, d0.1)

v1 vs v2-calibrated: teacher 5m/10m 1.000/1.000 both modes; student 5m
0.917/0.833 and 10m 1.000/1.000 (within n=12 noise). v2 is calibrated so
existing policies transfer at short range; differences grow with aggressive
throttle usage (sag, current limiting) and fast attitude moves (gyro term).
