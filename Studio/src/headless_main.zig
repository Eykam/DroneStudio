//! Headless physics-only episode runner for the auto-researcher inner loop.
//!
//! Design (HEADLESS_API.md v1, mandated implementation 2026-09-03):
//! - No window, no GL, no renderer, no ECS. One Bullet world, one rigid
//!   body (the drone), analytic obstacle/floor/goal checks identical to
//!   QuadNavEnv so results stay comparable across backends.
//! - Fast loop REUSES FlightController.RateController + PIDController
//!   verbatim (his firmware code path), dt = 1/500 s, 25 fast steps per
//!   policy step (20 Hz), matching QuadNavEnv's constants.
//! - Concrete scenes are sampled Python-side and passed in on reset
//!   (spawn/goal/obstacle list), so the Zig engine and the numpy engine
//!   run the exact same scenario - parity is in dynamics, not RNG.
//! - Actuation model v1: PID body torques + collective thrust applied
//!   directly (QuadNavEnv parity). The app's motor mixer + motor lag path
//!   (FlightControllerComponent.updateMotorMixer/updateMotorLag) is the
//!   v1.1 fidelity upgrade, gated on verifying its free parameters.
//!
//! Protocol: stdio JSON-lines, one message per line.
//!   <- {"cmd":"reset","seed":7,"scene":{"spawn":[x,y,z],"goal":[x,y,z],
//!        "obstacles":[[x,y,z,r],...],"extent":40,"max_steps":250,
//!        "dynamics_noise":0.05}}
//!   -> {"obs":[15 floats],"reward":0,"done":false}
//!   <- {"cmd":"step","action":[roll,pitch,yaw,thrust]}   // each in [-1,1]
//!   -> {"obs":[...],"reward":R,"done":B,"info":{"collided":B,"succeeded":B,"steps":N}}
//!   <- {"cmd":"ping"} -> {"ok":true}
//!   <- {"cmd":"close"} -> exits

const std = @import("std");
const Math = @import("core/Math.zig");
const FC = @import("core/ecs/components/FlightController.zig");
const CM = @import("core/ChassisManifest.zig");

const bullet = @cImport({
    @cInclude("cbullet.h");
});

const Vec3 = Math.Vec3;
const Quaternion = Math.Quaternion;

// --- constants: identical to autoresearch/env_quad.py ---------------------
const FAST_DT: f32 = 1.0 / 500.0;
const FAST_PER_POLICY: u32 = 25;
const MAX_RATES = [3]f32{ 10.47, 10.47, 5.24 };
const MAX_THRUST: f32 = 40.0;
const FILTER_TAU: f32 = 0.05;
const MASS: f32 = 1.5;
const IXX: f32 = 0.040;
const IYY: f32 = 0.040;
const IZZ: f32 = 0.047;
const DRONE_RADIUS: f32 = 0.3;
const GOAL_RADIUS: f32 = 2.0;

/// Dynamics profile: `abstract` (default, QuadNavEnv-parity values) or
/// manifest-driven (dronestudio.chassis/1.1 via set_dynamics).
const Dynamics = struct {
    mass: f32 = MASS,
    inertia: [3]f32 = .{ IXX, IYY, IZZ },
    max_thrust: f32 = MAX_THRUST, // total, N
    arm_length: f32 = 0.15,
    motor_lag_s: f32 = 0.04,
    motor_drag_ratio: f32 = 0.15,
    motor_max_thrust: f32 = 10.0,
    aero_area: [3]f32 = .{ 0, 0, 0 }, // body axes, m^2; all-zero = no drag
    aero_cd: f32 = 1.1,
    // Real motor model (manifest mode): 4 motors at manifest positions,
    // mixed from collective thrust + PID torque, first-order lag, ktau yaw.
    motor_mode: bool = false,
    com: [3]f32 = .{ 0, 0, 0 }, // center of mass, sim body frame (+Y up)
    motor_pos: [4][3]f32 = undefined, // sim body frame (+Y up), RELATIVE TO COM
    motor_yaw_sign: [4]f32 = undefined, // -1 cw, +1 ccw (his prop_drag_signs)
    mix_inv: [4][4]f32 = undefined, // maps [T, taux, tauy, tauz] -> motor thrusts
    motor_lag_state: [4]f32 = .{ 0, 0, 0, 0 },
    // Motor model v2: electromechanical per-motor prop-speed state
    // (BLDC + ESC + battery). Constants below are ANCHORS from the manifest
    // plus spec-sheet estimates pending bench sysid - see PROGRESS.md.
    motor_v2: bool = false,
    m2_omega: [4]f32 = .{ 0, 0, 0, 0 }, // prop speed, rad/s
    m2_cmd_hist: [4][4]f32 = undefined, // ESC latency queue (rad/s commands)
    m2_hist_idx: usize = 0,
    m2_omega_max: f32 = 3560.0, // rad/s, ~2300KV * 14.8V (ESTIMATE)
    m2_kf: f32 = 0.0, // thrust = kf*omega^2; set from max_thrust/omega_max^2
    m2_kd_over_kf: f32 = 0.015, // drag/thrust ratio, m (typical 5in prop)
    m2_ke: f32 = 0.00416, // V*s/rad = N*m/A, from omega_max at nom voltage
    m2_rw: f32 = 0.06, // winding resistance, ohm (ESTIMATE)
    m2_jr: f32 = 9.0e-6, // rotor+prop inertia, kg*m^2 (ESTIMATE)
    m2_imax: f32 = 40.0, // ESC per-motor current limit, A (ESTIMATE)
    m2_batt_r_int: f32 = 0.024, // 4S pack internal resistance, ohm (ESTIMATE)
    m2_batt_cap_as: f32 = 4680.0, // charge, A*s (~1.3Ah, ASSUMPTION)
    m2_soc: f32 = 1.0,
    m2_esc_delay_ticks: usize = 1, // DShot latency in 500Hz ticks (ESTIMATE)
    // Prop inflow + ground-effect fidelity (2026-09-05):
    m2_prop_d: f32 = 0.127, // prop diameter, m (manifest prop_diameter_m)
    m2_j0: f32 = 1.2, // advance ratio where thrust -> 0 (ESTIMATE, 4.5in pitch)
    m2_ge: bool = true, // ground-effect thrust amplification on/off
};
const GROUND_Y: f32 = 0.05;

const Obstacle = struct { center: Vec3, radius: f32 };

const Scenario = enum { goto, hover_hold, land };

const Scene = struct {
    spawn: Vec3,
    spawn_vel: Vec3 = Vec3.zero(), // IC v1: initial linear velocity (default rest)
    goal: Vec3,
    obstacles: []Obstacle,
    extent: f32 = 40.0,
    max_steps: u32 = 250,
    dynamics_noise: f32 = 0.05,
    // scenario sampler (all default to today's goto behavior):
    success_radius: f32 = 2.0, // was the GOAL_RADIUS const
    scenario: Scenario = .goto,
    hold_s: f32 = 4.0, // hover_hold: seconds inside radius to succeed
    max_touchdown_vs: f32 = 0.5, // land: max |vertical speed| at touchdown, m/s
    shaping_v2: bool = false, // small smooth-flight penalty, all scenarios
    face_reward_w: f32 = 0.0, // per-step reward for nose alignment with the current target (sustained facing; user 2026-09-04)
    // T3 (harder-scenes Phase 2): ordered waypoints to pass through (goto
    // only) before the final goal. Empty = today's behavior.
    waypoints: []Vec3 = &.{},
};

fn vecFromJson(v: std.json.Value) !Vec3 {
    const arr = switch (v) {
        .array => |a| a,
        else => return error.BadVec,
    };
    if (arr.items.len != 3) return error.BadVec;
    var out: [3]f32 = undefined;
    for (arr.items, 0..) |item, i| {
        out[i] = switch (item) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => return error.BadVec,
        };
    }
    return Vec3.init(out[0], out[1], out[2]);
}

fn f32FromJson(v: std.json.Value, default: f32) f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |n| @floatFromInt(n),
        else => default,
    };
}

/// 4x4 inverse via Gauss-Jordan with partial pivoting (mixing matrix is
/// well-conditioned for any physical quad layout).
fn invert4(A: [4][4]f64) [4][4]f32 {
    var aug: [4][8]f64 = undefined;
    for (0..4) |i| {
        for (0..4) |j| aug[i][j] = A[i][j];
        for (0..4) |j| aug[i][4 + j] = if (i == j) 1.0 else 0.0;
    }
    for (0..4) |col| {
        var piv = col;
        for (col + 1..4) |r| {
            if (@abs(aug[r][col]) > @abs(aug[piv][col])) piv = r;
        }
        if (piv != col) std.mem.swap([8]f64, &aug[col], &aug[piv]);
        const d = aug[col][col];
        for (0..8) |j| aug[col][j] /= d;
        for (0..4) |r| {
            if (r == col) continue;
            const fctr = aug[r][col];
            for (0..8) |j| aug[r][j] -= fctr * aug[col][j];
        }
    }
    var out: [4][4]f32 = undefined;
    for (0..4) |i| for (0..4) |j| {
        out[i][j] = @floatCast(aug[i][4 + j]);
    };
    return out;
}

const World = struct {
    world: bullet.CbtWorldHandle,
    body: bullet.CbtBodyHandle,
    shape: bullet.CbtShapeHandle,
    rate_ctrl: FC.RateController,
    dbg_motors: bool = false,
    dbg_tick: u32 = 0,
    dyn: Dynamics = .{},
    dyn_name_buf: [80]u8 = undefined,
    dyn_name_len: usize = 0,
    filtered_rates: [3]f32 = .{ 0, 0, 0 },
    filtered_thrust: f32 = 0.0,
    rng: std.Random.Xoshiro256,
    scene: Scene = undefined,
    steps: u32 = 0,
    prev_dist: f32 = 0,
    collided: bool = false,
    succeeded: bool = false,
    done: bool = false,
    obs_v2: bool = false, // 19-dim yaw-relative obs (default off: 15-dim v1)
    obs_v3: bool = false, // 26-dim: v2 layout + T3 waypoint channels (supersedes v2 when on)
    obs_v4: bool = false, // 27-dim: v3 layout + motor_v2 battery SoC (supersedes v3 when on)
    waypoint_idx: usize = 0, // T3: next waypoint to pass
    hold_steps: u32 = 0, // hover_hold consecutive in-radius policy steps
    last_torque: [3]f32 = .{ 0, 0, 0 }, // PID torque output, body frame (telemetry)

    fn init(alloc: std.mem.Allocator) !*World {
        _ = alloc;
        const w = try std.heap.page_allocator.create(World);
        w.world = bullet.cbtWorldCreate();
        var gravity = [3]f32{ 0, -9.81, 0 };
        bullet.cbtWorldSetGravity(w.world, &gravity);

        // Drone body: sphere proxy (radius 0.3 - exact parity with
        // QuadNavEnv's point + 0.3 m collision margin), mass/inertia from
        // prefabs/Drone.zig (computed values: 1.5 kg, 0.040/0.040/0.047).
        w.shape = bullet.cbtShapeAllocate(bullet.CBT_SHAPE_TYPE_SPHERE);
        bullet.cbtShapeSphereCreate(w.shape, DRONE_RADIUS);
        w.body = bullet.cbtBodyAllocate();
        const identity = [4][3]f32{
            .{ 1, 0, 0 },
            .{ 0, 1, 0 },
            .{ 0, 0, 1 },
            .{ 0, 2, 0 },
        };
        bullet.cbtBodyCreate(w.body, MASS, @ptrCast(&identity), w.shape);
        const inertia = [3]f32{ IXX, IYY, IZZ };
        bullet.cbtBodySetMassProps(w.body, MASS, &inertia);
        bullet.cbtBodySetDamping(w.body, 0.0, 0.0); // QuadNavEnv has no aero damping
        bullet.cbtBodySetActivationState(w.body, bullet.CBT_DISABLE_DEACTIVATION);
        bullet.cbtWorldAddBody(w.world, w.body);

        w.rate_ctrl = FC.RateController.init(1.0, 1.0); // integrator/output limits
        w.dbg_motors = std.posix.getenv("HEADLESS_DBG_MOTORS") != null;
        w.rng = std.Random.Xoshiro256.init(0);
        return w;
    }

    fn gaussian(self: *World) f32 {
        // Box-Muller on Xoshiro256 - statistically equivalent to numpy's
        // normal(0,1); per-draw values differ by engine, that's fine for a
        // stochastic disturbance term.
        const r = self.rng.random();
        var ua = r.float(f32);
        if (ua < 1e-7) ua = 1e-7;
        const ub = r.float(f32);
        return @sqrt(-2.0 * @log(ua)) * @cos(2.0 * std.math.pi * ub);
    }

    fn bodyQuat(self: *World) Quaternion {
        var t: [4][3]f32 = undefined;
        bullet.cbtBodyGetCenterOfMassTransform(self.body, @ptrCast(&t));
        // Bullet basis is row-major 3x3 (+ position row). Mat3.from_array
        // takes column-major - transpose.
        const cm = [9]f32{
            t[0][0], t[1][0], t[2][0],
            t[0][1], t[1][1], t[2][1],
            t[0][2], t[1][2], t[2][2],
        };
        return Quaternion.from_mat3(Math.Mat3.from_array(cm));
    }

    fn bodyPos(self: *World) Vec3 {
        var p: [3]f32 = undefined;
        bullet.cbtBodyGetCenterOfMassPosition(self.body, &p);
        return Vec3.from_array(p);
    }

    fn bodyVel(self: *World) Vec3 {
        var v: [3]f32 = undefined;
        bullet.cbtBodyGetLinearVelocity(self.body, &v);
        return Vec3.from_array(v);
    }

    fn bodyOmega(self: *World) Vec3 {
        var o: [3]f32 = undefined;
        bullet.cbtBodyGetAngularVelocity(self.body, &o);
        // Bullet stores angular velocity in the WORLD frame; every consumer
        // here (rate PID measurement, obs rates, rotor gyro term) wants BODY
        // frame - rotate it back. Feeding world-frame omega was the root
        // cause of the long-standing "sustained yaw tumbles the quad" issue
        // (user-reported, root-caused 2026-09-04): at heading psi the PID's
        // roll/pitch feedback was rotated by -psi from the true body rates,
        // an energy pump proportional to yaw rate (any tilt seeded it, f32
        // noise suffices; growth ~10/s at 0.42 rad/s yaw; zero-yaw flight
        // near-unaffected since the frames agree for a level quad).
        return Vec3.from_array(o).rotate_by_quaternion(self.bodyQuat().conjugate());
    }

    fn reset(self: *World, scene: Scene, seed: u64) void {
        self.scene = scene;
        self.rng = std.Random.Xoshiro256.init(seed);
        self.rate_ctrl.reset();
        self.filtered_rates = .{ 0, 0, 0 };
        self.filtered_thrust = 0;
        self.dyn.motor_lag_state = .{ 0, 0, 0, 0 };
        self.dyn.m2_omega = .{ 0, 0, 0, 0 };
        self.dyn.m2_soc = 1.0;
        self.steps = 0;
        self.hold_steps = 0;
        self.waypoint_idx = 0;
        self.collided = false;
        self.succeeded = false;
        self.done = false;

        const t = [4][3]f32{
            .{ 1, 0, 0 },
            .{ 0, 1, 0 },
            .{ 0, 0, 1 },
            .{ scene.spawn.x(), scene.spawn.y(), scene.spawn.z() },
        };
        bullet.cbtBodySetCenterOfMassTransform(self.body, @ptrCast(&t));
        const zero = [3]f32{ 0, 0, 0 };
        const sv0 = [3]f32{ scene.spawn_vel.x(), scene.spawn_vel.y(), scene.spawn_vel.z() };
        bullet.cbtBodySetLinearVelocity(self.body, &sv0);
        bullet.cbtBodySetAngularVelocity(self.body, &zero);
        bullet.cbtBodySetActivationState(self.body, bullet.CBT_ACTIVE_TAG);
        self.prev_dist = (if (scene.waypoints.len > 0) scene.waypoints[0] else scene.goal).sub(scene.spawn).length();
    }

    /// One fast (500 Hz) step: input filter -> his RateController PID ->
    /// torque + thrust -> Bullet integration. Mirrors env_quad._fast_step.
    fn fastStep(self: *World, desired_rates: [3]f32, thrust_cmd: f32) void {
        const alpha = FAST_DT / (FILTER_TAU + FAST_DT);
        // disturbance on the commanded rates, like env_quad (noise * 5.0)
        var noisy = desired_rates;
        if (self.scene.dynamics_noise > 0) {
            inline for (0..3) |i| {
                noisy[i] += self.gaussian() * self.scene.dynamics_noise * 5.0;
            }
        }
        inline for (0..3) |i| {
            self.filtered_rates[i] += alpha * (noisy[i] - self.filtered_rates[i]);
        }
        self.filtered_thrust += alpha * (thrust_cmd - self.filtered_thrust);

        const omega = self.bodyOmega();
        const torque_body = self.rate_ctrl.update(self.filtered_rates, omega, FAST_DT);
        self.last_torque = torque_body;

        const q = self.bodyQuat();
        var f: [3]f32 = undefined;
        var tq: [3]f32 = undefined;
        if (self.dyn.motor_mode) {
            // Real motor model: mix collective thrust + PID torque to 4 motor
            // thrusts via the manifest geometry (precomputed inverse), clamp
            // per his applySaturation, lag per his updateMotorLag, then
            // reconstruct force/torque from the lagged motor states.
            const d = &self.dyn;
            const cmd = [4]f32{ self.filtered_thrust, torque_body[0], torque_body[1], torque_body[2] };
            var t: [4]f32 = undefined;
            inline for (0..4) |i| {
                t[i] = d.mix_inv[i][0] * cmd[0] + d.mix_inv[i][1] * cmd[1] + d.mix_inv[i][2] * cmd[2] + d.mix_inv[i][3] * cmd[3];
            }
            const avg = (t[0] + t[1] + t[2] + t[3]) / 4.0;
            const min_t = @max(0.1, avg * 0.1); // his dynamic minimum
            var total: f32 = 0;
            var torque_b = [3]f32{ 0, 0, 0 };
            if (d.motor_v2) {
                // Electromechanical path: cmd thrust -> omega_cmd (invert
                // kf*omega^2) -> ESC latency -> DC motor + prop inertia ->
                // thrust/drag from ACTUAL omega^2. Battery: 4S LiPo with
                // SoC-dependent open-circuit voltage and I*R sag.
                // Prop fidelity (2026-09-05): static kf*omega^2 corrected for
                //   (a) axial inflow: T *= clamp(1 - J/J0), J = 2*pi*v_ax/(omega*D),
                //       v_ax = body speed along the rotor axis (positive = climb);
                //   (b) ground effect: T *= 1/(1 - (R/4z)^2), z clamped >= 0.6R.
                const kf = d.m2_kf;
                const kd = kf * d.m2_kd_over_kf;
                const v_oc = 12.0 + 4.8 * d.m2_soc; // 4S: 12.0 empty .. 16.8 full
                const axis_w = Vec3.init(0, 1, 0).rotate_by_quaternion(q);
                const v_w = self.bodyVel();
                const v_ax = v_w.x() * axis_w.x() + v_w.y() * axis_w.y() + v_w.z() * axis_w.z();
                const prop_r = 0.5 * d.m2_prop_d;
                const z_ge = @max(self.bodyPos().y() - GROUND_Y, 0.6 * prop_r);
                const ge_factor: f32 = if (d.m2_ge) 1.0 / (1.0 - (prop_r / (4.0 * z_ge)) * (prop_r / (4.0 * z_ge))) else 1.0;
                var i_total: f32 = 0;
                var ocmds: [4]f32 = undefined;
                inline for (0..4) |i| {
                    const tc = std.math.clamp(t[i], min_t, d.motor_max_thrust);
                    var oc = @sqrt(tc / kf);
                    // ESC latency: push through the delay queue
                    d.m2_cmd_hist[i][d.m2_hist_idx % 4] = oc;
                    oc = d.m2_cmd_hist[i][(d.m2_hist_idx + 4 - d.m2_esc_delay_ticks) % 4];
                    ocmds[i] = @min(oc, d.m2_omega_max);
                }
                d.m2_hist_idx += 1;
                // pass 1: currents at unsagged bus, to size this tick's sag
                var currents: [4]f32 = undefined;
                inline for (0..4) |i| {
                    const duty = @min(1.0, (d.m2_ke * ocmds[i] + d.m2_rw * kd * ocmds[i] * ocmds[i] / d.m2_ke) / v_oc);
                    currents[i] = std.math.clamp((duty * v_oc - d.m2_ke * d.m2_omega[i]) / d.m2_rw, 0.0, d.m2_imax);
                    i_total += currents[i];
                }
                const v_batt = @max(9.0, v_oc - i_total * d.m2_batt_r_int);
                inline for (0..4) |i| {
                    // RPM-governor feedforward duty at the sagged bus;
                    // saturation + current limit give the true equilibrium
                    const duty = @min(1.0, (d.m2_ke * ocmds[i] + d.m2_rw * kd * ocmds[i] * ocmds[i] / d.m2_ke) / v_batt);
                    const cur = std.math.clamp((duty * v_batt - d.m2_ke * d.m2_omega[i]) / d.m2_rw, 0.0, d.m2_imax);
                    const tau_m = d.m2_ke * cur;
                    const dw = (tau_m - kd * d.m2_omega[i] * d.m2_omega[i]) / d.m2_jr;
                    d.m2_omega[i] = @max(0.0, d.m2_omega[i] + dw * FAST_DT);
                    const w2 = d.m2_omega[i] * d.m2_omega[i];
                    const adv = if (d.m2_omega[i] > 1.0)
                        std.math.clamp(1.0 - (2.0 * std.math.pi * v_ax / (d.m2_omega[i] * d.m2_prop_d)) / d.m2_j0, 0.0, 1.5)
                    else
                        1.0;
                    const ti = kf * w2 * adv * ge_factor;
                    total += ti;
                    const mp = d.motor_pos[i];
                    torque_b[0] += ti * -mp[2];
                    torque_b[2] += ti * mp[0];
                    torque_b[1] += kd * w2 * adv * ge_factor * d.motor_yaw_sign[i];
                }
                d.m2_soc = @max(0.0, d.m2_soc - i_total * FAST_DT / d.m2_batt_cap_as);
                // Rotor gyroscopic reaction: -Omega_body x (0, L_r, 0)
                var l_r: f32 = 0;
                inline for (0..4) |i| l_r += d.m2_jr * d.m2_omega[i] * d.motor_yaw_sign[i];
                const om = self.bodyOmega();
                // cross(om, [0, l_r, 0]) = (-om.z*l_r, 0, om.x*l_r); reaction on body = -that
                torque_b[0] += om.z() * l_r;
                torque_b[2] += -om.x() * l_r;
            } else {
                inline for (0..4) |i| {
                    const tc = std.math.clamp(t[i], min_t, d.motor_max_thrust);
                    const lag_alpha = FAST_DT / (d.motor_lag_s + FAST_DT);
                    d.motor_lag_state[i] += lag_alpha * (tc - d.motor_lag_state[i]);
                    const ti = d.motor_lag_state[i];
                    total += ti;
                    const mp = d.motor_pos[i];
                    // force = ti * +Y at mp; torque_arm = mp x (ti*Y) = ti*(-mp.z, 0, mp.x)
                    torque_b[0] += ti * -mp[2];
                    torque_b[2] += ti * mp[0];
                    torque_b[1] += ti * d.motor_drag_ratio * d.motor_yaw_sign[i]; // ktau yaw
                }
                // Yaw-instability forensics (2026-09-04): per-policy-step motor
                // channel dump to stderr, gated by HEADLESS_DBG_MOTORS.
                if (self.dbg_motors) {
                    self.dbg_tick +%= 1;
                    if (self.dbg_tick % 25 == 0) {
                        std.debug.print("MOT t=({d:.4},{d:.4},{d:.4},{d:.4}) lag=({d:.4},{d:.4},{d:.4},{d:.4}) tq=({d:.5},{d:.5},{d:.5}) fthr={d:.4} om=({d:.4},{d:.4},{d:.4})\n", .{ t[0], t[1], t[2], t[3], d.motor_lag_state[0], d.motor_lag_state[1], d.motor_lag_state[2], d.motor_lag_state[3], torque_b[0], torque_b[1], torque_b[2], self.filtered_thrust, self.bodyOmega().x(), self.bodyOmega().y(), self.bodyOmega().z() });
                    }
                }
            }
            const thrust_w = Vec3.init(0, total, 0).rotate_by_quaternion(q);
            const torque_w = Vec3.from_array(torque_b).rotate_by_quaternion(q);
            f = .{ thrust_w.x(), thrust_w.y(), thrust_w.z() };
            tq = .{ torque_w.x(), torque_w.y(), torque_w.z() };
        } else {
            const thrust_world = Vec3.init(0, self.filtered_thrust, 0).rotate_by_quaternion(q);
            const torque_world = Vec3.from_array(torque_body).rotate_by_quaternion(q);
            f = .{ thrust_world.x(), thrust_world.y(), thrust_world.z() };
            tq = .{ torque_world.x(), torque_world.y(), torque_world.z() };
        }
        // Manifest aero: flat-plate drag per body axis, assembled in the
        // world frame from body-axis directions (no quaternion inverse).
        const aa = self.dyn.aero_area;
        if (aa[0] > 0 or aa[1] > 0 or aa[2] > 0) {
            const v = self.bodyVel();
            const axes = [3]Vec3{
                Vec3.init(1, 0, 0).rotate_by_quaternion(q),
                Vec3.init(0, 1, 0).rotate_by_quaternion(q),
                Vec3.init(0, 0, 1).rotate_by_quaternion(q),
            };
            const rho: f32 = 1.225;
            var drag = Vec3.init(0, 0, 0);
            inline for (0..3) |i| {
                const vi = Vec3.dot(v, axes[i]);
                const k = 0.5 * rho * self.dyn.aero_cd * aa[i];
                drag = drag.sub(axes[i].scale(k * vi * @abs(vi)));
            }
            f[0] += drag.x();
            f[1] += drag.y();
            f[2] += drag.z();
        }
        bullet.cbtBodyApplyCentralForce(self.body, &f);
        bullet.cbtBodyApplyTorque(self.body, &tq);
        _ = bullet.cbtWorldStepSimulation(self.world, FAST_DT, 1, FAST_DT);
    }

    /// One policy (20 Hz) step: 25 fast steps + reward/termination,
    /// identical formulas to env_quad.step.
    fn policyStep(self: *World, action: [4]f32) struct { reward: f32, done: bool } {
        const desired = [3]f32{
            std.math.clamp(action[0], -1.0, 1.0) * MAX_RATES[0],
            std.math.clamp(action[1], -1.0, 1.0) * MAX_RATES[1],
            std.math.clamp(action[2], -1.0, 1.0) * MAX_RATES[2],
        };
        const thrust_cmd = (std.math.clamp(action[3], -1.0, 1.0) + 1.0) / 2.0 * self.dyn.max_thrust;
        for (0..FAST_PER_POLICY) |_| {
            if (self.done) break;
            self.fastStep(desired, thrust_cmd);
        }
        self.steps += 1;

        const pos = self.bodyPos();
        const scenario = self.scene.scenario;
        // T3 waypoints (goto only): progress is measured against the current
        // waypoint until all are passed, then against the final goal. Passing
        // a waypoint pays the same +10 as goal entry and rebases prev_dist so
        // the target switch never reads as a progress penalty.
        const wp_pending = scenario == .goto and self.waypoint_idx < self.scene.waypoints.len;
        const target = if (wp_pending) self.scene.waypoints[self.waypoint_idx] else self.scene.goal;
        const dist = target.sub(pos).length();
        var reward = (self.prev_dist - dist) - 0.01;
        self.prev_dist = dist;
        if (wp_pending and dist < self.scene.success_radius) {
            reward += 10.0;
            self.waypoint_idx += 1;
            const nt = if (self.waypoint_idx < self.scene.waypoints.len) self.scene.waypoints[self.waypoint_idx] else self.scene.goal;
            self.prev_dist = nt.sub(pos).length();
        }
        if (self.scene.shaping_v2) {
            reward -= 0.005 * self.bodyOmega().length() / 10.0;
        }
        // sustained facing (user 2026-09-04): per-step nose-to-target
        // alignment vs the CURRENT target (waypoint while pending, else
        // goal). Bounded by w * episode steps; +10/-5 outcome terms stay
        // dominant at w <= 0.03.
        if (self.scene.face_reward_w != 0.0 and !self.done) {
            const to_t = target.sub(pos);
            const dxz = @sqrt(to_t.x() * to_t.x() + to_t.z() * to_t.z());
            if (dxz > 0.05) {
                const fwd = Vec3.init(1, 0, 0).rotate_by_quaternion(self.bodyQuat());
                const fxz = @sqrt(fwd.x() * fwd.x() + fwd.z() * fwd.z());
                if (fxz > 0.05) {
                    const cos_e = (fwd.x() * to_t.x() + fwd.z() * to_t.z()) / (fxz * dxz);
                    reward += self.scene.face_reward_w * @max(0.0, cos_e);
                }
            }
        }
        // hover_hold: inside the radius the progress term is replaced by a
        // drift penalty and the hold counter runs; exiting resets it.
        if (scenario == .hover_hold and !self.done) {
            if (dist < self.scene.success_radius) {
                reward = -0.01 - 0.02 * self.bodyVel().length();
                self.hold_steps += 1;
                const need: u32 = @intFromFloat(self.scene.hold_s * 20.0); // 20 policy Hz
                if (self.hold_steps >= need) {
                    reward += 10.0;
                    self.succeeded = true;
                    self.done = true;
                }
            } else {
                self.hold_steps = 0;
            }
        }

        if (pos.y() < GROUND_Y and !self.done) {
            if (scenario == .land) {
                // touchdown: horizontal dist to the pad + vertical speed decide
                const ddx = self.scene.goal.x() - pos.x();
                const ddz = self.scene.goal.z() - pos.z();
                const dist_xz = @sqrt(ddx * ddx + ddz * ddz);
                const vs = @abs(self.bodyVel().y());
                if (dist_xz <= self.scene.success_radius and vs <= self.scene.max_touchdown_vs) {
                    reward += 10.0;
                    self.succeeded = true;
                } else {
                    reward -= 5.0;
                    self.collided = true;
                }
            } else {
                reward -= 5.0;
                self.collided = true;
            }
            self.done = true;
        }
        if (!self.done) {
            for (self.scene.obstacles) |ob| {
                if (ob.center.sub(pos).length() < ob.radius + DRONE_RADIUS) {
                    reward -= 5.0;
                    self.collided = true;
                    self.done = true;
                    break;
                }
            }
        }
        // in-air radius entry succeeds only for goto (land succeeds at
        // touchdown, hover_hold on completing the hold)
        if (!self.done and scenario == .goto and !wp_pending and dist < self.scene.success_radius) {
            reward += 10.0;
            self.succeeded = true;
            self.done = true;
        }
        // land approach shaping: below 1m near the pad, penalize horizontal
        // speed (discourages slamming in sideways)
        if (!self.done and scenario == .land and pos.y() < 1.0) {
            const ddx = self.scene.goal.x() - pos.x();
            const ddz = self.scene.goal.z() - pos.z();
            if (@sqrt(ddx * ddx + ddz * ddz) < 2.0 * self.scene.success_radius) {
                const vv = self.bodyVel();
                reward -= 0.01 * @sqrt(vv.x() * vv.x() + vv.z() * vv.z());
            }
        }
        if (!self.done and self.steps >= self.scene.max_steps) {
            self.done = true;
        }
        return .{ .reward = reward, .done = self.done };
    }

    /// Observation: 15 floats, exact QuadNavEnv layout:
    /// rel_goal/extent(3), vel/10(3), gravity-in-body/9.81(3),
    /// omega/10(3), nearest-obstacle-rel/extent(3)
    fn obs(self: *World) [15]f32 {
        const pos = self.bodyPos();
        const vel = self.bodyVel();
        const omega = self.bodyOmega();
        const q = self.bodyQuat();
        const q_conj = q.conjugate();

        const extent = @max(self.scene.extent, 1.0);
        const rel_goal = self.scene.goal.sub(pos).scale(1.0 / extent);
        const v = vel.scale(1.0 / 10.0);
        const g_body = Vec3.init(0, -9.81, 0).rotate_by_quaternion(q_conj).scale(1.0 / 9.81);
        const rates = omega.scale(1.0 / 10.0);
        var rel_obs = Vec3.zero();
        if (self.scene.obstacles.len > 0) {
            var best_d: f32 = std.math.inf(f32);
            var best = Vec3.zero();
            for (self.scene.obstacles) |ob| {
                const dvec = ob.center.sub(pos);
                const d = dvec.length();
                if (d < best_d) {
                    best_d = d;
                    best = dvec;
                }
            }
            rel_obs = best.scale(1.0 / extent);
        }
        return .{
            rel_goal.x(), rel_goal.y(), rel_goal.z(),
            v.x(),        v.y(),        v.z(),
            g_body.x(),   g_body.y(),   g_body.z(),
            rates.x(),    rates.y(),    rates.z(),
            rel_obs.x(),  rel_obs.y(),  rel_obs.z(),
        };
    }

    /// Rotate a world-frame vector into the yaw frame: level frame whose x
    /// axis is the body x axis projected to the world xz plane. At spawn
    /// (identity rotation) this is the identity, so the teacher's implicit
    /// yaw=0 world<->body mapping is exact in this frame at any heading.
    fn yawFrame(v: Vec3, yaw: f32) Vec3 {
        const c = @cos(yaw);
        const s = @sin(yaw);
        return Vec3.init(v.x() * c - v.z() * s, v.y(), v.x() * s + v.z() * c);
    }

    /// Observation v2: 19 floats. rel_goal, velocity and the obstacle vector
    /// are yaw-frame (heading-relative) so nav is yaw-invariant; g_body and
    /// rates are unchanged; then scenario one-hot(3) + success_radius/extent.
    fn obsV2(self: *World) [19]f32 {
        const pos = self.bodyPos();
        const vel = self.bodyVel();
        const omega = self.bodyOmega();
        const q = self.bodyQuat();
        const q_conj = q.conjugate();
        const fwd = Vec3.init(1, 0, 0).rotate_by_quaternion(q);
        const yaw = std.math.atan2(-fwd.z(), fwd.x());

        const extent = @max(self.scene.extent, 1.0);
        const rel_goal = yawFrame(self.scene.goal.sub(pos), yaw).scale(1.0 / extent);
        const v = yawFrame(vel, yaw).scale(1.0 / 10.0);
        const g_body = Vec3.init(0, -9.81, 0).rotate_by_quaternion(q_conj).scale(1.0 / 9.81);
        const rates = omega.scale(1.0 / 10.0);
        var rel_obs = Vec3.zero();
        if (self.scene.obstacles.len > 0) {
            var best_d: f32 = std.math.inf(f32);
            var best = Vec3.zero();
            for (self.scene.obstacles) |ob| {
                const dvec = ob.center.sub(pos);
                const d = dvec.length();
                if (d < best_d) {
                    best_d = d;
                    best = dvec;
                }
            }
            rel_obs = yawFrame(best, yaw).scale(1.0 / extent);
        }
        const one_hot: [3]f32 = switch (self.scene.scenario) {
            .goto => .{ 1, 0, 0 },
            .hover_hold => .{ 0, 1, 0 },
            .land => .{ 0, 0, 1 },
        };
        return .{
            rel_goal.x(), rel_goal.y(), rel_goal.z(),
            v.x(),        v.y(),        v.z(),
            g_body.x(),   g_body.y(),   g_body.z(),
            rates.x(),    rates.y(),    rates.z(),
            rel_obs.x(),  rel_obs.y(),  rel_obs.z(),
            one_hot[0],   one_hot[1],   one_hot[2],
            self.scene.success_radius / extent,
        };
    }

    /// Observation v3: 26 floats. The first 19 are exactly obsV2 (same
    /// order, same scales) so v2 policies warm-start with zero-padded input
    /// columns; appended are the T3 waypoint channels:
    ///   current-target rel(3) yaw-frame /extent - the next waypoint, or the
    ///     final goal once all waypoints are passed (equals rel_goal then);
    ///   next-hop rel(3) yaw-frame /extent - from the current waypoint to the
    ///     one after it (or to the final goal); zeros with no waypoints;
    ///   waypoint progress(1) - idx/count in [0,1]; 0 with no waypoints.
    fn obsV3(self: *World) [26]f32 {
        const base = self.obsV2();
        const q = self.bodyQuat();
        const fwd = Vec3.init(1, 0, 0).rotate_by_quaternion(q);
        const yaw = std.math.atan2(-fwd.z(), fwd.x());
        const extent = @max(self.scene.extent, 1.0);
        const wps = self.scene.waypoints;
        // v3.1: cur is a DELTA from the final goal (wp - goal), not an
        // absolute offset - on scenes with no pending waypoints it is exactly
        // zero, so a v2 warm-start with zero-padded weights is BIT-EXACT on
        // T0-T2 and the frozen-trunk T3 learner (ppo_v35) cannot regress them.
        var cur = Vec3.zero();
        var nxt = Vec3.zero();
        var prog: f32 = 0;
        if (wps.len > 0) {
            if (self.waypoint_idx < wps.len) {
                const wp = wps[self.waypoint_idx];
                cur = wp.sub(self.scene.goal);
                const after = if (self.waypoint_idx + 1 < wps.len) wps[self.waypoint_idx + 1] else self.scene.goal;
                nxt = after.sub(wp);
                prog = @as(f32, @floatFromInt(self.waypoint_idx)) / @as(f32, @floatFromInt(wps.len));
            } else {
                prog = 1.0;
            }
        }
        const cur_f = yawFrame(cur, yaw).scale(1.0 / extent);
        const nxt_f = yawFrame(nxt, yaw).scale(1.0 / extent);
        return .{
            base[0],  base[1],  base[2],  base[3],  base[4],  base[5],
            base[6],  base[7],  base[8],  base[9],  base[10], base[11],
            base[12], base[13], base[14], base[15], base[16], base[17],
            base[18],
            cur_f.x(), cur_f.y(), cur_f.z(),
            nxt_f.x(), nxt_f.y(), nxt_f.z(),
            prog,
        };
    }

    /// Observation v4: 27 floats. The first 26 are exactly obsV3; appended
    /// is the motor_v2 battery state of charge in [0,1] (1.0 when the v1
    /// plant is active). Exposes the SoC sag that drives hover/land trim
    /// drift so feedforward students (and stateless teachers) can
    /// compensate what the m2 teacher previously needed an integrator for.
    fn obsV4(self: *World) [27]f32 {
        const base = self.obsV3();
        var out: [27]f32 = undefined;
        for (base, 0..) |x, i| out[i] = x;
        out[26] = self.dyn.m2_soc;
        return out;
    }
};

fn writeObsReply(writer: anytype, w: *World, reward: f32, done: bool, with_info: bool) !void {
    if (w.obs_v4) {
        const o = w.obsV4();
        try writer.print("{{\"obs\":[{d:.6}", .{o[0]});
        for (o[1..]) |x| try writer.print(",{d:.6}", .{x});
    } else if (w.obs_v3) {
        const o = w.obsV3();
        try writer.print("{{\"obs\":[{d:.6}", .{o[0]});
        for (o[1..]) |x| try writer.print(",{d:.6}", .{x});
    } else if (w.obs_v2) {
        const o = w.obsV2();
        try writer.print("{{\"obs\":[{d:.6}", .{o[0]});
        for (o[1..]) |x| try writer.print(",{d:.6}", .{x});
    } else {
        const o = w.obs();
        try writer.print("{{\"obs\":[{d:.6}", .{o[0]});
        for (o[1..]) |x| try writer.print(",{d:.6}", .{x});
    }
    try writer.print("],\"reward\":{d:.6},\"done\":{}", .{ reward, done });
    if (with_info) {
        const p3 = w.bodyPos();
        const q4 = w.bodyQuat();
        const v3 = w.bodyVel();
        // telemetry powers the dashboard live stream (streamer reads
        // info.pos/quat/vel per policy step)
        try writer.print(",\"info\":{{\"collided\":{},\"succeeded\":{},\"steps\":{d},\"hold_steps\":{d},\"wp\":{d},\"pos\":[{d:.4},{d:.4},{d:.4}],\"quat\":[{d:.4},{d:.4},{d:.4},{d:.4}],\"vel\":[{d:.4},{d:.4},{d:.4}]}}", .{
            w.collided, w.succeeded, w.steps, w.hold_steps, w.waypoint_idx,
            p3.x(), p3.y(), p3.z(),
            q4.data[0], q4.data[1], q4.data[2], q4.data[3],
            v3.x(), v3.y(), v3.z(),
        });
    }
    try writer.writeAll("}\n");
    try writer.context.flush();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const stdin = std.io.getStdIn().reader();
    var stdout_buf = std.io.bufferedWriter(std.io.getStdOut().writer());
    const stdout = stdout_buf.writer();

    var world = try World.init(alloc);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    // Scene data (obstacle list) must outlive the line that parsed it -
    // separate lifetime from the per-line parse arena.
    var scene_arena = std.heap.ArenaAllocator.init(alloc);
    defer scene_arena.deinit();

    while (true) {
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();
        const line = stdin.readUntilDelimiterOrEofAlloc(a, '\n', 1 << 20) catch |e| {
            if (e == error.EndOfStream) break;
            return e;
        } orelse break;
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;

        const parsed = std.json.parseFromSlice(std.json.Value, a, trimmed, .{}) catch {
            try stdout.writeAll("{\"error\":\"bad json\"}\n");
            try stdout_buf.flush();
            continue;
        };
        const root = parsed.value;
        if (root != .object) continue;
        const cmd_v = root.object.get("cmd") orelse continue;
        const cmd = switch (cmd_v) {
            .string => |s| s,
            else => continue,
        };

        if (std.mem.eql(u8, cmd, "ping")) {
            try stdout.writeAll("{\"ok\":true}\n");
            try stdout_buf.flush();
        } else if (std.mem.eql(u8, cmd, "close")) {
            break;
        } else if (std.mem.eql(u8, cmd, "reset")) {
            const scene_v = root.object.get("scene") orelse {
                try stdout.writeAll("{\"error\":\"reset needs scene\"}\n");
                try stdout_buf.flush();
                continue;
            };
            var scene = Scene{
                .spawn = try vecFromJson(scene_v.object.get("spawn").?),
                .goal = try vecFromJson(scene_v.object.get("goal").?),
                .obstacles = &.{},
            };
            if (scene_v.object.get("spawn_vel")) |sv| scene.spawn_vel = try vecFromJson(sv);
            if (scene_v.object.get("extent")) |e| scene.extent = f32FromJson(e, 40.0);
            if (scene_v.object.get("max_steps")) |m| scene.max_steps = @intFromFloat(f32FromJson(m, 250));
            if (scene_v.object.get("dynamics_noise")) |n| scene.dynamics_noise = f32FromJson(n, 0.05);
            if (scene_v.object.get("success_radius")) |r| scene.success_radius = f32FromJson(r, 2.0);
            if (scene_v.object.get("hold_s")) |h| scene.hold_s = f32FromJson(h, 4.0);
            if (scene_v.object.get("max_touchdown_vs")) |tv| scene.max_touchdown_vs = f32FromJson(tv, 0.5);
            if (scene_v.object.get("shaping_v2")) |sv| scene.shaping_v2 = (sv == .bool and sv.bool);
            if (scene_v.object.get("face_reward_w")) |fw| scene.face_reward_w = f32FromJson(fw, 0.0);
            if (scene_v.object.get("scenario")) |sc| {
                if (sc == .string) {
                    if (std.mem.eql(u8, sc.string, "hover_hold")) scene.scenario = .hover_hold;
                    if (std.mem.eql(u8, sc.string, "land")) scene.scenario = .land;
                }
            }
            _ = scene_arena.reset(.retain_capacity);
            if (scene_v.object.get("obstacles")) |obs_v| {
                const arr = obs_v.array;
                var list = try scene_arena.allocator().alloc(Obstacle, arr.items.len);
                for (arr.items, 0..) |item, i| {
                    const oarr = item.array.items;
                    if (oarr.len != 4) return error.BadObstacle;
                    var vals: [4]f32 = undefined;
                    for (oarr, 0..) |ov, j| vals[j] = f32FromJson(ov, 0);
                    list[i] = .{
                        .center = Vec3.init(vals[0], vals[1], vals[2]),
                        .radius = vals[3],
                    };
                }
                scene.obstacles = list;
            }
            if (scene_v.object.get("waypoints")) |wps_v| {
                const arr = wps_v.array;
                const wlist = try scene_arena.allocator().alloc(Vec3, arr.items.len);
                for (arr.items, 0..) |item, i| {
                    wlist[i] = try vecFromJson(item);
                }
                scene.waypoints = wlist;
            }
            const seed: u64 = if (root.object.get("seed")) |s| switch (s) {
                .integer => |n| @intCast(n),
                .float => |f| @intFromFloat(f),
                else => 0,
            } else 0;
            world.reset(scene, seed);
            try writeObsReply(stdout, world, 0.0, false, false);
        } else if (std.mem.eql(u8, cmd, "set_gains")) {
            // Tuning support (RATE_CONTROLLER_ASSESSMENT.md): swap PID gains
            // per axis at runtime and reset controller/filter state, so a
            // sweep never needs a rebuild. Firmware defaults stay the
            // compiled-in baseline; this only affects this process.
            if (root.object.get("roll")) |g| {
                const ga = g.array.items;
                if (ga.len == 3) world.rate_ctrl.pid.roll.setGains(f32FromJson(ga[0], 0), f32FromJson(ga[1], 0), f32FromJson(ga[2], 0));
            }
            if (root.object.get("pitch")) |g| {
                const ga = g.array.items;
                if (ga.len == 3) world.rate_ctrl.pid.pitch.setGains(f32FromJson(ga[0], 0), f32FromJson(ga[1], 0), f32FromJson(ga[2], 0));
            }
            if (root.object.get("yaw")) |g| {
                const ga = g.array.items;
                if (ga.len == 3) world.rate_ctrl.pid.yaw.setGains(f32FromJson(ga[0], 0), f32FromJson(ga[1], 0), f32FromJson(ga[2], 0));
            }
            world.rate_ctrl.reset();
            world.filtered_rates = .{ 0, 0, 0 };
            world.filtered_thrust = 0;
            try stdout.writeAll("{\"ok\":true}\n");
            try stdout_buf.flush();
        } else if (std.mem.eql(u8, cmd, "rate_step")) {
            // Tuning probe (RATE_CONTROLLER_ASSESSMENT.md): hold a constant
            // body-rate setpoint for `ticks` fast ticks at `thrust` (N) and
            // stream [t, wx, wy, wz, tq_x, tq_y, tq_z] every `sample_every`.
            // Runs the same fastStep path as episodes, so his input filters
            // and PID are all in the loop. noise=0 recommended for clean reads.
            const sp_v = root.object.get("setpoint") orelse continue;
            const sp_arr = sp_v.array.items;
            if (sp_arr.len != 3) continue;
            var sp: [3]f32 = undefined;
            for (sp_arr, 0..) |v, i| sp[i] = f32FromJson(v, 0);
            const ticks: usize = if (root.object.get("ticks")) |t| @intFromFloat(f32FromJson(t, 500)) else 500;
            const every: usize = if (root.object.get("sample_every")) |t| @max(1, @as(usize, @intFromFloat(f32FromJson(t, 5)))) else 5;
            const thrust: f32 = if (root.object.get("thrust")) |t| f32FromJson(t, world.dyn.mass * 9.81) else world.dyn.mass * 9.81;
            if (root.object.get("noise")) |n| world.scene.dynamics_noise = f32FromJson(n, 0);
            try stdout.writeAll("{\"samples\":[");
            var i: usize = 0;
            while (i < ticks) : (i += 1) {
                world.fastStep(sp, thrust);
                if (i % every == 0) {
                    const om = world.bodyOmega();
                    const t = @as(f32, @floatFromInt(i)) * FAST_DT;
                    if (i > 0) try stdout.writeAll(",");
                    try stdout.print("[{d:.4},{d:.4},{d:.4},{d:.4},{d:.5},{d:.5},{d:.5}]", .{ t, om.x(), om.y(), om.z(), world.last_torque[0], world.last_torque[1], world.last_torque[2] });
                }
            }
            try stdout.writeAll("],\"ok\":true}\n");
            try stdout_buf.flush();
        } else if (std.mem.eql(u8, cmd, "set_dynamics")) {
            // Manifest-driven dynamics (dronestudio.chassis/1.1). CAD frame is
            // +X fwd / +Z up; sim body is +Y up. Diagonal remap (proper
            // rotation): sim.x = cad.y, sim.y = cad.z, sim.z = cad.x.
            // "abstract" restores the built-in QuadNavEnv-parity profile.
            const path_v = root.object.get("path") orelse continue;
            if (path_v != .string) continue;
            if (std.mem.eql(u8, path_v.string, "abstract")) {
                world.dyn = .{}; // motor_mode resets to false here too
                const inertia = [3]f32{ IXX, IYY, IZZ };
                bullet.cbtBodySetMassProps(world.body, MASS, &inertia);
                try stdout.writeAll("{\"ok\":true,\"dynamics\":\"abstract\"}\n");
                try stdout_buf.flush();
                continue;
            }
            var mparsed = CM.ChassisManifest.load(a, path_v.string) catch {
                try stdout.writeAll("{\"ok\":false,\"error\":\"manifest load failed\"}\n");
                try stdout_buf.flush();
                continue;
            };
            defer mparsed.deinit();
            const m = &mparsed.value;
            const I = m.dynamics.inertia_about_com_kgm2;
            world.dyn.mass = m.totalMassKg();
            world.dyn.inertia = .{ @floatCast(I.iyy), @floatCast(I.izz), @floatCast(I.ixx) };
            world.dyn.max_thrust = m.maxThrustPerMotorN() * 4.0;
            world.dyn.arm_length = m.armLengthM();
            world.dyn.motor_lag_s = m.motorTimeConstantS();
            world.dyn.motor_drag_ratio = m.motorDragRatio();
            world.dyn.motor_max_thrust = m.maxThrustPerMotorN();
            if (m.aero) |ae| {
                world.dyn.aero_area = .{ @floatCast(ae.projected_area_m2.y), @floatCast(ae.projected_area_m2.z), @floatCast(ae.projected_area_m2.x) };
                world.dyn.aero_cd = @floatCast(ae.cd_flat_plate_estimate);
            }
            // Motor model: solve the 4x4 mixing system once (motors are fixed
            // per design). Rows: [sum=T, tau_x, tau_y(ktau yaw), tau_z].
            // CAD -> sim body map: sim = (-cad.y, cad.z, -cad.x).
            // CoM offset (max-fidelity directive): torques must be taken
            // about the true com, so motor lever arms are position - com.
            const com_cad = m.comM();
            const com_sim = [3]f64{ -com_cad[1], com_cad[2], -com_cad[0] };
            world.dyn.com = .{ @floatCast(com_sim[0]), @floatCast(com_sim[1]), @floatCast(com_sim[2]) };
            if (m.motors.len == 4) {
                var A: [4][4]f64 = undefined;
                for (m.motors, 0..) |mo, i| {
                    const px = -mo.position_m[1] - com_sim[0];
                    const py = mo.position_m[2] - com_sim[1];
                    const pz = -mo.position_m[0] - com_sim[2];
                    world.dyn.motor_pos[i] = .{ @floatCast(px), @floatCast(py), @floatCast(pz) };
                    const s: f64 = if (std.mem.eql(u8, mo.direction, "cw")) 1.0 else -1.0; // sim +Y up = -z_NED: flipped vs his NED signs
                    world.dyn.motor_yaw_sign[i] = @floatCast(s);
                    A[0][i] = 1.0;
                    A[1][i] = -pz; // torque about body x
                    A[2][i] = s * mo.drag_ratio; // yaw (ktau), about body y (up)
                    A[3][i] = px; // torque about body z
                }
                world.dyn.mix_inv = invert4(A);
                world.dyn.motor_mode = true;
                // m2_* defaults: World.init uses create() (undefined memory),
                // so struct field defaults never applied. Assign explicitly.
                world.dyn.motor_v2 = false;
                world.dyn.m2_omega = .{ 0, 0, 0, 0 };
                world.dyn.m2_hist_idx = 0;
                world.dyn.m2_omega_max = 4046.0; // rad/s: 2300KV * 16.8V absolute no-load cap; the electrical equilibrium self-limits below this (~3500 at full charge)
                world.dyn.m2_kd_over_kf = 0.015; // physical 5in prop torque/thrust ratio, m. NOTE: manifest drag_ratio=0.15 is non-physical for a 2205; mixer allocation still uses the manifest value
                world.dyn.m2_ke = 0.004152; // V*s/rad = N*m/A = 60/(2*pi*2300) from RS2205 2300KV spec
                world.dyn.m2_rw = 0.065; // RS2205-2300KV internal resistance spec, ohm (AKK clone of EMAX design)
                world.dyn.m2_jr = 9.0e-6; // rotor+prop inertia, kg*m^2 (ESTIMATE)
                world.dyn.m2_imax = 30.0; // A; thrust-stand peak 28.6A (HQ5045BN 4S), ESC assumed 30A class
                world.dyn.m2_batt_r_int = 0.024; // 4S pack internal resistance, ohm (ESTIMATE)
                world.dyn.m2_batt_cap_as = 4680.0; // charge, A*s (~1.3Ah, ASSUMPTION)
                world.dyn.m2_soc = 1.0;
                world.dyn.m2_esc_delay_ticks = 1; // DShot latency in 500Hz ticks (ESTIMATE)
                world.dyn.m2_kf = 7.9e-7; // N/(rad/s)^2, thrust-stand fit (oscarliang RS2205-2300KV HQ5045BN 4S); NOT derived from max_thrust so CAD thrust-cap changes don't recalibrate the prop
                world.dyn.m2_prop_d = @floatCast(m.motors[0].prop_diameter_m);
                world.dyn.m2_j0 = 1.2;
                world.dyn.m2_ge = true;
                inline for (0..4) |i| world.dyn.m2_cmd_hist[i] = .{ 0, 0, 0, 0 };
            }
            const n = @min(m.name.len, world.dyn_name_buf.len);
            @memcpy(world.dyn_name_buf[0..n], m.name[0..n]);
            world.dyn_name_len = n;
            bullet.cbtBodySetMassProps(world.body, world.dyn.mass, &world.dyn.inertia);
            const cross_max = @max(@abs(I.ixy), @max(@abs(I.ixz), @abs(I.iyz)));
            const diag_min = @min(I.ixx, @min(I.iyy, I.izz));
            try stdout.print("{{\"ok\":true,\"dynamics\":\"{s}\",\"mass\":{d:.4},\"inertia\":[{d:.6},{d:.6},{d:.6}],\"cross_term_ratio\":{d:.4},\"com\":[{d:.5},{d:.5},{d:.5}]}}\n", .{ m.name, world.dyn.mass, world.dyn.inertia[0], world.dyn.inertia[1], world.dyn.inertia[2], cross_max / diag_min, world.dyn.com[0], world.dyn.com[1], world.dyn.com[2] });
            try stdout_buf.flush();
        } else if (std.mem.eql(u8, cmd, "motor_v2")) {
            const on_v = root.object.get("on") orelse continue;
            world.dyn.motor_v2 = (on_v == .bool and on_v.bool);
            // Yaw allocation must match the ACTIVE physics (2026-09-04 yaw
            // fix): the m2 path applies drag torque kd*omega^2 with
            // kd_over_kf = 0.015, while the lag-model path applies
            // ti*drag_ratio (manifest 0.15). Allocating yaw with the
            // manifest ratio under m2 commanded a ~10x thrust split per
            // unit yaw torque; the slow motors pinned at the min-thrust
            // clamp, corrupting roll/pitch allocation whenever collective
            // or attitude was modulated on top - the long-standing
            // "sustained yaw destabilizes the quad" issue. Rebuild mix_inv
            // with the yaw ratio of the ACTIVE model. Zero-yaw flight is
            // bit-exact either way (tq_y = 0 contributes nothing).
            if (world.dyn.motor_mode) {
                const yr: f64 = if (world.dyn.motor_v2) world.dyn.m2_kd_over_kf else world.dyn.motor_drag_ratio;
                var A: [4][4]f64 = undefined;
                for (0..4) |i| {
                    A[0][i] = 1.0;
                    A[1][i] = -@as(f64, world.dyn.motor_pos[i][2]);
                    A[2][i] = @as(f64, world.dyn.motor_yaw_sign[i]) * yr;
                    A[3][i] = @as(f64, world.dyn.motor_pos[i][0]);
                }
                world.dyn.mix_inv = invert4(A);
            }
            try stdout.print("{{\"ok\":true,\"motor_v2\":{}}}\n", .{world.dyn.motor_v2});
            try stdout_buf.flush();
        } else if (std.mem.eql(u8, cmd, "obs_v2")) {
            const on_v = root.object.get("on") orelse continue;
            world.obs_v2 = (on_v == .bool and on_v.bool);
            try stdout.print("{{\"ok\":true,\"obs_v2\":{}}}\n", .{world.obs_v2});
            try stdout_buf.flush();
        } else if (std.mem.eql(u8, cmd, "obs_v3")) {
            const on_v = root.object.get("on") orelse continue;
            world.obs_v3 = (on_v == .bool and on_v.bool);
            try stdout.print("{{\"ok\":true,\"obs_v3\":{}}}\n", .{world.obs_v3});
            try stdout_buf.flush();
        } else if (std.mem.eql(u8, cmd, "obs_v4")) {
            const on_v = root.object.get("on") orelse continue;
            world.obs_v4 = (on_v == .bool and on_v.bool);
            try stdout.print("{{\"ok\":true,\"obs_v4\":{}}}\n", .{world.obs_v4});
            try stdout_buf.flush();
        } else if (std.mem.eql(u8, cmd, "step")) {
            const act_v = root.object.get("action") orelse continue;
            const arr = act_v.array.items;
            if (arr.len != 4) continue;
            var action: [4]f32 = undefined;
            for (arr, 0..) |av, i| action[i] = f32FromJson(av, 0);
            const r = world.policyStep(action);
            try writeObsReply(stdout, world, r.reward, r.done, true);
        }
    }
}
