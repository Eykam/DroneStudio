const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const PIDController = @import("../../flight/PIDController.zig");
const IMUSensor = @import("IMUSensor.zig");
const PhysicsThread = @import("PhysicsThread.zig");
const Controller = @import("Controller.zig");

const Vec3 = Math.Vec3;
const Quaternion = Math.Quaternion;
const Mat3 = Math.Mat3;
const IMUSample = IMUSensor.IMUSample;

/// Coordinate system adapter for flight control using transformation matrices
const CoordinateAdapter = struct {
    const Self = @This();

    // Quaternions built *once* from the same matrices
    q_FN: Quaternion, // Engine → Flight
    q_NF: Quaternion, // Flight → Engine (conjugate)

    /// Get gravity vector in engine coordinates
    gravity_engine: Vec3,

    const EngineToFlightResult = struct {
        q_NB: Quaternion, // roll, pitch, yaw  (rad)
        rates_flight: Vec3, // p, q, r          (rad/s)
    };

    const FlightToEngineResult = struct {
        torque_engine: Vec3, // body torque in Engine frame
        force_engine: Vec3, // world force in Engine frame
    };

    /// Convert engine pose to flight control coordinates
    pub fn engineToFlight(
        self: Self,
        q_WB_engine: Quaternion, // body attitude in Engine world frame
        omega_B_engine: Vec3, // angular vel in Engine body axes
    ) EngineToFlightResult {
        const q_NB = self.q_FN.multiply(q_WB_engine);
        const rates = omega_B_engine.rotate_by_quaternion(self.q_FN);

        return .{
            .q_NB = q_NB,
            .rates_flight = rates,
        };
    }

    /// Convert flight control outputs to engine coordinates
    pub fn flightToEngine(
        self: Self,
        torque_flight: Vec3, // body torque  in Flight axes
        force_flight: Vec3, // world force  in Flight world
    ) FlightToEngineResult {
        const tau_e = torque_flight.rotate_by_quaternion(self.q_NF);
        const F_e = force_flight.rotate_by_quaternion(self.q_NF);

        return .{
            .torque_engine = tau_e,
            .force_engine = F_e,
        };
    }
};

/// OpenGL coordinate system adapter
/// Uses the transformation matrices provided: NED ↔ OpenGL
pub const OpenGLAdapter = struct {
    pub fn create() CoordinateAdapter {
        // Column-major array for Mat3.from_array()
        // Engine: X->East, Y->Up, Z->South => Flight: NED (X->North, Y->East, Z->Down)
        //     N =  0·X + 0·Y −1·Z
        //     E =  1·X + 0·Y + 0·Z
        //     D =  0·X −1·Y + 0·Z
        const R_FN = Mat3.from_array([9]f32{
            0, 0,  -1,
            1, 0,  0,
            0, -1, 0,
        });

        const q_FN = Quaternion.from_mat3(R_FN);
        const q_NF = q_FN.conjugate();

        return .{
            .q_FN = q_FN,
            .q_NF = q_NF,
            .gravity_engine = Vec3.init(0, -9.81, 0),
        };
    }
};

/// Input state for processing raw inputs
pub const InputState = struct {
    // Key states
    arm: bool = false,
    disarm: bool = false,
    throttle_up: bool = false,
    throttle_down: bool = false,
    yaw_left: bool = false,
    yaw_right: bool = false,

    // Mouse movement (pixels per frame)
    mouse_dx: f32 = 0.0,
    mouse_dy: f32 = 0.0,
};

/// Input processing parameters
pub const InputParams = struct {
    // Input sensitivity
    throttle_sensitivity: f32 = 200.0, // N/s (thrust change rate)
    yaw_sensitivity: f32 = 3.14, // rad/s per key press
    roll_pitch_sensitivity: f32 = 0.1, // rad/s per pixel of mouse movement

    // Input limits (safety bounds)
    max_roll_rate: f32 = 10.47, // rad/s (600 deg/s)
    max_pitch_rate: f32 = 10.47, // rad/s (600 deg/s)
    max_yaw_rate: f32 = 5.24, // rad/s (300 deg/s)
    max_thrust: f32 = 40.0, // Newtons (3g for 1kg drone)

    // Low-pass filter parameters
    throttle_filter_tau: f32 = 0.05, // seconds
    rate_filter_tau: f32 = 0.05, // seconds
};

/// Rate controller - direct angular velocity control
pub const RateController = struct {
    const Self = @This();

    roll_gains: [3]f32 = [3]f32{ 0.1, 0.005, 0.001 },
    pitch_gains: [3]f32 = [3]f32{ 0.1, 0.005, 0.001 },
    yaw_gains: [3]f32 = [3]f32{ 0.05, 0.003, 0.0005 },
    max_roll: f32 = 10.47, // rad/s
    max_pitch: f32 = 10.47, // rad/s
    max_yaw: f32 = 5.24, // rad/s

    // PID controllers for each axis
    pid: PIDController.PosePIDController,

    // Input processing state
    filtered_thrust: f32 = 0.0,
    filtered_rates: [3]f32 = [3]f32{ 0, 0, 0 },

    pub fn init(
        integrator_limit: f32,
        output_limit: f32,
    ) Self {
        var self = Self{
            .pid = undefined,
        };

        self.pid = PIDController.PosePIDController.init(
            self.roll_gains,
            self.pitch_gains,
            self.yaw_gains,
            integrator_limit,
            output_limit,
        );

        return self;
    }

    /// Process input and generate control setpoints
    pub fn processInput(
        self: *Self,
        input_state: InputState,
        params: InputParams,
        dt: f32,
    ) ControlSetpoints {
        // Process throttle input
        var throttle_command: f32 = 0.0;
        if (input_state.throttle_up) {
            throttle_command += params.throttle_sensitivity * dt;
        }
        if (input_state.throttle_down) {
            throttle_command -= params.throttle_sensitivity * dt;
        }

        // Apply low-pass filter to throttle
        const throttle_alpha = dt / (params.throttle_filter_tau + dt);
        const target_thrust = std.math.clamp(self.filtered_thrust + throttle_command, 0.0, params.max_thrust);
        self.filtered_thrust = self.filtered_thrust + throttle_alpha * (target_thrust - self.filtered_thrust);

        // Process yaw (rate control)
        var yaw_rate: f32 = 0.0;
        if (input_state.yaw_left) {
            yaw_rate += params.yaw_sensitivity;
        }
        if (input_state.yaw_right) {
            yaw_rate -= params.yaw_sensitivity;
        }
        yaw_rate = std.math.clamp(yaw_rate, -params.max_yaw_rate, params.max_yaw_rate);

        // Apply filter to rates
        const rate_alpha = dt / (params.rate_filter_tau + dt);
        self.filtered_rates[2] = self.filtered_rates[2] + rate_alpha * (yaw_rate - self.filtered_rates[2]);

        // Mouse directly controls angular rates
        var raw_rates = [3]f32{ 0, 0, 0 };
        raw_rates[0] = input_state.mouse_dx * params.roll_pitch_sensitivity;
        raw_rates[1] = -input_state.mouse_dy * params.roll_pitch_sensitivity;

        // Apply limits
        raw_rates[0] = std.math.clamp(raw_rates[0], -params.max_roll_rate, params.max_roll_rate);
        raw_rates[1] = std.math.clamp(raw_rates[1], -params.max_pitch_rate, params.max_pitch_rate);

        // Apply low-pass filter to roll/pitch rates
        self.filtered_rates[0] = self.filtered_rates[0] + rate_alpha * (raw_rates[0] - self.filtered_rates[0]);
        self.filtered_rates[1] = self.filtered_rates[1] + rate_alpha * (raw_rates[1] - self.filtered_rates[1]);

        return .{
            .Rate = .{
                .rates = self.filtered_rates,
                .thrust = self.filtered_thrust,
            },
        };
    }

    /// Process rate control loop
    pub fn update(
        self: *Self,
        desired_rates: [3]f32,
        current_rates: Vec3,
        dt: f32,
    ) [3]f32 {
        // Calculate rate errors (setpoint - actual)
        const rate_err = [3]f32{
            desired_rates[0] - current_rates.x(),
            desired_rates[1] - current_rates.y(),
            desired_rates[2] - current_rates.z(),
        };

        // Run PID controllers to get desired torques
        return self.pid.step(rate_err, dt);
    }

    pub fn reset(self: *Self) void {
        self.pid.reset();
        self.filtered_thrust = 0.0;
        self.filtered_rates = [3]f32{ 0, 0, 0 };
    }
};

/// Attitude controller - orientation stabilization
pub const AttitudeController = struct {
    const Self = @This();

    Kp_roll: f32 = 0.05,
    Kp_pitch: f32 = 0.05,
    Kp_yaw: f32 = 0.02,

    max_roll: f32 = 0.52, // rad (30 degrees)
    max_pitch: f32 = 0.52, // rad (30 degrees)
    max_yaw_rate: f32 = 5.24, // rad/s (yaw stays rate mode)

    // Inner rate controller
    rate_controller: RateController,

    // Input processing state
    filtered_thrust: f32 = 0.0,
    filtered_yaw_rate: f32 = 0.0,
    roll_angle: f32 = 0.0,
    pitch_angle: f32 = 0.0,

    pub fn init(
        integrator_limit: f32,
        output_limit: f32,
    ) Self {
        var self = Self{
            .rate_controller = undefined,
        };

        self.rate_controller = RateController.init(
            integrator_limit,
            output_limit,
        );

        return self;
    }

    /// Process input and generate control setpoints
    pub fn processInput(
        self: *Self,
        input_state: InputState,
        params: InputParams,
        dt: f32,
    ) ControlSetpoints {
        // Process throttle input
        var throttle_command: f32 = 0.0;
        if (input_state.throttle_up) {
            throttle_command += params.throttle_sensitivity * dt;
        }
        if (input_state.throttle_down) {
            throttle_command -= params.throttle_sensitivity * dt;
        }

        // Apply low-pass filter to throttle
        const throttle_alpha = dt / (params.throttle_filter_tau + dt);
        const target_thrust = std.math.clamp(self.filtered_thrust + throttle_command, 0.0, params.max_thrust);
        self.filtered_thrust = self.filtered_thrust + throttle_alpha * (target_thrust - self.filtered_thrust);

        // Process yaw (rate control even in attitude mode)
        var yaw_rate: f32 = 0.0;
        if (input_state.yaw_left) {
            yaw_rate += params.yaw_sensitivity;
        }
        if (input_state.yaw_right) {
            yaw_rate -= params.yaw_sensitivity;
        }
        yaw_rate = std.math.clamp(yaw_rate, -params.max_yaw_rate, params.max_yaw_rate);

        // Apply filter to yaw rate
        const rate_alpha = dt / (params.rate_filter_tau + dt);
        self.filtered_yaw_rate = self.filtered_yaw_rate + rate_alpha * (yaw_rate - self.filtered_yaw_rate);

        // Mouse controls desired angles
        const angle_sensitivity = 0.02; // rad per pixel

        // Integrate mouse movement to get angles
        self.roll_angle += input_state.mouse_dx * angle_sensitivity;
        self.pitch_angle += -input_state.mouse_dy * angle_sensitivity;

        // Clamp angles to reasonable limits (±30 degrees)
        const max_angle: f32 = 0.52; // ~30 degrees in radians
        self.roll_angle = std.math.clamp(self.roll_angle, -max_angle, max_angle);
        self.pitch_angle = std.math.clamp(self.pitch_angle, -max_angle, max_angle);

        return .{
            .Attitude = .{
                .angles = [2]f32{ self.roll_angle, self.pitch_angle },
                .yaw_rate = self.filtered_yaw_rate,
                .thrust = self.filtered_thrust,
            },
        };
    }

    /// Process attitude control loop (cascade with rate control)
    pub fn update(
        self: *Self,
        q_des: Quaternion,
        q_act: Quaternion,
        w_act: Vec3,
        yaw_sp: f32,
        dt: f32,
    ) [3]f32 {
        // Error quaternion: body-fixed rotation actual -> desired
        const q_err = q_des.multiply(q_act.conjugate()).normalize();

        // For small angles (|θ| < ~60°): 2·vector_part ~= rotation error (rad)
        const e = Vec3.init(
            q_err.x() * 2.0,
            q_err.y() * 2.0,
            q_err.z() * 2.0,
        );

        // Desired body rates from proportional attitude error
        const w_sp = Vec3.init(
            self.Kp_roll * e.x(),
            self.Kp_pitch * e.y(),
            yaw_sp + self.Kp_yaw * e.z(), // yaw rate + yaw hold
        );

        std.debug.print("Q_ERR: {}E: {d}\nW_SP: {}\n\n", .{
            q_err,
            [_]f32{
                Math.degrees(e.x()),
                Math.degrees(e.y()),
                Math.degrees(e.z()),
            },
            w_sp,
        });

        // Clamp rate set-points
        // ω_sp = ω_sp.clamp(-max_rate, max_rate);

        // Inner rate loop → torque
        return self.rate_controller.update(w_sp.data, w_act, dt);
    }

    pub fn reset(self: *Self) void {
        self.rate_controller.reset();
        self.filtered_thrust = 0.0;
        self.filtered_yaw_rate = 0.0;
        self.roll_angle = 0.0;
        self.pitch_angle = 0.0;
    }
};

pub const RateSetpoints = struct {
    rates: [3]f32 = [3]f32{ 0, 0, 0 }, // [roll, pitch, yaw] rad/s
    thrust: f32 = 0.0, // Newtons total
};

/// Attitude controller setpoints
pub const AttitudeSetpoints = struct {
    angles: [2]f32 = [2]f32{ 0, 0 }, // [roll, pitch] rad
    yaw_rate: f32 = 0.0, // rad/s (yaw stays in rate mode)
    thrust: f32 = 0.0, // Newtons total
};

/// Tagged union matching the controller type
pub const ControlSetpoints = union(ControllerType) {
    Rate: RateSetpoints,
    Attitude: AttitudeSetpoints,

    pub fn init(controller_type: ControllerType) ControlSetpoints {
        return switch (controller_type) {
            .Rate => .{ .Rate = RateSetpoints{} },
            .Attitude => .{ .Attitude = AttitudeSetpoints{} },
        };
    }

    /// Get thrust regardless of mode
    pub fn getThrust(self: ControlSetpoints) f32 {
        return switch (self) {
            .Rate => |r| r.thrust,
            .Attitude => |a| a.thrust,
        };
    }
};

pub const ControllerType = enum { Rate, Attitude };
pub const DroneController = union(ControllerType) {
    Rate: RateController,
    Attitude: AttitudeController,

    /// Process input and generate control setpoints
    pub fn processInput(
        self: *@This(),
        input_state: InputState,
        params: InputParams,
        dt: f32,
    ) ControlSetpoints {
        return switch (self.*) {
            .Rate => |*rate| rate.processInput(input_state, params, dt),
            .Attitude => |*attitude| attitude.processInput(input_state, params, dt),
        };
    }

    /// Update the controller based on its type
    pub fn update(
        self: *@This(),
        setpoints: ControlSetpoints,
        q_des: Quaternion, // desired body attitude (Flight)
        q_act: Quaternion, // current attitude               "
        w_act: Vec3, // current body rates (Flight)
        dt: f32,
    ) [3]f32 {
        switch (self.*) {
            .Rate => |*rc| {
                const rate_setpoints = setpoints.Rate;
                return rc.update(rate_setpoints.rates, w_act, dt);
            },
            .Attitude => |*ac| {
                const att_setpoints = setpoints.Attitude;
                return ac.update(q_des, q_act, w_act, att_setpoints.yaw_rate, dt);
            },
        }
    }

    pub fn reset(self: *@This()) void {
        switch (self.*) {
            .Rate => |*rate_ctrl| rate_ctrl.reset(),
            .Attitude => |*att_ctrl| att_ctrl.reset(),
        }
    }
};

/// Flight controller configuration parameters
pub const FlightControllerParams = struct {
    // Control limits
    max_torque_per_kg: f32 = 2.0, // N⋅m per kg (increased for realistic control authority)

    // Motor configuration for quad-X
    motor_arm_length: f32 = 0.15, // meters
    motor_drag_ratio: f32 = 0.15, // kτ = c_drag / c_thrust

    // Motor dynamics
    motor_time_constant: f32 = 0.04, // seconds (40ms lag)
    motor_max_thrust: f32 = 10.0, // Newtons per motor

    // Control rate
    control_rate_hz: u32 = 400, // 400 Hz control loop
};

/// Motor commands (after mixing and lag filtering)
pub const MotorCommands = struct {
    motor_thrusts: [4]f32 = [4]f32{ 0, 0, 0, 0 }, // Newtons per motor
    total_thrust_world: Vec3 = Vec3.init(0, 0, 0), // World frame force vector
    total_torque_body: Vec3 = Vec3.init(0, 0, 0), // Body frame torque vector
};

/// Flight controller component
pub const FlightControllerComponent = struct {
    const Self = @This();

    // Configuration
    params: FlightControllerParams = FlightControllerParams{},
    entity_id: Core.EntityID = undefined,
    mass: f32 = 1.0, // kg (will be set from rigid body)

    // Coordinate system adapter (optional - null means engine and flight coordinates are the same)
    coord_adapter: ?CoordinateAdapter,

    armed: bool = false,
    controller: DroneController,
    controller_type: ControllerType,
    setpoints: ControlSetpoints,
    motor_commands: MotorCommands = MotorCommands{},

    // State estimation (in engine coordinates)
    attitude_estimate: Quaternion = Quaternion.identity(),
    rate_estimate: Vec3 = Vec3.init(0, 0, 0),

    // Motor lag filtering state
    motor_filtered: [4]f32 = [4]f32{ 0, 0, 0, 0 },
    last_update_us: u64 = 0,

    pub fn init(
        params: FlightControllerParams,
        mass: f32,
        controller_type: ControllerType,
        coord_adapter: ?CoordinateAdapter,
    ) Self {
        const torque_limit = params.max_torque_per_kg * mass;
        const integrator_limit = torque_limit * 0.5; // 50% for integral windup

        const controller = switch (controller_type) {
            .Rate => DroneController{ .Rate = RateController.init(integrator_limit, torque_limit) },
            .Attitude => DroneController{ .Attitude = AttitudeController.init(integrator_limit, torque_limit) },
        };

        return Self{
            .params = params,
            .mass = mass,
            .controller = controller,
            .controller_type = controller_type,
            .setpoints = ControlSetpoints.init(controller_type),
            .coord_adapter = coord_adapter,
        };
    }

    pub fn attach(self: *Self, ecs: anytype, eid: Core.EntityID) !void {
        self.entity_id = eid;
        try ecs.flight_controller_components.add(eid, self.*);
    }

    /// Update control setpoints from input system
    pub fn setControlSetpoints(self: *Self, setpoints: ControlSetpoints) void {
        // Ensure setpoints match controller type
        std.debug.assert(@as(ControllerType, self.controller) == @as(ControllerType, setpoints));
        self.setpoints = setpoints;
    }

    /// Process IMU sample and update state estimate (complementary filter)
    pub fn updateStateEstimate(self: *Self, imu_sample: IMUSample, dt: f32) void {
        // Extract gyro rates and apply low-pass filter for rate estimate
        const raw_rates, const raw_accel = if (self.coord_adapter) |adapter| blk: {
            const rates_opengl = Vec3.from_array(imu_sample.gyro);
            const accel_opengl = Vec3.from_array(imu_sample.accel);

            break :blk .{
                rates_opengl.rotate_by_quaternion(adapter.q_FN),
                accel_opengl.rotate_by_quaternion(adapter.q_FN),
            };
        } else .{
            // No adapter - keep in OpenGL
            Vec3.from_array(imu_sample.gyro),
            Vec3.from_array(imu_sample.accel),
        };

        // === COMPLEMENTARY FILTER FOR ATTITUDE ===
        // Integrate gyroscope for short-term attitude (high frequency)
        const theta = raw_rates.scale(dt); // Small angle approximation
        const dq = Quaternion.init(
            theta.x() * 0.5,
            theta.y() * 0.5,
            theta.z() * 0.5,
            1.0,
        ).normalize();

        var q_gyro = self.attitude_estimate.multiply(dq).normalize();
        const accel_mag = raw_accel.length();

        // Only use accelerometer if it's measuring mostly gravity (not accelerating)
        // if (accel_mag > 8.81 and accel_mag < 10.81) { // Normalize accelerometer reading
        // Expected gravity in body frame based on current attitude estimate

        const expected_accel_NED = Vec3.init(0, 0, -9.81); // World frame gravity
        const expected_accel_body = expected_accel_NED.rotate_by_quaternion(q_gyro.conjugate()).normalize();

        //     // Calculate error between measured and expected gravity
        const accel_err = raw_accel.cross(expected_accel_body);

        //     // Complementary filter gain (how much to trust accelerometer)
        const kp: f32 = 1.5; // Proportional gain for accelerometer correction

        //     // Apply correction
        const correction = accel_err.scale(kp * dt);
        const q_correction = Quaternion.init(
            correction.x() * 0.5,
            correction.y() * 0.5,
            correction.z() * 0.5,
            1.0,
        ).normalize();

        //     // Apply accelerometer correction to gyro attitude
        q_gyro = q_correction.multiply(q_gyro).normalize();
        // }
        self.attitude_estimate = q_gyro;

        std.debug.print("DT: {}\nRAW_RATES: {}\nRAW_ACCEL: {}\nACCEL_MAG: {}\nQ_GYRO: {}\n", .{
            dt,
            raw_rates,
            raw_accel,
            accel_mag,
            q_gyro,
        });

        const rate_filter_alpha: f32 = 0.3; // 30% new data, 70% old - reduces noise
        self.rate_estimate = self.rate_estimate.scale(1.0 - rate_filter_alpha).add(raw_rates.scale(rate_filter_alpha));
    }

    /// Run the control loop using the selected controller
    pub fn updateControl(self: *Self, dt: f32) [3]f32 {
        switch (self.controller) {
            .Rate => |*rate_ctrl| {
                const rate_setpoints = self.setpoints.Rate;

                // Direct rate control
                return rate_ctrl.update(
                    rate_setpoints.rates,
                    self.rate_estimate,
                    dt,
                );
            },

            .Attitude => |*att_ctrl| {
                const att_setpoints = self.setpoints.Attitude;

                // Extract current angles from quaternion
                const current_euler = self.attitude_estimate.to_euler();

                // Build desired quaternion from angle setpoints
                const q_des = Quaternion.from_euler(
                    att_setpoints.angles[1], // Desired pitch
                    current_euler[1], // Keep current yaw
                    att_setpoints.angles[0], // Desired roll
                );

                // Run attitude controller
                return att_ctrl.update(
                    q_des,
                    self.attitude_estimate,
                    self.rate_estimate,
                    att_setpoints.yaw_rate, // Pass through yaw rate
                    dt,
                );
            },
        }
    }

    /// Mix thrust and torque commands to individual motor thrusts (quad-X configuration)
    pub fn updateMotorMixer(self: *Self, torque_body: [3]f32) void {
        std.debug.print("MIXER: rates=[{d:.2}, {d:.2}, {d:.2}] → torque=[{d:.2}, {d:.2}, {d:.2}]\n\n", .{
            self.rate_estimate.x(), self.rate_estimate.y(), self.rate_estimate.z(),
            torque_body[0],         torque_body[1],         torque_body[2],
        });

        const thrust = self.setpoints.getThrust();
        const tau_x = torque_body[0]; // Roll torque (around X-axis)
        const tau_y = torque_body[1]; // Pitch torque (around Y-axis)
        const tau_z = torque_body[2]; // Yaw torque (around Z-axis)

        const L = self.params.motor_arm_length;
        const k_tau = self.params.motor_drag_ratio;

        // Quad-X motor mixer matrix
        // Motors are at 45° angles: M1(front-right), M2(front-left), M3(rear-left), M4(rear-right)
        // Motor rotation: M1(CW), M2(CCW), M3(CW), M4(CCW)
        const thrust_per_motor = thrust / 4.0;

        // For quad-X configuration:
        // - Roll torque is generated by differential thrust on diagonal motors
        // - Pitch torque is generated by differential thrust front vs rear
        // - Each motor contributes to torque based on its arm distance
        const L_eff = L * @sqrt(2.0); // Effective moment arm for roll/pitch in quad-X

        // Mixing equations:
        // Roll right (+τx) increases M1,M4 and decreases M2,M3
        // Pitch forward (+τy) increases M3,M4 and decreases M1,M2
        // Yaw right (+τz) increases CCW motors (M2,M4) and decreases CW motors (M1,M3)
        const yaw_term = tau_z / (4.0 * L * k_tau);

        self.motor_commands.motor_thrusts[0] = thrust_per_motor + tau_x / L_eff - tau_y / L_eff - yaw_term;
        self.motor_commands.motor_thrusts[1] = thrust_per_motor - tau_x / L_eff - tau_y / L_eff + yaw_term;
        self.motor_commands.motor_thrusts[2] = thrust_per_motor - tau_x / L_eff + tau_y / L_eff - yaw_term;
        self.motor_commands.motor_thrusts[3] = thrust_per_motor + tau_x / L_eff + tau_y / L_eff + yaw_term;

        self.applySaturation();
    }

    /// Apply motor saturation limits and redistribute if needed
    fn applySaturation(self: *Self) void {
        const max_motor_thrust = self.params.motor_max_thrust;

        // Calculate total thrust and dynamic minimum
        const total_thrust = self.motor_commands.motor_thrusts[0] +
            self.motor_commands.motor_thrusts[1] +
            self.motor_commands.motor_thrusts[2] +
            self.motor_commands.motor_thrusts[3];
        const avg_thrust = total_thrust / 4.0;

        // Dynamic minimum: 10% of average thrust, but never below 0.1N
        const min_motor_thrust = @max(0.1, avg_thrust * 0.1);

        // Clamp all motors to [min_thrust, max_thrust]
        for (&self.motor_commands.motor_thrusts) |*thrust| {
            thrust.* = std.math.clamp(thrust.*, min_motor_thrust, max_motor_thrust);
        }

        // TODO: Add more sophisticated redistribution if any motor saturates
        // For now, simple clamping is sufficient

        // Safety check: detect death spiral conditions
        self.checkForDeathSpiral();
    }

    /// Safety monitor to detect unrecoverable control situations
    fn checkForDeathSpiral(self: *Self) void {
        var saturated_motors: u32 = 0;
        var min_saturated: u32 = 0;
        var max_saturated: u32 = 0;

        // Calculate dynamic minimum (same logic as applySaturation)
        const total_thrust = self.motor_commands.motor_thrusts[0] +
            self.motor_commands.motor_thrusts[1] +
            self.motor_commands.motor_thrusts[2] +
            self.motor_commands.motor_thrusts[3];
        const avg_thrust = total_thrust / 4.0;
        const min_motor_thrust = @max(0.1, avg_thrust * 0.1);

        // Count saturated motors
        for (self.motor_commands.motor_thrusts) |thrust| {
            if (thrust >= self.params.motor_max_thrust - 0.01) {
                max_saturated += 1;
            }
            if (thrust <= min_motor_thrust + 0.01) { // Near dynamic minimum
                min_saturated += 1;
            }
            if (thrust >= self.params.motor_max_thrust - 0.01 or thrust <= min_motor_thrust + 0.01) {
                saturated_motors += 1;
            }
        }

        // Check attitude errors (convert to degrees for easier reading)
        const euler = self.attitude_estimate.to_euler();
        const roll_deg = std.math.radiansToDegrees(euler[2]);
        const pitch_deg = std.math.radiansToDegrees(euler[0]);
        const yaw_deg = std.math.radiansToDegrees(euler[1]);

        // Death spiral conditions
        const extreme_attitude = @abs(roll_deg) > 30.0 or @abs(pitch_deg) > 30.0;
        const severe_saturation = saturated_motors >= 3;
        const opposite_saturation = min_saturated >= 1 and max_saturated >= 1;

        if (extreme_attitude and (severe_saturation or opposite_saturation)) {
            std.debug.print("🚨 DEATH SPIRAL DETECTED! 🚨\n", .{});
            std.debug.print("Attitude: roll={d:.1}° pitch={d:.1}° yaw={d:.1}°\n", .{ roll_deg, pitch_deg, yaw_deg });
            std.debug.print("Motors: [{d:.2}, {d:.2}, {d:.2}, {d:.2}]N\n", .{
                self.motor_commands.motor_thrusts[0],
                self.motor_commands.motor_thrusts[1],
                self.motor_commands.motor_thrusts[2],
                self.motor_commands.motor_thrusts[3],
            });
            std.debug.print("Saturated: {d}/4 motors (max: {d}, min: {d})\n", .{ saturated_motors, max_saturated, min_saturated });
            std.debug.print("Control authority lost - system unrecoverable\n", .{});

            @panic("Flight controller death spiral detected - check logs above");
        }
    }

    /// Apply motor lag filter (1st order with time constant)
    pub fn updateMotorLag(self: *Self, dt: f32) void {
        const alpha = dt / (self.params.motor_time_constant + dt);

        for (0..4) |i| {
            self.motor_filtered[i] = self.motor_filtered[i] + alpha * (self.motor_commands.motor_thrusts[i] - self.motor_filtered[i]);
        }
    }

    /// Convert motor thrusts to world forces and body torques for physics engine
    pub fn calculatePhysicsForces(self: *Self, attitude_ned: Quaternion) void {
        // Motor positions in NED body frame
        const L = self.params.motor_arm_length;
        const L_diag = L / @sqrt(2.0);
        const motor_positions = [4]Vec3{
            Vec3.init(L_diag, L_diag, 0), // M1: front-right
            Vec3.init(L_diag, -L_diag, 0), // M2: front-left
            Vec3.init(-L_diag, -L_diag, 0), // M3: rear-left
            Vec3.init(-L_diag, L_diag, 0), // M4: rear-right
        };

        // Calculate total thrust
        var total_thrust: f32 = 0;
        for (self.motor_filtered) |motor_thrust| {
            total_thrust += motor_thrust;
        }

        // Calculate torques in NED body frame
        const thrust_dir_ned_body = Vec3.init(0, 0, -1); // Upward in NED body
        var total_torque_ned = Vec3.init(0, 0, 0);

        for (0..4) |i| {
            const motor_force = thrust_dir_ned_body.scale(self.motor_filtered[i]);
            const torque_from_arm = motor_positions[i].cross(motor_force);
            total_torque_ned = total_torque_ned.add(torque_from_arm);
        }

        // Add prop drag torques
        const prop_drag_signs = [4]f32{ -1, 1, -1, 1 };
        for (0..4) |i| {
            const drag_torque = self.motor_filtered[i] * self.params.motor_drag_ratio * prop_drag_signs[i];
            total_torque_ned = total_torque_ned.add(Vec3.init(0, 0, drag_torque));
        }

        // Transform to OpenGL for physics
        if (self.coord_adapter) |adapter| {
            // Convert attitude from NED to OpenGL
            const attitude_opengl = adapter.q_NF.multiply(attitude_ned);

            // Rotate to world frame using OpenGL attitude
            const thrust_dir_world = thrust_dir_ned_body.rotate_by_quaternion(attitude_ned).normalize();
            const thrust_world_opengl = thrust_dir_world.rotate_by_quaternion(attitude_opengl);

            self.motor_commands.total_thrust_world = thrust_world_opengl.scale(total_thrust);
            self.motor_commands.total_torque_body = total_torque_ned.rotate_by_quaternion(adapter.q_NF);
        } else {
            // No adapter - already in correct frame
            const thrust_world = thrust_dir_ned_body.rotate_by_quaternion(attitude_ned);
            self.motor_commands.total_thrust_world = thrust_world.scale(total_thrust);
            self.motor_commands.total_torque_body = total_torque_ned;
        }
    }

    /// Reset flight controller state to initial values
    pub fn reset(self: *Self) void {
        // Reset controller
        self.controller.reset();

        // Reset setpoints
        self.setpoints = ControlSetpoints.init(self.controller_type);

        // Reset motor commands
        self.motor_commands = MotorCommands{};

        // Reset state estimates
        self.attitude_estimate = Quaternion.identity();
        self.rate_estimate = Vec3.init(0, 0, 0);

        // Reset motor lag filtering state
        self.motor_filtered = [4]f32{ 0, 0, 0, 0 };

        // Reset timing
        self.last_update_us = 0;
    }
};

/// Flight controller system that manages all flight controllers and runs the 400Hz control thread
pub const FlightControllerSystem = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    flight_controller_components: *SparseSet(FlightControllerComponent),

    // Control thread management
    control_thread: ?std.Thread = null,
    should_shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    // References to other systems
    imu_system: ?*IMUSensor.IMUSystem = null,
    physics_thread: ?*PhysicsThread.ThreadedPhysicsSystem = null,

    // Timing
    target_rate_hz: f64 = 400.0, // 400 Hz control loop
    last_timestamp_us: u64 = 0, // For calculating actual dt

    pub fn init(allocator: std.mem.Allocator, flight_controller_components: *SparseSet(FlightControllerComponent)) Self {
        return Self{
            .allocator = allocator,
            .flight_controller_components = flight_controller_components,
        };
    }

    pub fn deinit(self: *Self) void {
        self.stopControlThread();
    }

    /// Set references to other systems
    pub fn setIMUSystem(self: *Self, imu_system: *IMUSensor.IMUSystem) void {
        self.imu_system = imu_system;
    }

    pub fn setPhysicsThread(self: *Self, physics_thread: *PhysicsThread.ThreadedPhysicsSystem) void {
        self.physics_thread = physics_thread;
    }

    /// Start the 400Hz control thread
    pub fn startControlThread(self: *Self) !void {
        if (self.control_thread != null) return; // Already running

        self.should_shutdown.store(false, .release);
        self.control_thread = try std.Thread.spawn(.{}, controlThreadMain, .{self});

        std.debug.print("Flight controller thread started at {d} Hz\n", .{self.target_rate_hz});
    }

    /// Stop the control thread
    pub fn stopControlThread(self: *Self) void {
        if (self.control_thread) |thread| {
            self.should_shutdown.store(true, .release);
            thread.join();
            self.control_thread = null;
            std.debug.print("Flight controller thread stopped\n", .{});
        }
    }

    /// Flight controller thread main loop - runs at 400Hz
    fn controlThreadMain(self: *Self) void {
        std.debug.print("Flight controller thread started\n", .{});

        var timer = std.time.Timer.start() catch {
            std.debug.print("Failed to start flight controller timer\n", .{});
            return;
        };

        const target_frame_time_ns = @as(u64, @intFromFloat(std.time.ns_per_s / self.target_rate_hz));
        var frame_count: u64 = 0;

        while (!self.should_shutdown.load(.acquire)) {
            const frame_start = timer.read();
            const timestamp_us = @as(u64, @intCast(std.time.microTimestamp()));

            const physics = self.physics_thread orelse continue;
            if (physics.isPhysicsPaused()) continue;

            // Process all flight controllers
            self.updateAllControllers(timestamp_us);

            frame_count += 1;

            // Debug output every 2 seconds
            if (frame_count % (2 * @as(u64, @intFromFloat(self.target_rate_hz))) == 0) {
                std.debug.print("Flight controller thread: {d:.1} Hz, processed {d} controllers\n", .{
                    @as(f64, @floatFromInt(frame_count)) / (@as(f64, @floatFromInt(timer.read())) / std.time.ns_per_s),
                    self.flight_controller_components.dense.items.len,
                });
            }

            // Sleep to maintain target rate
            const frame_time = timer.read() - frame_start;
            if (frame_time < target_frame_time_ns) {
                const sleep_time = target_frame_time_ns - frame_time;
                std.time.sleep(sleep_time);
            }
        }

        std.debug.print("Flight controller thread shutting down after {d} frames\n", .{frame_count});
    }

    /// Update all flight controllers with latest IMU data
    fn updateAllControllers(self: *Self, timestamp_us: u64) void {
        // Calculate actual dt from previous frame (convert μs to seconds)
        const dt: f32 = if (self.last_timestamp_us > 0)
            @as(f32, @floatFromInt(timestamp_us - self.last_timestamp_us)) / 1_000_000.0 // μs to seconds
        else
            1.0 / @as(f32, @floatCast(self.target_rate_hz)); // Fallback to target dt on first frame
        self.last_timestamp_us = timestamp_us;
        var controller_iter = self.flight_controller_components.iterator();
        while (controller_iter.next()) |entry| {
            const controller = entry.component;
            const entity_id = entry.entity_id;

            if (!controller.armed) continue;

            if (self.getLatestIMUSample(entity_id)) |imu_sample| {
                controller.updateStateEstimate(imu_sample, dt);

                const torque_body = controller.updateControl(dt);

                controller.updateMotorMixer(torque_body);
                controller.updateMotorLag(dt);
                controller.calculatePhysicsForces(controller.attitude_estimate);

                // if (@mod(timestamp_us / 10000, 100) == 0) { // Every ~100ms
                const euler = controller.attitude_estimate.to_euler();
                std.debug.print("ATTITUDE: pitch={d:.1}°, yaw={d:.1}°, roll={d:.1}° | rates=[{d:.2}, {d:.2}, {d:.2}]\n", .{
                    euler[0] * 180.0 / std.math.pi, // Your to_euler returns [pitch, yaw, roll]
                    euler[1] * 180.0 / std.math.pi,
                    euler[2] * 180.0 / std.math.pi,
                    controller.rate_estimate.x(),
                    controller.rate_estimate.y(),
                    controller.rate_estimate.z(),
                });
                std.debug.print("SETPOINTS: {}\n", .{controller.setpoints});
                std.debug.print("MOTORS[{d}]: thrust=[{d:.2}, {d:.2}, {d:.2}, {d:.2}]N total_force=[{d:.2}, {d:.2}, {d:.2}]N\n", .{
                    entity_id.id,
                    controller.motor_filtered[0],
                    controller.motor_filtered[1],
                    controller.motor_filtered[2],
                    controller.motor_filtered[3],
                    controller.motor_commands.total_thrust_world.x(),
                    controller.motor_commands.total_thrust_world.y(),
                    controller.motor_commands.total_thrust_world.z(),
                });
                // }
                std.debug.print("\n\n{s}\n\n", .{"=" ** 80});
                self.sendPhysicsCommands(entity_id, controller, dt);
            }
        }
    }

    /// Get the latest IMU sample for a given entity (from IMU system)
    fn getLatestIMUSample(self: *Self, entity_id: Core.EntityID) ?IMUSample {
        if (self.imu_system == null) return null;

        var imu_iter = self.imu_system.?.imu_components.iterator();
        while (imu_iter.next()) |entry| {
            if (entry.entity_id.id == entity_id.id) {
                return entry.component.drainToLatest();
            }
        }

        return null;
    }

    /// Send motor commands to physics thread as forces and torques
    fn sendPhysicsCommands(self: *Self, entity_id: Core.EntityID, flight_controller: *FlightControllerComponent, dt: f32) void {
        const physics = self.physics_thread orelse return;

        const motor_commands = flight_controller.motor_commands;

        const force_command = PhysicsThread.PhysicsCommand{
            .ApplyForce = .{
                .entity_id = entity_id,
                .force = motor_commands.total_thrust_world.data,
                .dt = dt, // Time delta for this force application
            },
        };

        // Get attitude in engine for torque transformation
        const attitude_engine = if (flight_controller.coord_adapter) |adapter|
            adapter.q_NF.multiply(flight_controller.attitude_estimate) // NED → Engine
        else
            flight_controller.attitude_estimate;

        // Transform body torque to world
        const world_torque = flight_controller.motor_commands.total_torque_body
            .rotate_by_quaternion(attitude_engine);

        const torque_command = PhysicsThread.PhysicsCommand{
            .ApplyTorque = .{
                .entity_id = entity_id,
                .torque = world_torque,
                .dt = dt, // Time delta for this torque application
            },
        };

        if (!physics.sendCommand(force_command)) {
            std.debug.print("Failed to send thrust command for entity {d}\n", .{entity_id.id});
        }

        if (!physics.sendCommand(torque_command)) {
            std.debug.print("Failed to send torque command for entity {d}\n", .{entity_id.id});
        }
    }
};

// Drone input controller for handling user input to the drone
pub const DroneInputController = struct {
    const ECSManager = @import("../ECSManager.zig");
    pub fn createComponent() Controller.ControllerComponent {
        var controller = Controller.ControllerComponent.init(2, "Drone", .Entity);

        // 6 - Toggle arm/disarm
        controller.addBinding(.{
            .key = .@"6",
            .handler = handleArmToggle,
            .context = null,
        }) catch unreachable;

        // WASD - Throttle and yaw control
        controller.addBinding(.{
            .key = .W,
            .handler = handleThrottleUp,
            .context = null,
        }) catch unreachable;

        controller.addBinding(.{
            .key = .S,
            .handler = handleThrottleDown,
            .context = null,
        }) catch unreachable;

        controller.addBinding(.{
            .key = .A,
            .handler = handleYawLeft,
            .context = null,
        }) catch unreachable;

        controller.addBinding(.{
            .key = .D,
            .handler = handleYawRight,
            .context = null,
        }) catch unreachable;

        // Mouse handler for pitch/roll control
        controller.setMouseHandler(handleMouseControl, null);

        return controller;
    }

    fn handleArmToggle(event: *Controller.InputEvent, context: ?*anyopaque) void {
        if (event.action != .Press) return;

        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        // Get flight controller component for this entity
        if (ecs.flight_controller_components.get(selected_entity)) |flight_controller| {
            flight_controller.armed = !flight_controller.armed;
            std.debug.print("Drone {}: {s}\n", .{ selected_entity.id, if (flight_controller.armed) "ARMED" else "DISARMED" });
        }
        event.consume();
    }

    fn handleThrottleUp(event: *Controller.InputEvent, context: ?*anyopaque) void {
        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        sendFlightInput(ecs, selected_entity, .throttle_up, event.action != .Release, event.dt);
        event.consume();
    }

    fn handleThrottleDown(event: *Controller.InputEvent, context: ?*anyopaque) void {
        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        sendFlightInput(ecs, selected_entity, .throttle_down, event.action != .Release, event.dt);
        event.consume();
    }

    fn handleYawLeft(event: *Controller.InputEvent, context: ?*anyopaque) void {
        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        sendFlightInput(ecs, selected_entity, .yaw_left, event.action != .Release, event.dt);
        event.consume();
    }

    fn handleYawRight(event: *Controller.InputEvent, context: ?*anyopaque) void {
        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        sendFlightInput(ecs, selected_entity, .yaw_right, event.action != .Release, event.dt);
        event.consume();
    }

    fn handleMouseControl(event: *Controller.MouseEvent, context: ?*anyopaque) void {
        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        // Only process mouse movement for flight input (not clicks)
        if (event.button != null) return;

        sendMouseInput(ecs, selected_entity, event.dx, event.dy, event.dt);
        event.consume();
    }

    fn sendFlightInput(
        ecs: *ECSManager,
        entity_id: Core.EntityID,
        input_type: enum {
            throttle_up,
            throttle_down,
            yaw_left,
            yaw_right,
        },
        pressed: bool,
        dt: f32,
    ) void {
        if (ecs.flight_controller_components.get(entity_id)) |flight_controller| {
            var input_state = InputState{};

            // Set the specific input
            switch (input_type) {
                .throttle_up => input_state.throttle_up = pressed,
                .throttle_down => input_state.throttle_down = pressed,
                .yaw_left => input_state.yaw_left = pressed,
                .yaw_right => input_state.yaw_right = pressed,
            }

            // Process input and generate setpoints
            const params = InputParams{};
            const setpoints = flight_controller.controller.processInput(input_state, params, dt);

            // Apply setpoints to flight controller
            flight_controller.setControlSetpoints(setpoints);
        }
    }

    fn sendMouseInput(ecs: *ECSManager, entity_id: Core.EntityID, dx: f32, dy: f32, dt: f32) void {
        if (ecs.flight_controller_components.get(entity_id)) |flight_controller| {
            const input_state = InputState{
                .mouse_dx = dx,
                .mouse_dy = dy,
            };

            // Process mouse input
            const params = InputParams{};
            const setpoints = flight_controller.controller.processInput(input_state, params, dt);

            // Apply setpoints to flight controller
            flight_controller.setControlSetpoints(setpoints);
        }
    }
};
