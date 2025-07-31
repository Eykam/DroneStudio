const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const PIDController = @import("../../flight/PIDController.zig");
const IMUSensor = @import("IMUSensor.zig");
const PhysicsThread = @import("PhysicsThread.zig");

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

/// Rate controller - direct angular velocity control
pub const RateController = struct {
    const Self = @This();

    // Configuration - REDUCED gains for stability
    roll_gains: [3]f32 = [3]f32{ 0.04, 0.0, 0.001 },
    pitch_gains: [3]f32 = [3]f32{ 0.04, 0.0, 0.001 },
    yaw_gains: [3]f32 = [3]f32{ 0.04, 0.0, 0.001 },
    max_roll: f32 = 10.47, // rad/s
    max_pitch: f32 = 10.47, // rad/s
    max_yaw: f32 = 5.24, // rad/s

    // PID controllers for each axis
    pid: PIDController.PosePIDController,

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
    }
};

/// Attitude controller - orientation stabilization
pub const AttitudeController = struct {
    const Self = @This();

    // Proportional gains (rad/s per rad)
    Kp_roll: f32 = 0.13,
    Kp_pitch: f32 = 0.13,
    Kp_yaw: f32 = 0.04,

    max_roll: f32 = 0.52, // rad (30 degrees)
    max_pitch: f32 = 0.52, // rad (30 degrees)
    max_yaw_rate: f32 = 5.24, // rad/s (yaw stays rate mode)

    // Inner rate controller
    rate_controller: RateController,

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
        const e = Quaternion.init(
            q_err.x() * 2.0,
            q_err.y() * 2.0,
            q_err.z() * 2.0,
            q_err.w(),
        );

        // Desired body rates from proportional attitude error
        const w_sp = Vec3.init(
            self.Kp_roll * e.x(),
            self.Kp_pitch * e.y(),
            yaw_sp + self.Kp_yaw * e.z(), // yaw rate + yaw hold
        );

        // Clamp rate set-points
        // ω_sp = ω_sp.clamp(-max_rate, max_rate);

        // Inner rate loop → torque
        return self.rate_controller.update(w_sp.data, w_act, dt);
    }

    pub fn reset(self: *Self) void {
        self.rate_controller.reset();
    }
};

pub const ControllerType = enum { Rate, Attitude };
pub const Controller = union(ControllerType) {
    Rate: RateController,
    Attitude: AttitudeController,

    /// Update the controller based on its type
    pub fn update(
        self: *Controller,
        setpoints: ControlSetpoints,
        q_des: Quaternion, // desired body attitude (Flight)
        q_act: Quaternion, // current attitude               "
        w_act: Vec3, // current body rates (Flight)
        dt: f32,
    ) [3]f32 {
        switch (self.*) {
            .Rate => |*rc| {
                return rc.update(setpoints.desired_rates, w_act, dt);
            },
            .Attitude => |*ac| {
                const yaw_rate_cmd = setpoints.desired_rates[2];
                return ac.update(q_des, q_act, w_act, yaw_rate_cmd, dt);
            },
        }
    }

    pub fn reset(self: *Controller) void {
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
    motor_drag_ratio: f32 = 0.05, // kτ = c_drag / c_thrust

    // Motor dynamics
    motor_time_constant: f32 = 0.04, // seconds (40ms lag)
    motor_max_thrust: f32 = 20.0, // Newtons per motor

    // Control rate
    control_rate_hz: u32 = 400, // 400 Hz control loop
};

/// Current control setpoints from input system
pub const ControlSetpoints = struct {
    desired_rates: [3]f32 = [3]f32{ 0, 0, 0 }, // [roll, pitch, yaw] rad/s
    desired_thrust: f32 = 0.0, // Newtons total
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

    controller: Controller,
    setpoints: ControlSetpoints = ControlSetpoints{},
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
            .Rate => Controller{ .Rate = RateController.init(integrator_limit, torque_limit) },
            .Attitude => Controller{ .Attitude = AttitudeController.init(integrator_limit, torque_limit) },
        };

        return Self{
            .params = params,
            .mass = mass,
            .controller = controller,
            .coord_adapter = coord_adapter,
        };
    }

    pub fn attach(self: *Self, ecs: anytype, eid: Core.EntityID) !void {
        self.entity_id = eid;
        try ecs.flight_controller_components.add(eid, self.*);
    }

    /// Update control setpoints from input system
    pub fn setControlSetpoints(self: *Self, setpoints: ControlSetpoints) void {
        self.setpoints = setpoints;
    }

    /// Process IMU sample and update state estimate (complementary filter)
    pub fn updateStateEstimate(self: *Self, imu_sample: IMUSample, dt: f32) void {
        // Extract gyro rates and apply low-pass filter for rate estimate
        const raw_rates = Vec3.init(imu_sample.gyro[0], imu_sample.gyro[1], imu_sample.gyro[2]);
        const rate_filter_alpha: f32 = 0.3; // 30% new data, 70% old - reduces noise
        self.rate_estimate = self.rate_estimate.scale(1.0 - rate_filter_alpha).add(raw_rates.scale(rate_filter_alpha));

        // === COMPLEMENTARY FILTER FOR ATTITUDE ===
        // Integrate gyroscope for short-term attitude (high frequency)
        const omega = raw_rates; // Use raw rates for integration (less lag)
        const theta = omega.scale(dt); // Small angle approximation
        const dq = Quaternion.init(theta.x() * 0.5, theta.y() * 0.5, theta.z() * 0.5, 1.0).normalize();

        var attitude_from_gyro = self.attitude_estimate.multiply(dq).normalize();

        // Get attitude from accelerometer (low frequency)
        const accel_body = Vec3.init(imu_sample.accel[0], imu_sample.accel[1], imu_sample.accel[2]);
        const accel_mag = accel_body.length();

        // Only use accelerometer if it's measuring mostly gravity (not accelerating)
        if (@abs(accel_mag - 9.81) < 1.0) { // Within 1 m/s² of gravity
            // Normalize accelerometer reading
            const accel_norm = accel_body.normalize();

            // Expected gravity in body frame based on current attitude estimate
            const gravity_world = Vec3.init(0, -9.81, 0); // World frame gravity
            const expected_gravity_body = gravity_world.rotate_by_quaternion(attitude_from_gyro.conjugate()).normalize();

            // Calculate error between measured and expected gravity
            const gravity_error = accel_norm.cross(expected_gravity_body);

            // Complementary filter gain (how much to trust accelerometer)
            const kp: f32 = 0.1; // Proportional gain for accelerometer correction

            // Apply correction
            const correction = gravity_error.scale(kp * dt);
            const correction_quat = Quaternion.init(correction.x() * 0.5, correction.y() * 0.5, correction.z() * 0.5, 1.0).normalize();

            // Apply accelerometer correction to gyro attitude
            attitude_from_gyro = correction_quat.multiply(attitude_from_gyro).normalize();
        }

        // Step 3: Update final attitude estimate
        self.attitude_estimate = attitude_from_gyro;
    }

    /// Run the control loop using the selected controller
    pub fn updateControl(self: *Self, dt: f32) [3]f32 {

        // No coordinate transformation - assume engine and flight coordinates are the same
        if (self.coord_adapter == null) {
            return self.controller.update(
                self.setpoints,
                Quaternion.identity(), // still [desired_rates, thrust]
                Quaternion.identity(), // attitude not needed in rate mode
                self.rate_estimate,
                dt,
            );
        }

        const adapter = self.coord_adapter.?;
        const pose_f = adapter.engineToFlight(self.attitude_estimate, self.rate_estimate);

        // Desired roll / pitch come from input as angles (rad).
        // Build a quaternion set-point with current yaw + commanded roll/pitch.
        const roll_sp = self.setpoints.desired_rates[0];
        const pitch_sp = self.setpoints.desired_rates[1];

        // Current yaw (about Down axis) from quaternion
        const yaw_now = std.math.atan2(
            2.0 * (pose_f.q_NB.w() * pose_f.q_NB.y() + pose_f.q_NB.x() * pose_f.q_NB.z()),
            1.0 - 2.0 * (pose_f.q_NB.y() * pose_f.q_NB.y() + pose_f.q_NB.z() * pose_f.q_NB.z()),
        );

        const q_des = Quaternion.from_euler( // X=pitch, Y=yaw, Z=roll
            pitch_sp, yaw_now, // hold current yaw
            roll_sp);

        // Run quaternion attitude loop
        const torque_f = switch (self.controller) {
            .Rate => unreachable,
            .Attitude => self.controller.update(
                self.setpoints,
                q_des, // desired attitude (Flight)
                pose_f.q_NB, // actual   attitude (Flight)
                pose_f.rates_flight, // current  ω_B      (Flight)
                dt,
            ),
        };

        return torque_f;
    }

    /// Mix thrust and torque commands to individual motor thrusts (quad-X configuration)
    pub fn updateMotorMixer(self: *Self, torque_body: [3]f32) void {
        const thrust = self.setpoints.desired_thrust;
        const tau_x = torque_body[0]; // Roll torque (around X-axis)
        const tau_y = torque_body[1]; // Pitch torque (around Y-axis)
        const tau_z = torque_body[2]; // Yaw torque (around Z-axis)

        const L = self.params.motor_arm_length;
        const k_tau = self.params.motor_drag_ratio;

        // Quad-X motor mixer matrix for NED coordinate system
        // Motors: M1(front-right), M2(front-left), M3(rear-left), M4(rear-right)
        // Motor positions: M1(+L,+L), M2(+L,-L), M3(-L,-L), M4(-L,+L)
        // Motor rotation: M1(CW), M2(CCW), M3(CW), M4(CCW)
        const thrust_per_motor = thrust / 4.0;

        // Mixing equations for NED frame:
        // Roll right (+τx) increases M1,M4 and decreases M2,M3
        // Pitch forward (+τy) increases M1,M2 and decreases M3,M4
        // Yaw right (+τz) increases CCW motors (M2,M4) and decreases CW motors (M1,M3)
        self.motor_commands.motor_thrusts[0] = thrust_per_motor + tau_x / (2.0 * L) + tau_y / (2.0 * L) - tau_z / (4.0 * k_tau); // M1 (front-right, CW)
        self.motor_commands.motor_thrusts[1] = thrust_per_motor - tau_x / (2.0 * L) + tau_y / (2.0 * L) + tau_z / (4.0 * k_tau); // M2 (front-left, CCW)
        self.motor_commands.motor_thrusts[2] = thrust_per_motor - tau_x / (2.0 * L) - tau_y / (2.0 * L) - tau_z / (4.0 * k_tau); // M3 (rear-left, CW)
        self.motor_commands.motor_thrusts[3] = thrust_per_motor + tau_x / (2.0 * L) - tau_y / (2.0 * L) + tau_z / (4.0 * k_tau); // M4 (rear-right, CCW)

        self.applySaturation();
    }

    /// Apply motor saturation limits and redistribute if needed
    fn applySaturation(self: *Self) void {
        const max_motor_thrust = self.params.motor_max_thrust;
        const min_motor_thrust: f32 = 0.1; // Minimum thrust to maintain controllability

        // Clamp all motors to [min_thrust, max_thrust]
        for (&self.motor_commands.motor_thrusts) |*thrust| {
            thrust.* = std.math.clamp(thrust.*, min_motor_thrust, max_motor_thrust);
        }

        // TODO: Add more sophisticated redistribution if any motor saturates
        // For now, simple clamping is sufficient
    }

    /// Apply motor lag filter (1st order with time constant)
    pub fn updateMotorLag(self: *Self, dt: f32) void {
        const alpha = dt / (self.params.motor_time_constant + dt);

        for (0..4) |i| {
            self.motor_filtered[i] = self.motor_filtered[i] + alpha * (self.motor_commands.motor_thrusts[i] - self.motor_filtered[i]);
        }
    }

    /// Convert motor thrusts to world forces and body torques for physics engine
    pub fn calculatePhysicsForces(self: *Self, attitude: Quaternion) void {
        // Motor positions in body frame (quad-X configuration)
        const L = self.params.motor_arm_length;
        const motor_positions = [4]Vec3{
            Vec3.init(L, L, 0), // M1: front-right (+X,+Y)
            Vec3.init(L, -L, 0), // M2: front-left (+X,-Y)
            Vec3.init(-L, -L, 0), // M3: rear-left (-X,-Y)
            Vec3.init(-L, L, 0), // M4: rear-right (-X,+Y)
        };

        // Each motor produces thrust in +Z body direction
        const thrust_direction_body = Vec3.init(0, 0, -1);
        const thrust_direction_world = thrust_direction_body.rotate_by_quaternion(attitude);

        // Calculate total thrust in world frame
        var total_thrust: f32 = 0;
        for (self.motor_filtered) |motor_thrust| {
            total_thrust += motor_thrust;
        }
        self.motor_commands.total_thrust_world = thrust_direction_world.scale(total_thrust);

        // Calculate total torque in body frame (from motor arm moments)
        var total_torque_body = Vec3.init(0, 0, 0);

        for (0..4) |i| {
            const motor_pos = motor_positions[i];
            const motor_force_body = thrust_direction_body.scale(self.motor_filtered[i]);
            const torque_from_arm = motor_pos.cross(motor_force_body);
            total_torque_body = total_torque_body.add(torque_from_arm);
        }

        // Add propeller drag torques (simplified) - FIXED to match motor rotation
        const prop_drag_signs = [4]f32{ -1, 1, -1, 1 }; // M1(CW), M2(CCW), M3(CW), M4(CCW)
        for (0..4) |i| {
            const drag_torque = self.motor_filtered[i] * self.params.motor_drag_ratio * prop_drag_signs[i];
            total_torque_body = total_torque_body.add(Vec3.init(0, 0, drag_torque));
        }

        self.motor_commands.total_torque_body = total_torque_body;
    }

    /// Reset flight controller state to initial values
    pub fn reset(self: *Self) void {
        // Reset controller
        self.controller.reset();

        // Reset setpoints
        self.setpoints = ControlSetpoints{};

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
        const dt: f32 = 1.0 / @as(f32, @floatCast(self.target_rate_hz)); // Fixed timestep

        var controller_iter = self.flight_controller_components.iterator();
        while (controller_iter.next()) |entry| {
            const controller = entry.component;
            const entity_id = entry.entity_id;

            if (self.getLatestIMUSample(entity_id)) |imu_sample| {
                controller.updateStateEstimate(imu_sample, dt);

                const torque_body = controller.updateControl(dt);

                if (@mod(timestamp_us / 10000, 100) == 0) { // Every ~100ms
                    std.debug.print("CTRL[{d}]: setpoint=[{d:.3}, {d:.3}, {d:.3}] actual=[{d:.3}, {d:.3}, {d:.3}] torque=[{d:.3}, {d:.3}, {d:.3}]\n", .{
                        entity_id.id,
                        controller.setpoints.desired_rates[0],
                        controller.setpoints.desired_rates[1],
                        controller.setpoints.desired_rates[2],
                        controller.rate_estimate.x(),
                        controller.rate_estimate.y(),
                        controller.rate_estimate.z(),
                        torque_body[0],
                        torque_body[1],
                        torque_body[2],
                    });
                }

                controller.updateMotorMixer(torque_body);
                controller.updateMotorLag(dt);
                controller.calculatePhysicsForces(controller.attitude_estimate);

                if (@mod(timestamp_us / 10000, 100) == 0) { // Every ~100ms
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
                }

                self.sendPhysicsCommands(entity_id, controller);
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
    fn sendPhysicsCommands(self: *Self, entity_id: Core.EntityID, flight_controller: *FlightControllerComponent) void {
        if (self.physics_thread) |physics| {
            const motor_commands = flight_controller.motor_commands;
            const coord_adapter = flight_controller.coord_adapter;

            const engine_results = if (coord_adapter) |adapter|
                adapter.flightToEngine(motor_commands.total_torque_body, motor_commands.total_thrust_world)
            else
                CoordinateAdapter.FlightToEngineResult{
                    .torque_engine = motor_commands.total_torque_body,
                    .force_engine = motor_commands.total_thrust_world,
                };

            const force_command = PhysicsThread.PhysicsCommand{
                .ApplyForce = .{
                    .entity_id = entity_id,
                    .force = engine_results.force_engine.data,
                },
            };

            if (!physics.sendCommand(force_command)) {
                std.debug.print("Failed to send thrust command for entity {d}\n", .{entity_id.id});
            }

            const world_torque = engine_results.torque_engine.rotate_by_quaternion(flight_controller.attitude_estimate);
            const torque_command = PhysicsThread.PhysicsCommand{
                .ApplyTorque = .{
                    .entity_id = entity_id,
                    .torque = world_torque,
                },
            };

            if (!physics.sendCommand(torque_command)) {
                std.debug.print("Failed to send torque command for entity {d}\n", .{entity_id.id});
            }
        }
    }
};
