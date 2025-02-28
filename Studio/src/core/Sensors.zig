const std = @import("std");
const Math = @import("Math.zig");
const Node = @import("Node.zig");
const Drone = @import("Drone.zig");

const DroneConfig = Drone.DroneConfig;
const Vec3 = Math.Vec3;
const time = std.time;
const Instant = time.Instant;

pub const SensorState = struct {
    const Self = @This();

    const ACCEL_GYRO_SAMPLES: u32 = 60000;
    const MAGNETOMETER_SAMPLES: u32 = 6000;

    const CalibrationType = enum {
        None,
        Magnetometer,
        AccelGyro,
    };

    filter: ?*MadgwickFilter = null,
    config: *DroneConfig,
    mag_updated: bool = false,
    previous_mag_magnitude: f32 = 0,

    position: Vec3,
    velocity: Vec3,

    samples: u32 = 0,
    sample_count: u32 = 0,
    calibrating: bool = false,
    calibration_type: CalibrationType = .None,

    declination: f32 = -12.46,

    accel_offset: Vec3 = Vec3.zero(),
    gyro_offset: Vec3 = Vec3.zero(),

    mag_hard_iron: Vec3 = Vec3.zero(),
    mag_soft_iron: Vec3 = .{ .x = 1, .y = 1, .z = 1 },

    mag_min: Vec3 = .{ .x = 99999.0, .y = 99999.0, .z = 99999.0 },
    mag_max: Vec3 = .{ .x = -99999.0, .y = -99999.0, .z = -99999.0 },

    pub fn init(allocator: std.mem.Allocator, config: *DroneConfig) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .config = config,
            .position = Vec3.zero(),
            .velocity = Vec3.zero(),
            .mag_hard_iron = config.sensor_calibration.mag_hard_iron,
            .mag_soft_iron = config.sensor_calibration.mag_soft_iron,
            .accel_offset = config.sensor_calibration.accel_offset,
            .gyro_offset = config.sensor_calibration.gyro_offset,
        };
        return self;
    }

    pub fn apply_mag_calibration(self: *Self, raw_mag: Vec3) Vec3 {
        // First subtract hard iron offset
        const hard_iron_corrected = Vec3{
            .x = raw_mag.x - self.mag_hard_iron.x,
            .y = raw_mag.y - self.mag_hard_iron.y,
            .z = raw_mag.z - self.mag_hard_iron.z,
        };

        // Then apply soft iron correction
        return Vec3{
            .x = hard_iron_corrected.x * self.mag_soft_iron.x,
            .y = hard_iron_corrected.y * self.mag_soft_iron.y,
            .z = hard_iron_corrected.z * self.mag_soft_iron.z,
        };
    }

    pub fn apply_accel_calibration(self: *Self, raw_accel: Vec3) Vec3 {
        return raw_accel.sub(self.accel_offset);
    }

    pub fn apply_gyro_calibration(self: *Self, raw_gyro: Vec3) Vec3 {
        return raw_gyro.sub(self.gyro_offset);
    }

    pub fn start_calibration(self: *Self, calibration_type: CalibrationType) void {
        self.calibration_type = calibration_type;
        self.samples = switch (calibration_type) {
            .AccelGyro => ACCEL_GYRO_SAMPLES,
            .Magnetometer => MAGNETOMETER_SAMPLES,
            .None => 0,
        };

        switch (calibration_type) {
            .AccelGyro => {
                self.calibrating = true;
                self.sample_count = 0;
                self.accel_offset = Vec3.zero();
                self.gyro_offset = Vec3.zero();
            },
            .Magnetometer => {
                self.calibrating = true;
                self.sample_count = 0;
                self.mag_min = .{ .x = 99999.0, .y = 99999.0, .z = 99999.0 };
                self.mag_max = .{ .x = -99999.0, .y = -99999.0, .z = -99999.0 };
            },
            .None => {},
        }
    }

    pub fn update_calibration(self: *Self, pose: Pose) void {
        if (self.sample_count >= self.samples) {
            self.finalize_calibration();
        }

        switch (self.calibration_type) {
            .AccelGyro => {
                const accel = pose.accel;
                const gyro = pose.gyro;

                self.accel_offset = self.accel_offset.add(Vec3{
                    .x = accel.x,
                    .y = accel.y,
                    .z = 1.0 + accel.z,
                });
                self.gyro_offset = self.gyro_offset.add(gyro);

                self.sample_count += 1;
            },
            .Magnetometer => {
                if (!self.mag_updated) return;

                const mag = pose.mag;
                self.mag_min = .{
                    .x = if (mag.x < self.mag_min.x) mag.x else self.mag_min.x,
                    .y = if (mag.y < self.mag_min.y) mag.y else self.mag_min.y,
                    .z = if (mag.z < self.mag_min.z) mag.z else self.mag_min.z,
                };
                self.mag_max = .{
                    .x = if (mag.x > self.mag_max.x) mag.x else self.mag_max.x,
                    .y = if (mag.y > self.mag_max.y) mag.y else self.mag_max.y,
                    .z = if (mag.z > self.mag_max.z) mag.z else self.mag_max.z,
                };

                self.sample_count += 1;
            },
            .None => {},
        }
    }

    pub fn finalize_calibration(self: *Self) void {
        switch (self.calibration_type) {
            .AccelGyro => {
                self.accel_offset = self.accel_offset.scale(1.0 / @as(f32, @floatFromInt(self.sample_count)));
                self.gyro_offset = self.gyro_offset.scale((1.0 / @as(f32, @floatFromInt(self.sample_count))));

                std.debug.print("Accel Offset: {d:.2}, Gyro Offset: {d:.2}", .{
                    [_]f32{
                        self.accel_offset.x,
                        self.accel_offset.y,
                        self.accel_offset.z,
                    },
                    [_]f32{
                        self.gyro_offset.x,
                        self.gyro_offset.y,
                        self.gyro_offset.z,
                    },
                });

                self.saveCalibration(self.config);
            },
            .Magnetometer => {
                const offset = self.mag_max.add(self.mag_min).scale(0.5);
                const span = self.mag_max.sub(self.mag_min).scale(0.5);

                // Hard-iron offset is the midpoint
                self.mag_hard_iron = offset;

                // Soft-iron can be approximated as the ratio to the largest axis
                // or a more sophisticated elliptical fit can be done.
                // For simplicity, let's store scale = (maxSpan / axisSpan).
                const max_axis = @max(@max(span.x, span.y), span.z);
                self.mag_soft_iron = .{
                    .x = max_axis / span.x,
                    .y = max_axis / span.y,
                    .z = max_axis / span.z,
                };

                self.saveCalibration(self.config);
            },
            .None => {},
        }

        self.calibrating = false;
        self.calibration_type = .None;

        self.config.saveToFile(null) catch |err| {
            std.debug.print("Failed to save calibration to config: {}\n", .{err});
        };
    }

    pub fn saveCalibration(self: *Self, config: *DroneConfig) void {
        config.sensor_calibration.mag_hard_iron = self.mag_hard_iron;
        config.sensor_calibration.mag_soft_iron = self.mag_soft_iron;
        config.sensor_calibration.accel_offset = self.accel_offset;
        config.sensor_calibration.gyro_offset = self.gyro_offset;
    }
};

pub const Pose = struct {
    accel: Vec3,
    gyro: Vec3,
    mag: Vec3,
    timestamp: i64,
};

pub const PoseHandler = struct {
    allocator: std.mem.Allocator,
    node: *Node,
    packet_count: u32 = 0,
    mag_count: u32 = 0,
    prev_instant: time.Instant,
    prev_timestamp: ?i64 = null,
    prev_pose: ?Pose = null,
    stale_count: u32 = 0,
    sensor_state: *SensorState,
    accel_gyro_freq: u32 = 0,
    mag_freq: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, node: *Node, config: *DroneConfig) !PoseHandler {
        return .{
            .allocator = allocator,
            .node = node,
            .prev_instant = time.Instant.now() catch unreachable,
            .sensor_state = try SensorState.init(allocator, config),
        };
    }

    pub fn parse(packet: []const u8) Pose {
        const accel = Vec3{
            .x = @bitCast(std.mem.readInt(u32, packet[0..4], .little)),
            .y = @bitCast(std.mem.readInt(u32, packet[4..8], .little)),
            .z = @bitCast(std.mem.readInt(u32, packet[8..12], .little)),
        };

        const gyro = Vec3{
            .x = Math.radians(@bitCast(std.mem.readInt(u32, packet[12..16], .little))),
            .y = Math.radians(@bitCast(std.mem.readInt(u32, packet[16..20], .little))),
            .z = Math.radians(@bitCast(std.mem.readInt(u32, packet[20..24], .little))),
        };

        const mag = Vec3{
            .x = @bitCast(std.mem.readInt(u32, packet[24..28], .little)),
            .y = @bitCast(std.mem.readInt(u32, packet[28..32], .little)),
            .z = @bitCast(std.mem.readInt(u32, packet[32..36], .little)),
        };

        const mag_sens = Vec3{
            .x = mag.y,
            .y = mag.x,
            .z = -mag.z,
        };

        const accel_ned = Vec3{
            .x = accel.z,
            .y = -accel.x,
            .z = -accel.y,
        };

        // Apply same rotation to gyroscope readings
        const gyro_ned = Vec3{
            .x = gyro.z,
            .y = -gyro.x,
            .z = -gyro.y,
        };

        // Apply rotation to magnetometer readings
        const mag_ned = Vec3{
            .x = mag_sens.z,
            .y = -mag_sens.x,
            .z = -mag_sens.y,
        };

        const timestamp: i64 = @bitCast(std.mem.readInt(i64, packet[36..44], .little));

        return Pose{
            .accel = accel_ned,
            .gyro = gyro_ned,
            .mag = mag_ned,
            .timestamp = timestamp,
        };
    }

    pub fn update(self: *PoseHandler, data: []const u8) !void {
        const pose = PoseHandler.parse(data);

        const mag = pose.mag;
        const magnitude_mag = mag.x + mag.y + mag.z;
        const mag_updated = @abs(magnitude_mag - self.sensor_state.previous_mag_magnitude) > 0.0001;

        if (mag_updated) self.mag_count += 1;

        self.sensor_state.mag_updated = mag_updated;
        self.sensor_state.previous_mag_magnitude = magnitude_mag;

        const curr_instant = try Instant.now();
        const delta_instant = Instant.since(curr_instant, self.prev_instant);
        const delta_instant_to_seconds = @as(f32, @floatFromInt(delta_instant)) / std.time.ns_per_s;

        if (delta_instant_to_seconds >= 1.0) {
            const accel_gyro_per_sec: u32 = self.packet_count / @as(u32, @intFromFloat(delta_instant_to_seconds));
            const mag_per_sec = self.mag_count / @as(u32, @intFromFloat(delta_instant_to_seconds));
            std.debug.print("\n==========\n IMU Info\n========== \nTwo seconds have passed.\nAccel & Gyro Packets / sec counted: {d}\nMag Packets / sec counted: {d}\nThroughput: {d} B/s\n\n", .{
                accel_gyro_per_sec,
                mag_per_sec,
                accel_gyro_per_sec * data.len,
            });

            self.accel_gyro_freq = accel_gyro_per_sec;
            self.mag_freq = mag_per_sec;
            self.prev_instant = curr_instant;
            self.packet_count = 0;
            self.mag_count = 0;
        }

        self.prev_pose = pose;
        self.packet_count += 1;

        if (self.sensor_state.calibrating) {
            self.sensor_state.update_calibration(pose);
            return;
        }

        if (self.prev_timestamp == null) {
            self.prev_timestamp = pose.timestamp;
        }

        const delta_time_us = @as(f32, @floatFromInt(pose.timestamp - self.prev_timestamp.?)) / std.time.us_per_s;
        if (delta_time_us < 0) {
            std.debug.print("Received stale packet, continuing...\n", .{});
            self.prev_timestamp = pose.timestamp;
            self.stale_count += 1;
            return;
        }

        self.prev_timestamp = pose.timestamp;

        const rotation = updateModelMatrix(
            self.allocator,
            pose,
            self.sensor_state.declination,
            delta_time_us,
            self.sensor_state,
        );
        self.node.setRotation(rotation);
    }
};

pub const MadgwickFilter = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    beta: f32, // algorithm gain
    beta_default: f32 = 0.03,
    beta_high: f32 = 1.0, // Higher gain for faster convergence
    beta_accel_only: f32, // gain when using only accel (no mag)
    convergence_threshold: f32 = 0.005, // Threshold to detect convergence
    convergence_counter: u32 = 0,
    previous_quaternion: Math.Quaternion = Math.Quaternion.identity(),
    q: Math.Quaternion, // current orientation estimate

    /// Create a new Madgwick filter with the specified gain and optional initial orientation.
    pub fn init(allocator: std.mem.Allocator, beta: f32, initial_orientation: ?Math.Quaternion) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .beta = self.beta_high,
            .beta_default = beta,
            .beta_accel_only = beta * 0.5,
            .q = if (initial_orientation) |init_q| init_q else Math.Quaternion.identity(),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn updateAdaptiveGain(self: *Self) void {
        // Calculate quaternion difference from previous update
        const q = self.q;

        const q_diff = @sqrt(
            std.math.pow(f32, q.w - self.previous_quaternion.w, 2) +
                std.math.pow(f32, q.x - self.previous_quaternion.x, 2) +
                std.math.pow(f32, q.y - self.previous_quaternion.y, 2) +
                std.math.pow(f32, q.z - self.previous_quaternion.z, 2),
        );

        // Update previous quaternion for next iteration
        self.previous_quaternion = q;

        if (q_diff < self.convergence_threshold) {
            self.convergence_counter += 1;
            if (self.convergence_counter > 1000) { // ~1 second at 1kHz
                self.beta = self.beta_default;
            }
        } else {
            self.convergence_counter = 0;
            self.beta = self.beta_high;
        }

        self.beta_accel_only = self.beta * 0.5;
    }

    /// Update the orientation estimate given calibrated gyro, accel, mag data and time step dt (seconds).
    /// Update the orientation estimate using only accelerometer and gyroscope data
    pub fn updateNoMag(self: *Self, gyro: Vec3, accel: Vec3, dt: f32) void {
        const q_local = self.q;
        var q0 = q_local.w;
        var q1 = q_local.x;
        var q2 = q_local.y;
        var q3 = q_local.z;

        // Local copies of raw data
        const gx = gyro.x;
        const gy = gyro.y;
        const gz = gyro.z;
        var ax = accel.x;
        var ay = accel.y;
        var az = accel.z;

        // Normalize accelerometer measurement
        const acc_norm_squared = ax * ax + ay * ay + az * az;
        if (acc_norm_squared < 0.000001) {
            // If accelerometer is zero, only integrate gyro
            const qDot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
            const qDot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
            const qDot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
            const qDot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

            q0 += qDot0 * dt;
            q1 += qDot1 * dt;
            q2 += qDot2 * dt;
            q3 += qDot3 * dt;
        } else {
            const acc_norm = @sqrt(acc_norm_squared);
            ax /= acc_norm;
            ay /= acc_norm;
            az /= acc_norm;

            // Auxiliary variables to reduce repeated calculations
            const _2q0 = 2.0 * q0;
            const _2q1 = 2.0 * q1;
            const _2q2 = 2.0 * q2;
            const _2q3 = 2.0 * q3;
            const _4q0 = 4.0 * q0;
            const _4q1 = 4.0 * q1;
            const _4q2 = 4.0 * q2;
            const _8q1 = 8.0 * q1;
            const _8q2 = 8.0 * q2;
            const q0q0 = q0 * q0;
            const q1q1 = q1 * q1;
            const q2q2 = q2 * q2;
            const q3q3 = q3 * q3;

            // Gradient descent algorithm corrective step
            // Only using accelerometer for gravity direction
            const s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
            const s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
            const s2 = 4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
            const s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay;

            const s_norm_squared = s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
            if (s_norm_squared < 0.000001) {
                std.debug.print("Gradient step too small in updateNoMag, using gyro only\n", .{});
                // If gradient is too small, just integrate gyro
                const qDot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
                const qDot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
                const qDot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
                const qDot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

                q0 += qDot0 * dt;
                q1 += qDot1 * dt;
                q2 += qDot2 * dt;
                q3 += qDot3 * dt;
            } else {
                const recip_norm = 1.0 / @sqrt(s_norm_squared);

                // Apply feedback step
                const qDot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz) - self.beta * s0 * recip_norm;
                const qDot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy) - self.beta * s1 * recip_norm;
                const qDot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx) - self.beta * s2 * recip_norm;
                const qDot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx) - self.beta * s3 * recip_norm;

                // Integrate rate of change of quaternion
                q0 += qDot0 * dt;
                q1 += qDot1 * dt;
                q2 += qDot2 * dt;
                q3 += qDot3 * dt;
            }
        }

        // Check for NaN values before normalization
        if (std.math.isNan(q0) or std.math.isNan(q1) or
            std.math.isNan(q2) or std.math.isNan(q3))
        {
            std.debug.print("NaN detected in quaternion components in updateNoMag, resetting\n", .{});
            // self.q = Math.Quaternion.identity();
            return;
        }

        // Normalize quaternion
        const q_norm_squared = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
        if (q_norm_squared < 0.000001) {
            std.debug.print("Quaternion magnitude too small in updateNoMag, resetting\n", .{});
            // self.q = Math.Quaternion.identity();
            return;
        }

        const recipNorm = 1.0 / @sqrt(q_norm_squared);

        // Only update the actual quaternion after all calculations are done
        self.q = .{
            .w = q0 * recipNorm,
            .x = q1 * recipNorm,
            .y = q2 * recipNorm,
            .z = q3 * recipNorm,
        };
    }

    /// Update the orientation estimate using gyro, accel, mag data and timestamp.
    /// Will fallback to accel+gyro only if mag is null or hasn't been updated recently.
    pub fn update(self: *Self, gyro: Vec3, accel: Vec3, mag: ?Vec3, dt: f32) void {
        defer self.updateAdaptiveGain();

        // If mag is null or hasn't been updated, use accel+gyro only
        if (mag == null) {
            self.updateNoMag(gyro, accel, dt);
            return;
        }
        const q_local = self.q;
        var q0 = q_local.w;
        var q1 = q_local.x;
        var q2 = q_local.y;
        var q3 = q_local.z;

        // Local copies of raw data
        const gx = gyro.x;
        const gy = gyro.y;
        const gz = gyro.z;
        var ax = accel.x;
        var ay = accel.y;
        var az = accel.z;
        var mx = mag.?.x;
        var my = mag.?.y;
        var mz = mag.?.z;

        // ==== 1) Normalize accelerometer ====
        const accel_norm = accel.length();
        if (accel_norm < 0.00001) {
            std.debug.print("Accelerometer magnitude too small, skipping\n", .{});
            return;
        }

        ax /= accel_norm;
        ay /= accel_norm;
        az /= accel_norm;

        // ==== 2) Normalize magnetometer ====
        const mag_norm = mag.?.length();
        if (mag_norm < 0.00001) {
            std.debug.print("Magnetometer magnitude too small, using accel+gyro only\n", .{});
            self.updateNoMag(gyro, accel, dt);
            return;
        }

        mx /= mag_norm;
        my /= mag_norm;
        mz /= mag_norm;

        // ==== 3) Auxiliary variables to reduce number of repeated operations ====
        const _2q0 = 2.0 * q0;
        const _2q1 = 2.0 * q1;
        const _2q2 = 2.0 * q2;
        const _2q3 = 2.0 * q3;

        const q0q0 = q0 * q0;
        const q1q1 = q1 * q1;
        const q2q2 = q2 * q2;
        const q3q3 = q3 * q3;

        const _2q0q1 = 2.0 * q0 * q1;
        const _2q0q2 = 2.0 * q0 * q2;
        const _2q1q1 = 2.0 * q1q1;
        const _2q1q3 = 2.0 * q1 * q3;
        const _2q2q2 = 2.0 * q2q2;
        const _2q2q3 = 2.0 * q2 * q3;

        // ==== 4) Reference direction of Earth's magnetic field ====
        const hx = mx * q0q0 - _2q0 * my * q3 + _2q0 * mz * q2 + mx * q1q1 + _2q1 * my * q2 + _2q1 * mz * q3 - mx * q2q2 - mx * q3q3;
        const hy = _2q0 * mx * q3 + my * q0q0 - _2q0 * mz * q1 + _2q1 * mx * q2 - my * q1q1 + my * q2q2 + _2q2 * mz * q3 - my * q3q3;

        // Check if hx and hy are non-zero before calculating _2bx
        const hx_hy_squared = hx * hx + hy * hy;
        if (hx_hy_squared < 0.000001) {
            std.debug.print("Horizontal magnetic field too small, using accel+gyro only\n", .{});
            self.updateNoMag(gyro, accel, dt);
            return;
        }

        const _2bx = @sqrt(hx_hy_squared);
        if (std.math.isNan(_2bx) or _2bx < 0.000001) {
            std.debug.print("Invalid _2bx value, using accel+gyro only\n", .{});
            self.updateNoMag(gyro, accel, dt);
            return;
        }

        const _2bz = -_2q0 * mx * q2 + _2q0 * my * q1 + mz * q0q0 + _2q1 * mx * q3 - mz * q1q1 + _2q2 * my * q3 - mz * q2q2 + mz * q3q3;

        // ==== 5) Gradient descent algorithm corrective step ====
        // (the 'f' vector and its Jacobian 'J')
        const f1 = _2q2q3 - _2q0q1 - ax;
        const f2 = _2q0q2 + _2q1q3 - ay;
        const f3 = 1.0 - _2q1q1 - _2q2q2 - az;
        const f4 = _2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1 * q3 - q0 * q2) - mx;
        const f5 = _2bx * (q1 * q2 - q0 * q3) + _2bz * (q0 * q1 + q2 * q3) - my;
        const f6 = _2bx * (q0 * q2 + q1 * q3) + _2bz * (0.5 - q1q1 - q2q2) - mz;

        const J_11or24 = _2q2; // J_11 neg
        const J_12or23 = _2q3;
        const J_13or22 = -_2q0;
        const J_14or21 = -_2q1;
        const J_32 = 2.0 * J_14or21;
        const J_33 = 2.0 * J_11or24;
        const J_41 = _2bz * q2;
        const J_42 = _2bz * q3;
        const J_43 = 2.0 * _2bx * q2 + _2bz * q0;
        const J_44 = 2.0 * _2bx * q3 - _2bz * q1;
        const J_51 = _2bx * q3 - _2bz * q1;
        const J_52 = _2bx * q2 + _2bz * q0;
        const J_53 = _2bx * q1 + _2bz * q3;
        const J_54 = _2bx * q0 - _2bz * q2;
        const J_61 = _2bx * q2;
        const J_62 = _2bx * q3 - 2.0 * _2bz * q1;
        const J_63 = _2bx * q0 - 2.0 * _2bz * q2;
        const J_64 = _2bx * q1;

        // Gradient for the 6 elements f1..f6:
        var grad_q0 = J_14or21 * f2 - J_11or24 * f1 + J_41 * f4 + J_51 * f5 + J_61 * f6;
        var grad_q1 = J_12or23 * f1 + J_13or22 * f2 - J_32 * f3 + J_42 * f4 + J_52 * f5 + J_62 * f6;
        var grad_q2 = J_12or23 * f2 - J_33 * f3 - J_13or22 * f1 + J_43 * f4 + J_53 * f5 + J_63 * f6;
        var grad_q3 = J_14or21 * f1 + J_11or24 * f2 + J_44 * f4 + J_54 * f5 + J_64 * f6;

        // Normalize the gradient
        const grad_norm_squared = grad_q0 * grad_q0 + grad_q1 * grad_q1 + grad_q2 * grad_q2 + grad_q3 * grad_q3;
        if (grad_norm_squared < 0.000001) {
            std.debug.print("Gradient too small, using gyro only\n", .{});
            // Only integrate gyro data if gradient is too small
            const qDot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
            const qDot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
            const qDot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
            const qDot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

            q0 += qDot0 * dt;
            q1 += qDot1 * dt;
            q2 += qDot2 * dt;
            q3 += qDot3 * dt;
        } else {
            const grad_norm = @sqrt(grad_norm_squared);
            grad_q0 /= grad_norm;
            grad_q1 /= grad_norm;
            grad_q2 /= grad_norm;
            grad_q3 /= grad_norm;

            // Compute quaternion derivative measured by gyroscopes
            const qDot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
            const qDot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
            const qDot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
            const qDot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

            // Apply feedback (gradient descent)
            q0 += (qDot0 - self.beta * grad_q0) * dt;
            q1 += (qDot1 - self.beta * grad_q1) * dt;
            q2 += (qDot2 - self.beta * grad_q2) * dt;
            q3 += (qDot3 - self.beta * grad_q3) * dt;
        }

        // Check for NaN values before normalization
        if (std.math.isNan(q0) or std.math.isNan(q1) or
            std.math.isNan(q2) or std.math.isNan(q3))
        {
            std.debug.print("NaN detected in quaternion components, resetting\n", .{});
            // self.q = Math.Quaternion.identity();
            return;
        }

        // Normalize quaternion
        const q_norm_squared = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
        if (q_norm_squared < 0.000001) {
            std.debug.print("Quaternion magnitude too small, resetting\n", .{});
            // self.q = Math.Quaternion.identity();
            return;
        }

        const recipNorm = 1.0 / @sqrt(q_norm_squared);

        // Only update the actual quaternion after all calculations are done
        self.q = .{
            .w = q0 * recipNorm,
            .x = q1 * recipNorm,
            .y = q2 * recipNorm,
            .z = q3 * recipNorm,
        };
    }
};

pub fn computeInitialOrientation(accel: Vec3, mag: ?Vec3, declination_deg: f32) Math.Quaternion {
    // Step 1: Normalize accelerometer reading
    const accel_norm = accel.length();
    if (accel_norm == 0.0) {
        // If accelerometer data is invalid, return identity quaternion
        return Math.Quaternion{ .w = 1, .x = 0, .y = 0, .z = 0 };
    }

    const ax = accel.x / accel_norm;
    const ay = accel.y / accel_norm;
    const az = accel.z / accel_norm;

    // Step 2: Calculate roll and pitch from accelerometer (gravity vector)
    // In NED frame, when the device is level, accelerometer reads [0, 0, 1]
    // Roll: rotation around X axis
    // Pitch: rotation around Y axis
    const roll: f32 = std.math.atan2(ay, az);
    const pitch: f32 = -std.math.atan2(ax, @sqrt(ay * ay + az * az));

    // Create quaternion for roll and pitch
    const cr = @cos(roll * 0.5);
    const sr = @sin(roll * 0.5);
    const cp = @cos(pitch * 0.5);
    const sp = @sin(pitch * 0.5);

    // Step 3: Calculate yaw using magnetometer if available
    var yaw: f32 = 0.0;

    if (mag) |m| {
        // Normalize magnetometer reading
        const mag_norm = m.length();
        if (mag_norm > 0.0) {
            const mx = m.x / mag_norm;
            const my = m.y / mag_norm;
            const mz = m.z / mag_norm;

            // Apply tilt compensation to get the horizontal components
            // Rotate mag readings by roll and pitch to get them in the horizontal plane
            // These equations compensate for the device's roll and pitch
            const bx = mx * @cos(pitch) + my * @sin(roll) * @sin(pitch) + mz * @cos(roll) * @sin(pitch);
            const by = my * @cos(roll) - mz * @sin(roll);

            // Calculate yaw from the horizontal components
            yaw = std.math.atan2(by, -bx);

            // Apply magnetic declination correction
            const declination_rad = Math.radians(declination_deg);
            yaw += declination_rad;

            // Normalize to -π to π
            if (yaw > std.math.pi) yaw -= 2.0 * std.math.pi;
            if (yaw < -std.math.pi) yaw += 2.0 * std.math.pi;
        }
    }

    // Create quaternion for yaw
    const cy = @cos(yaw * 0.5);
    const sy = @sin(yaw * 0.5);

    // Combine rotations (roll, pitch, then yaw)
    // This creates a quaternion representing the initial orientation
    return Math.Quaternion{ .w = cy * cp * cr + sy * sp * sr, .x = cy * cp * sr - sy * sp * cr, .y = cy * sp * cr + sy * cp * sr, .z = sy * cp * cr - cy * sp * sr };
}

pub fn updateModelMatrix(
    allocator: std.mem.Allocator,
    pose: Pose,
    declination: f32,
    delta_time: f32,
    sensor_state: *SensorState,
) Math.Quaternion {
    _ = declination;

    const calibrated_acc = sensor_state.apply_accel_calibration(pose.accel);
    const calibrated_gyro = sensor_state.apply_gyro_calibration(pose.gyro);
    const calibrated_mag: ?Vec3 =
        if (sensor_state.mag_updated)
        sensor_state.apply_mag_calibration(pose.mag)
    else
        null;

    if (sensor_state.filter == null) {
        const initQ = computeInitialOrientation(calibrated_acc, calibrated_mag, 0);
        sensor_state.filter = MadgwickFilter.init(allocator, 0.03, initQ) catch null;
    }

    const iterations = 1;
    const delta_t_iter = delta_time / iterations;

    for (0..iterations) |_| {
        sensor_state.filter.?.update(
            calibrated_gyro,
            calibrated_acc,
            calibrated_mag,
            delta_t_iter,
        );
    }

    const q = sensor_state.filter.?.q;

    const euler_gl = q.toEuler();
    const q_gl2 = Math.Quaternion.fromAxisAngle(.{ .x = -1, .y = 0, .z = 0 }, euler_gl[1] - Math.radians(180.0));
    const q_gl3 = Math.Quaternion.fromAxisAngle(.{ .x = 0, .y = -1, .z = 0 }, euler_gl[2]);
    const q_gl4 = Math.Quaternion.fromAxisAngle(.{ .x = 0, .y = 0, .z = 1 }, euler_gl[0]);

    return q_gl4.multiply(q_gl2).multiply(q_gl3);
}
