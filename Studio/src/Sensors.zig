const std = @import("std");
const Math = @import("Math.zig");
// const KalmanState = Transformations.KalmanState;
const Vec3 = Math.Vec3;
const Node = @import("Node.zig");
const time = std.time;
const Instant = time.Instant;

const DECLINATION_ANGLE: f32 = -10;

pub const SensorState = struct {
    filter: ?MadgwickFilter = undefined,
    initialized: bool = false,
    sample_count: u32 = 0,
    previous_mag: f32 = 0,
    mag_updated: bool = false,
    gyro_offset: Vec3 = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
    accel_offset: Vec3 = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
    velocity: Vec3 = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
    position: Vec3,

    pub fn init(node: *Node) SensorState {
        return SensorState{
            .position = Vec3{ .x = node.position[0], .y = node.position[1], .z = node.position[2] },
        };
    }
};

pub const Pose = struct {
    accel: Vec3,
    gyro: Vec3,
    mag: Vec3,
    timestamp: i64,
};

pub const PoseHandler = struct {
    node: *Node,
    packet_count: usize = 0,
    prev_instant: time.Instant,
    prev_timestamp: ?i64 = null,
    stale_count: usize = 0,
    sensor_state: SensorState,

    pub fn init(node: *Node) PoseHandler {
        return .{
            .node = node,
            .prev_instant = time.Instant.now() catch unreachable,
            .sensor_state = SensorState.init(node),
        };
    }

    pub fn parse(packet: []const u8) !Pose {
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

        const timestamp: i64 = @bitCast(std.mem.readInt(i64, packet[36..44], .little));

        return Pose{
            .accel = accel,
            .gyro = gyro,
            .mag = mag,
            .timestamp = timestamp,
        };
    }

    pub fn update(self: *PoseHandler, data: []const u8) !void {
        if (!self.sensor_state.initialized) {
            const pose = try PoseHandler.parse(data);

            if (self.sensor_state.sample_count < 10000) {
                self.sensor_state.accel_offset = self.sensor_state.accel_offset.add(Vec3{
                    .x = pose.accel.x,
                    .y = pose.accel.y,
                    .z = 1.0 - pose.accel.z,
                });
                self.sensor_state.gyro_offset = self.sensor_state.gyro_offset.add(pose.gyro);
                self.sensor_state.sample_count += 1;
                return;
            }

            self.sensor_state.accel_offset = self.sensor_state.accel_offset.scale(1.0 / 10000.0);
            self.sensor_state.gyro_offset = self.sensor_state.gyro_offset.scale((1.0 / 10000.0));

            std.debug.print("Accel Offset: {d:.2}, Gyro Offset: {d:.2}", .{
                [_]f32{
                    self.sensor_state.accel_offset.x,
                    self.sensor_state.accel_offset.y,
                    self.sensor_state.accel_offset.z,
                },
                [_]f32{
                    self.sensor_state.gyro_offset.x,
                    self.sensor_state.gyro_offset.y,
                    self.sensor_state.gyro_offset.z,
                },
            });

            self.sensor_state.initialized = true;
        }

        const pose = try PoseHandler.parse(data);

        if (self.prev_timestamp == null) {
            self.prev_timestamp = pose.timestamp;
        }

        const delta_time = @as(f32, @floatFromInt(pose.timestamp - self.prev_timestamp.?)) / 1e6;
        if (delta_time < 0) {
            std.debug.print("Received stale packet, continuing...\n", .{});
            self.prev_timestamp = pose.timestamp;
            self.stale_count += 1;
            return;
        }

        self.prev_timestamp = pose.timestamp;

        const curr_instant = try Instant.now();
        const delta_instant = Instant.since(curr_instant, self.prev_instant);
        const delta_instant_to_seconds = @as(f32, @floatFromInt(delta_instant)) / 1e9;

        const magnitude_mag = pose.mag.x + pose.mag.y + pose.mag.z;

        self.sensor_state.mag_updated = @abs(magnitude_mag - self.sensor_state.previous_mag) > 0.01;
        self.sensor_state.previous_mag = magnitude_mag;

        if (delta_instant_to_seconds >= 1.0) {
            const packets_per_sec = self.packet_count / @as(u32, @intFromFloat(delta_instant_to_seconds));
            std.debug.print("==========\nTwo seconds have passed.\nPackets / sec counted: {d}\nThroughput: {d} B/s\n", .{ packets_per_sec, packets_per_sec * data.len });
            // std.debug.print("{any}\n", .{pose.mag});
            self.prev_instant = curr_instant;
            self.packet_count = 0;
        }

        const accel_calibrated = pose.accel.sub(self.sensor_state.accel_offset);
        const gyro_calibrated = pose.gyro.sub(self.sensor_state.gyro_offset);

        const rotation = updateModelMatrix(accel_calibrated, gyro_calibrated, pose.mag, DECLINATION_ANGLE, delta_time, &self.sensor_state);
        self.node.setRotation(rotation);
        self.packet_count += 1;
    }
};

pub const MadgwickFilter = struct {
    const Self = @This();
    q: Math.Quaternion,
    initial_orientation: ?Math.Quaternion = null,
    err: [3]f32,
    beta: f32,
    zeta: f32,
    bx: f32,
    bz: f32,
    w_bx: f32 = 0,
    w_by: f32 = 0,
    w_bz: f32 = 0,

    pub fn init() Self {
        // These gains significantly affect stability
        // const gyroMeasError = std.math.pi * (0.001 / 180.0); // increased from 5.0
        // const gyroMeasDrift = std.math.pi * (0.0001 / 180.0); // kept same

        return Self{
            .q = Math.Quaternion.identity(),
            .beta = 0.0000020, // rad/s
            .zeta = 0.0000001,
            // .beta = std.math.sqrt(3.0 / 4.0) * gyroMeasError,
            // .zeta = std.math.sqrt(3.0 / 4.0) * gyroMeasDrift,
            .err = [_]f32{ 0.0, 0.0, 0.0 },
            .bx = 1.0,
            .bz = 0.0,
        };
    }

    pub fn update(
        self: *Self,
        gyro: Vec3,
        accel: Vec3,
        mag: Vec3,
        delta_t: f32,
    ) void {
        // Pre-compute quantities used multiple times
        var q_conj = self.q.conjugate();

        var q1 = self.q.w;
        var q2 = self.q.x;
        var q3 = self.q.y;
        var q4 = self.q.z;

        var q1q2 = q1 * q2;
        var q1q3 = q1 * q3;
        var q1q4 = q1 * q4;
        var q2q2 = q2 * q2;
        var q2q3 = q2 * q3;
        var q2q4 = q2 * q4;
        var q3q3 = q3 * q3;
        var q3q4 = q3 * q4;
        var q4q4 = q4 * q4;

        const a_norm = accel.normalize();

        const m_norm = mag.normalize();

        // Gradient decent algorithm corrective step
        const F = [6]f32{
            2.0 * (q2q4 - q1q3) - a_norm.x,
            2.0 * (q1q2 + q3q4) - a_norm.y,
            2.0 * (0.5 - q2q2 - q3q3) - a_norm.z,
            2.0 * self.bx * (0.5 - q3q3 - q4q4) + 2.0 * self.bz * (q2q4 - q1q3) - m_norm.x,
            2.0 * self.bx * (q2q3 - q1q4) + 2.0 * self.bz * (q1q2 + q3q4) - m_norm.y,
            2.0 * self.bx * (q1q3 + q2q4) + 2.0 * self.bz * (0.5 - q2q2 - q3q3) - m_norm.z,
        };

        const J_t = [6][4]f32{
            [4]f32{ -2.0 * q3, 2.0 * q4, -2.0 * q1, 2.0 * q2 },
            [4]f32{ 2.0 * q2, 2.0 * q1, 2.0 * q4, 2.0 * q3 },
            [4]f32{ 0.0, -4.0 * q2, -4.0 * q3, 0.0 },
            [4]f32{ -2.0 * self.bz * q3, 2.0 * self.bz * q4, -4.0 * self.bx * q3 - 2.0 * self.bz * q1, -4.0 * self.bx * q4 + 2.0 * self.bz * q2 },
            [4]f32{ -2.0 * self.bx * q4 + 2.0 * self.bz * q2, 2.0 * self.bx * q3 + 2.0 * self.bz * q1, 2.0 * self.bx * q2 + 2.0 * self.bz * q4, -2.0 * self.bx * q1 + 2.0 * self.bz * q3 },
            [4]f32{ 2.0 * self.bx * q3, 2.0 * self.bx * q4 - 4.0 * self.bz * q2, 2.0 * self.bx * q1 - 4.0 * self.bz * q3, 2.0 * self.bx * q2 },
        };

        var step = [4]f32{ 0, 0, 0, 0 };

        // Compute gradient (matrix multiplication)
        for (0..4) |j| {
            for (0..6) |i| {
                step[j] += J_t[i][j] * F[i];
            }
        }

        // Normalize step magnitude
        var step_vector = Vec3{
            .x = step[1],
            .y = step[2],
            .z = step[3],
        };
        step_vector = step_vector.normalize();

        var w_err = Vec3{
            .x = q_conj.x,
            .y = q_conj.y,
            .z = q_conj.z,
        };
        w_err = w_err.cross(step_vector).scale(2.0);

        // **Gyroscope Bias Correction**
        // Update gyro biases based on step and zeta
        self.w_bx += self.zeta * w_err.x * delta_t;
        self.w_by += self.zeta * w_err.y * delta_t;
        self.w_bz += self.zeta * w_err.z * delta_t;

        const gyro_corrected = Vec3{
            .x = gyro.x - self.w_bx,
            .y = gyro.y - self.w_by,
            .z = gyro.z - self.w_bz,
        };

        // Compute quaternion derivative from gyroscope data
        const q_dot_gyro = self.q.multiply(Math.Quaternion{
            .w = 0.0,
            .x = gyro_corrected.x,
            .y = gyro_corrected.y,
            .z = gyro_corrected.z,
        }).scale(0.5);

        // Compute quaternion derivative from gradient step
        const q_dot_step = Math.Quaternion{
            .w = 0.0, // No scalar component
            .x = -self.beta * step_vector.x, // Derived from step_vector
            .y = -self.beta * step_vector.y, // Derived from step_vector
            .z = -self.beta * step_vector.z, // Derived from step_vector
        };

        // Combine derivatives
        const q_dot = Math.Quaternion{
            .w = q_dot_gyro.w + q_dot_step.w,
            .x = q_dot_gyro.x + q_dot_step.x,
            .y = q_dot_gyro.y + q_dot_step.y,
            .z = q_dot_gyro.z + q_dot_step.z,
        };

        // Integrate to yield new quaternion
        self.q.w += q_dot.w * delta_t;
        self.q.x += q_dot.x * delta_t;
        self.q.y += q_dot.y * delta_t;
        self.q.z += q_dot.z * delta_t;

        self.q = self.q.normalize();

        q1 = self.q.w;
        q2 = self.q.x;
        q3 = self.q.y;
        q4 = self.q.z;

        // Update reference direction of flux
        // Reference direction of Earth's magnetic field
        q1q2 = q1 * q2;
        q1q3 = q1 * q3;
        q1q4 = q1 * q4;
        q2q2 = q2 * q2;
        q2q3 = q2 * q3;
        q2q4 = q2 * q4;
        q3q3 = q3 * q3;
        q3q4 = q3 * q4;
        q4q4 = q4 * q4;

        const mag_q = Math.Quaternion{
            .w = 0.0,
            .x = m_norm.x,
            .y = m_norm.y,
            .z = m_norm.z,
        };

        q_conj = self.q.conjugate();

        const h = self.q.multiply(mag_q).normalize().multiply(q_conj).normalize();

        const bx = std.math.sqrt(h.x * h.x + h.y * h.y);
        const bz = h.z;

        self.bx = bx;
        self.bz = bz;
    }
};

pub fn updateModelMatrix(
    accel: Vec3,
    gyro: Vec3,
    mag: Vec3,
    declination: f32,
    delta_time: f32,
    sensor_state: *SensorState,
) Math.Quaternion {
    const accel_gl = Vec3{
        .x = accel.x, // Right
        .y = accel.y, // Up
        .z = accel.z, // Back
    };

    const gyro_gl = Vec3{
        .x = gyro.x,
        .y = gyro.y,
        .z = gyro.z,
    };

    const mag_gl = Vec3{
        .x = mag.y, // Right
        .y = mag.x, // Up
        .z = -mag.z, // Back
    };

    if (sensor_state.filter == null) {
        sensor_state.filter = MadgwickFilter.init();
    }

    _ = declination;

    sensor_state.filter.?.update(
        gyro_gl,
        accel_gl,
        mag_gl,
        delta_time,
    );

    var q = Math.Quaternion{
        .w = sensor_state.filter.?.q.w,
        .x = sensor_state.filter.?.q.x,
        .y = sensor_state.filter.?.q.y,
        .z = sensor_state.filter.?.q.z,
    };

    if (sensor_state.filter.?.initial_orientation == null) {
        sensor_state.filter.?.initial_orientation = sensor_state.filter.?.q;
    }

    const initial_conj = sensor_state.filter.?.initial_orientation.?.conjugate();

    q = initial_conj.multiply(q);
    q = Math.Quaternion{
        .w = q.w,
        .x = -q.x,
        .y = -q.z,
        .z = q.y,
    };

    const rotation = q.toMatrix();
    const accel_corrected = Vec3{
        .x = accel_gl.x - (rotation[1] * 1),
        .y = accel_gl.z - (rotation[5] * 1),
        .z = accel_gl.y - (rotation[9] * 1),
    };

    sensor_state.velocity = accel_corrected.scale(delta_time).add(sensor_state.velocity);
    sensor_state.position = sensor_state.velocity.scale(delta_time).add(sensor_state.position);

    std.debug.print("Position: {d:.3} =>  Velocity: {d:.3}\n", .{
        [_]f32{
            sensor_state.position.x,
            @max(sensor_state.position.y, 0.0),
            sensor_state.position.z,
        },
        [_]f32{
            sensor_state.velocity.x,
            sensor_state.velocity.y,
            sensor_state.velocity.z,
        },
    });

    return q;
}
