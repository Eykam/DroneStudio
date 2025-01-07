const std = @import("std");
const Math = @import("Math.zig");
// const KalmanState = Transformations.KalmanState;
const Vec3 = Math.Vec3;
const MadgwickFilter = Math.MadgwickFilter;
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
    gyro_offset: Vec3(f32) = Vec3(f32).zero(),
    accel_offset: Vec3(f32) = Vec3(f32).zero(),
    velocity: Vec3(f32) = Vec3(f32).zero(),
    position: Vec3(f32),

    pub fn init(node: *Node) SensorState {
        return SensorState{
            .position = Vec3(f32).init(node.position[0], node.position[1], node.position[2]),
        };
    }
};

pub const Pose = struct {
    accel: Vec3(f32),
    gyro: Vec3(f32),
    mag: Vec3(f32),
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
        const accel = Vec3(f32).init(
            @bitCast(std.mem.readInt(u32, packet[0..4], .little)),
            @bitCast(std.mem.readInt(u32, packet[4..8], .little)),
            @bitCast(std.mem.readInt(u32, packet[8..12], .little)),
        );

        const gyro = Vec3(f32).init(
            Math.radians(@bitCast(std.mem.readInt(u32, packet[12..16], .little))),
            Math.radians(@bitCast(std.mem.readInt(u32, packet[16..20], .little))),
            Math.radians(@bitCast(std.mem.readInt(u32, packet[20..24], .little))),
        );

        const mag = Vec3(f32).init(
            @bitCast(std.mem.readInt(u32, packet[24..28], .little)),
            @bitCast(std.mem.readInt(u32, packet[28..32], .little)),
            @bitCast(std.mem.readInt(u32, packet[32..36], .little)),
        );

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
                self.sensor_state.accel_offset = self.sensor_state.accel_offset.add(Vec3(f32).init(
                    pose.accel.x,
                    pose.accel.y,
                    1.0 - pose.accel.z,
                ));
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

        const rotation = Math.updateModelMatrix(accel_calibrated, gyro_calibrated, pose.mag, DECLINATION_ANGLE, delta_time, &self.sensor_state);
        self.node.setRotation(rotation);
        self.packet_count += 1;
    }
};
