const std = @import("std");
const Transformations = @import("Transformations.zig");
const KalmanState = Transformations.KalmanState;
const Vec3 = Transformations.Vec3;
const Node = @import("Node.zig");
const time = std.time;
const Instant = time.Instant;

pub const SensorState = struct {
    previous_yaw: f32,
    gyro_integrated_yaw: f32,
    mag_valid: bool,
    mag_updated: bool = false,
    previous_mag: f32 = 0,
    rollKalman: KalmanState = Transformations.initKalmanState(0.0, 0.0),
    pitchKalman: KalmanState = Transformations.initKalmanState(0.0, 0.0),

    pub fn init() SensorState {
        return SensorState{
            .previous_yaw = 0,
            .gyro_integrated_yaw = 0,
            .mag_valid = false,
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
    prev_timestamp: i64 = 0,
    stale_count: usize = 0,
    sensor_state: SensorState,

    pub fn init(node: *Node) PoseHandler {
        return .{
            .node = node,
            .prev_instant = time.Instant.now() catch unreachable,
            .sensor_state = SensorState.init(),
        };
    }

    pub fn parse(packet: []const u8) !Pose {
        const accel = Vec3{
            .x = @bitCast(std.mem.readInt(u32, packet[0..4], .little)),
            .y = @as(f32, @bitCast(std.mem.readInt(u32, packet[8..12], .little))),
            .z = -1.0 * @as(f32, @bitCast(std.mem.readInt(u32, packet[4..8], .little))),
        };

        const gyro = Vec3{
            .x = @bitCast(std.mem.readInt(u32, packet[12..16], .little)),
            .y = @bitCast(std.mem.readInt(u32, packet[20..24], .little)),
            .z = -1.0 * @as(f32, @bitCast(std.mem.readInt(u32, packet[16..20], .little))),
        };

        const mag = Vec3{
            .x = @bitCast(std.mem.readInt(u32, packet[28..32], .little)),
            .y = @bitCast(std.mem.readInt(u32, packet[32..36], .little)),
            .z = -1.0 * @as(f32, @bitCast(std.mem.readInt(u32, packet[24..28], .little))),
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
        const pose = try PoseHandler.parse(data);

        const delta_time = @as(f32, @floatFromInt(pose.timestamp - self.prev_timestamp)) / 1e6;
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

        // placeholder for current position
        const updatedMatrix = Transformations.updateModelMatrix(pose.accel, pose.gyro, pose.mag, delta_time, &self.sensor_state);
        self.node.setRotation(updatedMatrix[0], updatedMatrix[1], updatedMatrix[2]);
        self.packet_count += 1;
    }
};
