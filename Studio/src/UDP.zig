// const Queue = @import("./queue.zig").Queue;
const std = @import("std");
const time = std.time;
const Instant = time.Instant;
const Mesh = @import("Shape.zig").Mesh;
const Transformations = @import("Transformations.zig");
const Vec3 = Transformations.Vec3;

const Self = @This();

const HandlerError = error{FailedToParse};

host_ip: []const u8,
host_port: u16,

client_ip: []const u8,
client_port: u16,

spawn_config: std.Thread.SpawnConfig = std.Thread.SpawnConfig{
    .allocator = std.heap.page_allocator,
    .stack_size = 16 * 1024 * 1024, // You can adjust the stack size as needed
},

pub const HandlerInterface = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        update: *const fn (ptr: *anyopaque, data: []const u8) anyerror!void,
    };

    pub fn update(self: @This(), data: []const u8) !void {
        return self.vtable.update(self.ptr, data);
    }
};

pub fn Handler(comptime T: type) type {
    return struct {
        inner: T,

        pub fn init(inner: T) @This() {
            return .{ .inner = inner };
        }

        pub fn interface(self: *@This()) HandlerInterface {
            return .{
                .ptr = @ptrCast(self),
                .vtable = &.{
                    .update = update,
                },
            };
        }

        fn update(ptr: *anyopaque, data: []const u8) !void {
            const self = @as(*@This(), @ptrCast(@alignCast(ptr)));
            return self.inner.update(data);
        }
    };
}

pub fn init(host_ip: []const u8, host_port: u16, client_ip: []const u8, client_port: u16) Self {
    return Self{
        .host_ip = host_ip,
        .host_port = host_port,
        .client_ip = client_ip,
        .client_port = client_port,
    };
}

pub fn start(self: *Self, handler: HandlerInterface) !void {
    _ = try std.Thread.spawn(self.spawn_config, receive, .{ self, handler });
    // var udp_transmitting_thread = try std.Thread.spawn(spawn_config, send, .{});

    // _ = receiveThread.join();
    // _ = udp_transmitting_thread.join();
}

//Refactor into a generic function event_handler
pub fn receive(self: *Self, handler: HandlerInterface) !void {
    std.debug.print("Starting UDP server\n", .{});
    const parsed_address = try std.net.Address.parseIp4("0.0.0.0", self.host_port);

    const socket = try std.posix.socket(
        std.posix.AF.INET,
        std.posix.SOCK.DGRAM,
        0,
    );

    try std.posix.bind(socket, &parsed_address.any, parsed_address.getOsSockLen());
    std.debug.print("UDP server listening on {}\n", .{parsed_address});

    var src_addr: std.posix.sockaddr = undefined;
    var src_addr_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);
    var recv_buf: [1024]u8 = undefined;

    while (true) {
        const bytes_received = try std.posix.recvfrom(
            socket,
            &recv_buf,
            0,
            &src_addr,
            &src_addr_len,
        );

        try handler.update(recv_buf[0..bytes_received]);
    }
}

pub fn send(self: *Self) !void {
    const socket = try std.posix.socket(
        std.posix.AF.INET,
        std.posix.SOCK.DGRAM,
        0,
    );

    std.debug.print("UDP Transmitting Server Started\n", .{});

    const dest_addr: std.posix.sockaddr = (try std.net.Address.parseIp4(self.client_ip, self.client_port)).any;
    const dest_addr_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);
    var send_buff: [1]u8 = undefined;

    const sleep_duration_ns = 250 * std.time.ns_per_ms; // 250 milliseconds to nanoseconds

    while (true) {
        std.debug.print("==============================\nSending Packet to Scanner => {any}\n", .{send_buff});
        _ = try std.posix.sendto(socket, &send_buff, 0, &dest_addr, dest_addr_len);

        // Sleep for 100 milliseconds
        std.time.sleep(sleep_duration_ns);
    }
}

pub const Pose = struct {
    accel: Vec3,
    gyro: Vec3,
    mag: Vec3,
    timestamp: i64,
};

pub const PoseHandler = struct {
    mesh: *Mesh,
    packet_count: usize = 0,
    prev_instant: time.Instant,
    prev_timestamp: i64 = 0,
    stale_count: usize = 0,

    pub fn init(mesh: *Mesh) PoseHandler {
        return .{
            .mesh = mesh,
            .prev_instant = time.Instant.now() catch unreachable,
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
            .x = @bitCast(std.mem.readInt(u32, packet[24..28], .little)),
            .y = -1.0 * @as(f32, @bitCast(std.mem.readInt(u32, packet[32..36], .little))),
            .z = -1.0 * @as(f32, @bitCast(std.mem.readInt(u32, packet[28..32], .little))),
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

        // const delta_instant = @as(f32, @floatFromInt(Instant.since(curr_time, prev_time))) / 1e9;
        // prev_time = curr_time;
        // std.debug.print("Delta Time: {d} vs default : {d} => {d}\n", .{ delta_tdelta_instantime, 1.0 / 875.0, delta_instant - (1.0 / 875.0) });

        if (delta_instant_to_seconds > 1.0) {
            const packets_per_sec = self.packet_count / @as(u32, @intFromFloat(delta_instant_to_seconds));
            std.debug.print("==========\nTwo seconds have passed.\nPackets / sec counted: {d}\nThroughput: {d} B/s\n", .{ packets_per_sec, packets_per_sec * data.len });
            self.prev_instant = curr_instant;
            self.packet_count = 0;
        }

        // for right now, 0.0 is a placeholder for delta_time passed to updateKalman function
        // need to figure out why giving actual seconds in dt gives janky results
        try Transformations.updateModelMatrix(self.mesh, pose.accel, pose.gyro, pose.mag, delta_time);
        self.packet_count += 1;
    }
};
