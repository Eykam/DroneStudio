// const Queue = @import("./queue.zig").Queue;
const std = @import("std");
const time = std.time;
const Instant = time.Instant;
const Mesh = @import("Shape.zig").Mesh;
const Transformations = @import("Transformations.zig");
const Vec3 = Transformations.Vec3;

const Self = @This();

host_ip: []const u8,
host_port: u16,

client_ip: []const u8,
client_port: u16,

spawn_config: std.Thread.SpawnConfig = std.Thread.SpawnConfig{
    .allocator = std.heap.page_allocator,
    .stack_size = 16 * 1024 * 1024, // You can adjust the stack size as needed
},

pub fn init(host_ip: []const u8, host_port: u16, client_ip: []const u8, client_port: u16) Self {
    return Self{
        .host_ip = host_ip,
        .host_port = host_port,
        .client_ip = client_ip,
        .client_port = client_port,
    };
}

pub fn start(self: *Self, mesh: *Mesh, update: anytype) !void {
    _ = try std.Thread.spawn(self.spawn_config, receive, .{ self, mesh, update });
    // var udp_transmitting_thread = try std.Thread.spawn(spawn_config, send, .{});

    // _ = receiveThread.join();
    // _ = udp_transmitting_thread.join();
}

pub fn receive(self: *Self, mesh: *Mesh, update: anytype) !void {
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
    var packet_count: usize = 0;
    var prev_time: time.Instant = try Instant.now();
    var prev_timestamp: i64 = 0;

    while (true) {
        const packet_size = try std.posix.recvfrom(
            socket,
            &recv_buf,
            0,
            &src_addr,
            &src_addr_len,
        );

        const accel = Vec3{
            .x = @bitCast(std.mem.readInt(u32, recv_buf[0..4], .little)),
            .y = @as(f32, @bitCast(std.mem.readInt(u32, recv_buf[8..12], .little))),
            .z = -1.0 * @as(f32, @bitCast(std.mem.readInt(u32, recv_buf[4..8], .little))),
        };

        const gyro = Vec3{
            .x = @bitCast(std.mem.readInt(u32, recv_buf[12..16], .little)),
            .y = @bitCast(std.mem.readInt(u32, recv_buf[20..24], .little)),
            .z = -1.0 * @as(f32, @bitCast(std.mem.readInt(u32, recv_buf[16..20], .little))),
        };

        const mag = Vec3{
            .x = @bitCast(std.mem.readInt(u32, recv_buf[24..28], .little)),
            .y = @bitCast(std.mem.readInt(u32, recv_buf[32..36], .little)),
            .z = -1.0 * @as(f32, @bitCast(std.mem.readInt(u32, recv_buf[28..32], .little))),
        };

        const timestamp: i64 = @bitCast(std.mem.readInt(i64, recv_buf[36..44], .little));

        if (timestamp - prev_timestamp < 0) {
            std.debug.print("Received stale packet, continuing...\n", .{ prev_timestamp, timestamp });
            prev_timestamp = timestamp;
            continue;
        }

        prev_timestamp = timestamp;

        const curr_time = try Instant.now();
        const delta_time = Instant.since(curr_time, prev_time);

        // const delta_time = @as(f32, @floatFromInt(Instant.since(curr_time, prev_time))) / 1e9;
        // prev_time = curr_time;
        // std.debug.print("Delta Time: {d} vs default : {d} => {d}\n", .{ delta_time, 1.0 / 875.0, delta_time - (1.0 / 875.0) });

        if (delta_time > 1e9) {
            const time_to_secs = delta_time / @as(u64, @intFromFloat(1e9));
            const packets_per_sec = packet_count / time_to_secs;
            std.debug.print("==========\nTwo seconds have passed.\nPackets / sec counted: {d}\nThroughput: {d} B/s\n", .{ packets_per_sec, packets_per_sec * packet_size });
            prev_time = curr_time;
            packet_count = 0;
        }

        // for right now, 0.0 is a placeholder for delta_time passed to updateKalman function
        // need to figure out why giving actual seconds in dt gives janky results
        try update(mesh, accel, gyro, mag, 0.0);
        packet_count += 1;
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
