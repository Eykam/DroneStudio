// const Queue = @import("./queue.zig").Queue;
const std = @import("std");
const time = std.time;
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

    while (true) {
        _ = try std.posix.recvfrom(
            socket,
            &recv_buf,
            0,
            &src_addr,
            &src_addr_len,
        );

        // std.debug.print("====================\nBuffer Length (Bytes): {d}\n", .{recv_len});

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

        try update(mesh, accel, gyro);

        std.debug.print("Accel => {d}\n", .{[_]f32{
            accel.x,
            accel.y,
            accel.z,
        }});
        std.debug.print("Gyro => {d}\n", .{[_]f32{
            gyro.x,
            gyro.y,
            gyro.z,
        }});
        // std.debug.print("Mag => {d}\n", .{[_]f32{
        //     @bitCast(std.mem.readInt(u32, recv_buf[24..28], .little)),
        //     @bitCast(std.mem.readInt(u32, recv_buf[28..32], .little)),
        //     @bitCast(std.mem.readInt(u32, recv_buf[32..36], .little)),
        // }});
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
