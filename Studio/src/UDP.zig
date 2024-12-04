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

const HandlerError = error{FailedToParse};

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

//store threads in a table, then when deinit is called, join threads to stop all processes
pub fn start(self: *Self, handler: HandlerInterface) !void {
    _ = try std.Thread.spawn(self.spawn_config, receive, .{ self, handler });
    // var udp_transmitting_thread = try std.Thread.spawn(spawn_config, send, .{});

    // _ = receiveThread.join();
    // _ = udp_transmitting_thread.join();
}

pub fn receive(self: *Self, handler: HandlerInterface) !void {
    std.debug.print("\nUDP Rx server started\n", .{});
    const parsed_address = try std.net.Address.parseIp4("0.0.0.0", self.host_port);

    const socket = try std.posix.socket(
        std.posix.AF.INET,
        std.posix.SOCK.DGRAM,
        0,
    );

    try std.posix.bind(socket, &parsed_address.any, parsed_address.getOsSockLen());
    std.debug.print("UDP server listening on {}\n\n", .{parsed_address});

    var src_addr: std.posix.sockaddr = undefined;
    var src_addr_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);
    var recv_buf: [1514]u8 = undefined;

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

    std.debug.print("UDP Tx Server Started\n", .{});

    const dest_addr: std.posix.sockaddr = (try std.net.Address.parseIp4(self.client_ip, self.client_port)).any;
    const dest_addr_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);
    var send_buff: [1]u8 = undefined;

    const sleep_duration_ns = 250 * std.time.ns_per_ms; // 250 milliseconds to nanoseconds

    while (true) {
        std.debug.print("==============================\nSending Packet to Client => {any}\n", .{send_buff});
        _ = try std.posix.sendto(socket, &send_buff, 0, &dest_addr, dest_addr_len);

        // Sleep for 100 milliseconds
        std.time.sleep(sleep_duration_ns);
    }
}
