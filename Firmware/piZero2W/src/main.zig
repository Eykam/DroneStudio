const std = @import("std");
const net = std.net;

const CLIENT_IP: []const u8 = "192.168.1.171";
const CLIENT_PORT: u16 = 8888;
const VIDEO_OFFSET = 16 + 4;
const MAX_PACKET_SIZE = 1500 - 28 - VIDEO_OFFSET;

// Helper function to convert integer to big-endian byte array
fn intToBytesBE(value: i128) [16]u8 {
    var bytes: [16]u8 = undefined;
    std.mem.writeInt(i128, &bytes, value, .big);
    return bytes;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const socket = try std.posix.socket(
        std.posix.AF.INET,
        std.posix.SOCK.DGRAM,
        0,
    );

    const dest_addr: std.posix.sockaddr = (try std.net.Address.parseIp4(CLIENT_IP, CLIENT_PORT)).any;
    const dest_addr_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);

    // Define the frame parameters

    // Define the command and its arguments
    const args = &[_][]const u8{
        "libcamera-vid",
        "-n",
        "-t",
        "0",
        "--level",
        "4.2",
        "--framerate",
        "75",
        "--width",
        "1280",
        "--height",
        "720",
        "--inline",
        "--denoise",
        "cdn_off",
        "-g",
        "1",
        // "--keyframe-period",
        // "60", //
        "-o",
        "-",
    };

    // Spawn the libcamera-vid process
    var proc = std.process.Child.init(args, allocator);
    proc.stdout_behavior = .Pipe;
    proc.stderr_behavior = .Pipe;

    try proc.spawn();
    defer {
        _ = proc.kill() catch |err| {
            std.debug.print("Failed to kill child process: {}", .{err});
        };
    }

    // Get the stdout pipe reader
    const stdout_pipe = proc.stdout orelse return error.StdoutPipeError;

    var packet = try allocator.alloc(u8, VIDEO_OFFSET + MAX_PACKET_SIZE);
    defer allocator.free(packet);

    var frame_id: u32 = 0;
    var frame_bytes_received: usize = 0;

    while (true) {
        var read_buffer: [8192]u8 = undefined;
        const bytes_read = try stdout_pipe.read(&read_buffer);

        if (bytes_read == 0) break; // EOF

        var offset: usize = 0;

        while (offset < bytes_read) {
            const remaining_frame_bytes = MAX_PACKET_SIZE - frame_bytes_received;
            const bytes_available = bytes_read - offset;
            const bytes_to_copy = @min(remaining_frame_bytes, bytes_available);

            // Append bytes to the frame buffer
            @memcpy(packet[VIDEO_OFFSET + frame_bytes_received .. VIDEO_OFFSET + frame_bytes_received + bytes_to_copy], read_buffer[offset .. offset + bytes_to_copy]);
            frame_bytes_received += bytes_to_copy;
            offset += bytes_to_copy;

            // Check if the frame buffer is full
            if (frame_bytes_received == MAX_PACKET_SIZE) {
                // Capture the timestamp
                const timestamp = std.time.nanoTimestamp(); // better to define here or when frame buffer is full??

                // Encode timestamp as big endian
                std.mem.writeInt(u32, packet[0..4], frame_id, .big);
                @memcpy(packet[4..20], intToBytesBE(@intCast(timestamp))[0..]);

                _ = try std.posix.sendto(socket, packet, 0, &dest_addr, dest_addr_len);

                // const N = 10;
                // var parity_packet = computeParity(packets[N - 10 .. N]);
                // try std.posix.sendto(socket, parity_packet, 0, &dest_addr, dest_addr_len);

                std.debug.print("Sent packet: {} bytes, Timestamp: {d}\n", .{
                    frame_bytes_received, timestamp,
                });

                frame_bytes_received = 0;
                frame_id += 1;
            }
        }
    }

    // Wait for the libcamera-vid process to exit
    _ = try proc.wait();

    std.debug.print("libcamera-vid process terminated.\n", .{});
}
