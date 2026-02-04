const std = @import("std");

const I2C_RDWR = 0x0707;
const I2C_M_RD = 0x0001;

const I2C_Msg = extern struct {
    addr: u16,
    flags: u16,
    len: u16,
    buf: [*]u8,
};

const I2C_Rdwr_Ioctl_Data = extern struct {
    msgs: [*]I2C_Msg,
    nmsgs: u32,
};

pub fn readBlock(fd: i32, addr: u8, reg: u8, buffer: []u8) !void {
    var msgs: [2]I2C_Msg = undefined;

    // Write register address
    msgs[0] = .{
        .addr = addr,
        .flags = 0,
        .len = 1,
        .buf = @ptrCast(@constCast(&reg)),
    };

    // Read data
    msgs[1] = .{
        .addr = addr,
        .flags = I2C_M_RD,
        .len = @intCast(buffer.len),
        .buf = buffer.ptr,
    };

    var data = I2C_Rdwr_Ioctl_Data{
        .msgs = &msgs,
        .nmsgs = 2,
    };

    if (std.os.linux.ioctl(fd, I2C_RDWR, @intFromPtr(&data)) < 0) {
        return error.I2CTransferFailed;
    }
}

pub fn openI2C(path: []const u8) !i32 {
    const fd = try std.posix.open(path, std.posix.O{ .ACCMODE = .RDWR }, 0);
    if (fd < 0) {
        std.debug.print("Failed to open I2C bus\n", .{});
        return error.I2COpenFailed;
    }
    return fd;
}

// Helper function to check if a device is present on the I2C bus
pub fn isDevicePresent(fd: i32, addr: u8, slave: u32) bool {
    if (std.os.linux.ioctl(fd, slave, addr) < 0) {
        return false;
    }

    // Try to read a single byte
    var buf: [1]u8 = undefined;
    const bytes_read = std.posix.read(fd, &buf) catch {
        return false;
    };

    return bytes_read == 1;
}
