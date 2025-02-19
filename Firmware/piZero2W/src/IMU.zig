const std = @import("std");
const time = std.time;
const math = std.math;

// MPU-9250 registers and constants
const MPU9250_ADDR = 0x68;
const WHO_AM_I = 0x75;
const PWR_MGMT_1 = 0x6B;
const CONFIG = 0x1A;
const GYRO_CONFIG = 0x1B;
const ACCEL_CONFIG = 0x1C;
const ACCEL_XOUT_H = 0x3B;
const GYRO_XOUT_H = 0x43;
const INT_PIN_CFG = 0x37;

// AK8963 registers
const AK8963_ADDR = 0x0C;
const AK8963_WHO_AM_I = 0x00;
const AK8963_CNTL1 = 0x0A;
const AK8963_ASAX = 0x10;
const AK8963_ST1 = 0x02;
const AK8963_HXL = 0x03;

// I2C Config
const I2C_PATH = "/dev/i2c-1";
const I2C_SLAVE = 0x0703;

const Readings = struct { x: f32, y: f32, z: f32 };

const ReadingsPacket = [44]u8;

pub const Mpu9250 = struct {
    i2c_fd: i32,
    mag_scale: [3]f32,
    prev_mag: Readings = .{
        .x = 0.0,
        .y = 0.0,
        .z = 0.0,
    },

    const Self = @This();

    pub fn init() !Self {
        const fd = try std.posix.open(I2C_PATH, std.posix.O{ .ACCMODE = .RDWR }, 0);
        if (fd < 0) {
            std.debug.print("Failed to open I2C bus\n", .{});
            return error.I2COpenFailed;
        }

        var self = Self{
            .i2c_fd = fd,
            .mag_scale = .{ 1.0, 1.0, 1.0 },
        };
        try self.initSensor();
        try self.initMagnetometer();
        return self;
    }

    fn writeByte(self: Self, addr: u8, reg: u8, value: u8) !void {
        if (std.os.linux.ioctl(self.i2c_fd, I2C_SLAVE, addr) < 0) {
            return error.I2CSlaveSelectFailed;
        }

        var buf = [_]u8{ reg, value };
        if (try std.posix.write(self.i2c_fd, &buf) != 2) {
            return error.I2CWriteFailed;
        }
    }

    fn readBytes(self: Self, addr: u8, reg: u8, buffer: []u8) !void {
        if (std.os.linux.ioctl(self.i2c_fd, I2C_SLAVE, addr) < 0) {
            return error.I2CSlaveSelectFailed;
        }

        // Write register address
        if (try std.posix.write(self.i2c_fd, &[_]u8{reg}) != 1) {
            return error.I2CWriteFailed;
        }

        // Read data
        if (try std.posix.read(self.i2c_fd, buffer) != buffer.len) {
            return error.I2CReadFailed;
        }
    }

    fn initSensor(self: Self) !void {
        // Check device ID
        var who_am_i: [1]u8 = undefined;
        try self.readBytes(MPU9250_ADDR, WHO_AM_I, &who_am_i);
        if (who_am_i[0] != 0x71) {
            std.debug.print("Wrong device ID: 0x{X:0>2}\n", .{who_am_i[0]});
            return error.WrongDeviceID;
        }

        // Wake up device
        try self.writeByte(MPU9250_ADDR, PWR_MGMT_1, 0x00);
        std.time.sleep(100 * std.time.ns_per_ms); // 100ms delay

        // Configure gyro (±250 dps)
        try self.writeByte(MPU9250_ADDR, GYRO_CONFIG, 0x00);

        // Configure accelerometer (±2g)
        try self.writeByte(MPU9250_ADDR, ACCEL_CONFIG, 0x00);
    }

    fn initMagnetometer(self: *Self) !void {
        // Enable I2C bypass to access magnetometer
        try self.writeByte(MPU9250_ADDR, INT_PIN_CFG, 0x02);
        std.time.sleep(10 * std.time.ns_per_ms);

        // Check magnetometer ID
        var mag_id: [1]u8 = undefined;
        try self.readBytes(AK8963_ADDR, AK8963_WHO_AM_I, &mag_id);
        if (mag_id[0] != 0x48) {
            std.debug.print("Wrong AK8963 ID: 0x{X:0>2}\n", .{mag_id[0]});
            return error.WrongMagnetometerID;
        }

        // Power down magnetometer
        try self.writeByte(AK8963_ADDR, AK8963_CNTL1, 0x00);
        std.time.sleep(10 * std.time.ns_per_ms);

        // Enter Fuse ROM access mode
        try self.writeByte(AK8963_ADDR, AK8963_CNTL1, 0x0F);
        std.time.sleep(10 * std.time.ns_per_ms);

        // Read sensitivity adjustment values
        var asa: [3]u8 = undefined;
        try self.readBytes(AK8963_ADDR, AK8963_ASAX, &asa);

        // Calculate scale factors
        self.mag_scale[0] = @as(f32, @floatFromInt(asa[0] - 128)) / 256.0 + 1.0;
        self.mag_scale[1] = @as(f32, @floatFromInt(asa[1] - 128)) / 256.0 + 1.0;
        self.mag_scale[2] = @as(f32, @floatFromInt(asa[2] - 128)) / 256.0 + 1.0;

        // Power down magnetometer
        try self.writeByte(AK8963_ADDR, AK8963_CNTL1, 0x00);
        std.time.sleep(10 * std.time.ns_per_ms);

        // Set continuous measurement mode (16-bit, 100Hz)
        try self.writeByte(AK8963_ADDR, AK8963_CNTL1, 0x16);
        std.time.sleep(10 * std.time.ns_per_ms);
    }

    pub fn readAccel(self: Self) !Readings {
        var buffer: [6]u8 = undefined;
        try self.readBytes(MPU9250_ADDR, ACCEL_XOUT_H, &buffer);

        // Convert to g's (±2g scale)
        const raw_x = @as(i16, @bitCast([2]u8{ buffer[1], buffer[0] }));
        const raw_y = @as(i16, @bitCast([2]u8{ buffer[3], buffer[2] }));
        const raw_z = @as(i16, @bitCast([2]u8{ buffer[5], buffer[4] }));

        return .{
            .x = @as(f32, @floatFromInt(raw_x)) / 16384.0,
            .y = @as(f32, @floatFromInt(raw_y)) / 16384.0,
            .z = @as(f32, @floatFromInt(raw_z)) / 16384.0,
        };
    }

    pub fn readGyro(self: Self) !Readings {
        var buffer: [6]u8 = undefined;
        try self.readBytes(MPU9250_ADDR, GYRO_XOUT_H, &buffer);

        // Convert to degrees per second (±250 dps scale)
        const raw_x = @as(i16, @bitCast([2]u8{ buffer[1], buffer[0] }));
        const raw_y = @as(i16, @bitCast([2]u8{ buffer[3], buffer[2] }));
        const raw_z = @as(i16, @bitCast([2]u8{ buffer[5], buffer[4] }));

        return .{
            .x = @as(f32, @floatFromInt(raw_x)) / 131.0,
            .y = @as(f32, @floatFromInt(raw_y)) / 131.0,
            .z = @as(f32, @floatFromInt(raw_z)) / 131.0,
        };
    }

    pub fn readMag(self: Self) !Readings {
        // Check data ready
        var st1: [1]u8 = undefined;
        try self.readBytes(AK8963_ADDR, AK8963_ST1, &st1);
        if ((st1[0] & 0x01) == 0) return error.MagDataNotReady;

        var buffer: [7]u8 = undefined;
        try self.readBytes(AK8963_ADDR, AK8963_HXL, &buffer);

        // Check magnetic sensor overflow
        if ((buffer[6] & 0x08) != 0) return error.MagOverflow;

        const raw_x = @as(i16, @bitCast([2]u8{ buffer[1], buffer[0] }));
        const raw_y = @as(i16, @bitCast([2]u8{ buffer[3], buffer[2] }));
        const raw_z = @as(i16, @bitCast([2]u8{ buffer[5], buffer[4] }));

        // Apply sensitivity adjustments and convert to microTesla
        return .{
            .x = @as(f32, @floatFromInt(raw_x)) * self.mag_scale[0] * 4912.0 / 32760.0,
            .y = @as(f32, @floatFromInt(raw_y)) * self.mag_scale[1] * 4912.0 / 32760.0,
            .z = @as(f32, @floatFromInt(raw_z)) * self.mag_scale[2] * 4912.0 / 32760.0,
        };
    }

    pub fn read(self: *Self) !ReadingsPacket {
        const accel = try self.readAccel();
        const gyro = try self.readGyro();
        const mag = self.readMag() catch |err| switch (err) {
            error.MagDataNotReady => blk: {
                std.debug.print("Mag Data not ready!\n", .{});
                break :blk self.prev_mag;
            },
            error.MagOverflow => blk: {
                std.debug.print("Mag Overflow Detected!\n", .{});
                break :blk self.prev_mag;
            },
            else => {
                std.debug.print("Unknown error reading from Magnetometer => {any}\n", .{err});
                return err;
            },
        };
        self.prev_mag = mag;

        // std.debug.print(
        //     \\Accel: X={d:6.2} Y={d:6.2} Z={d:6.2} (g)
        //     \\Gyro:  X={d:6.2} Y={d:6.2} Z={d:6.2} (deg/s)
        //     \\Mag:   X={d:6.2} Y={d:6.2} Z={d:6.2} (uT)
        //     \\
        //     \\
        // , .{
        //     accel.x, accel.y, accel.z,
        //     gyro.x,  gyro.y,  gyro.z,
        //     mag.x,   mag.y,   mag.z,
        // });

        var packet: ReadingsPacket = undefined;
        std.mem.writeInt(u32, packet[0..4], @bitCast(accel.x), .little);
        std.mem.writeInt(u32, packet[4..8], @bitCast(accel.y), .little);
        std.mem.writeInt(u32, packet[8..12], @bitCast(accel.z), .little);

        std.mem.writeInt(u32, packet[12..16], @bitCast(gyro.x), .little);
        std.mem.writeInt(u32, packet[16..20], @bitCast(gyro.y), .little);
        std.mem.writeInt(u32, packet[20..24], @bitCast(gyro.z), .little);

        std.mem.writeInt(u32, packet[24..28], @bitCast(mag.x), .little);
        std.mem.writeInt(u32, packet[28..32], @bitCast(mag.y), .little);
        std.mem.writeInt(u32, packet[32..36], @bitCast(mag.z), .little);

        return packet;
    }

    pub fn deinit(self: Self) void {
        _ = std.posix.close(self.i2c_fd);
    }
};

pub const UDP_Provider = struct {
    imu: Mpu9250,
    socket: std.posix.socket_t,
    dest_addr: std.posix.sockaddr,
    dest_addr_len: std.posix.socklen_t,
    start_time: i64,

    const Self = @This();

    pub fn init(server_ip: []const u8, server_port: u16) !Self {
        const socket = try std.posix.socket(
            std.posix.AF.INET,
            std.posix.SOCK.DGRAM,
            0,
        );

        const dest_addr = (try std.net.Address.parseIp4(server_ip, server_port)).any;

        return Self{
            .imu = try Mpu9250.init(),
            .socket = socket,
            .dest_addr = dest_addr,
            .dest_addr_len = @sizeOf(std.posix.sockaddr),
            .start_time = std.time.milliTimestamp(),
        };
    }

    pub fn send(self: *Self) !void {
        var packet = try self.imu.read();

        // Add timestamp
        const timestamp: i64 = std.time.milliTimestamp() * 1000; // Convert to microseconds
        std.mem.writeInt(i64, packet[36..44], timestamp, .little);

        _ = try std.posix.sendto(
            self.socket,
            &packet,
            0,
            &self.dest_addr,
            self.dest_addr_len,
        );
    }

    pub fn run(self: *Self) !void {
        std.debug.print("Starting IMU test data transmission at 1000Hz to {s}...\n", .{"192.168.1.171"});

        while (true) {
            try self.send();
            std.time.sleep(1 * std.time.ns_per_ms); // 1ms delay
        }
    }

    pub fn deinit(self: *Self) void {
        std.os.close(self.socket);
    }
};

pub fn main() !void {
    const server_ip = "192.168.1.171";
    const server_port = 8000;

    var imu_reader = try UDP_Provider.init(
        server_ip,
        server_port,
    );

    try imu_reader.run();
}
