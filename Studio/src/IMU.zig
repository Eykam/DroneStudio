const std = @import("std");
const I2C = @import("core/I2C.zig");
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

const INA219_ADDR = 0x40; // Default I2C address
const INA219_CONFIG_REG = 0x00;
const INA219_SHUNT_VOLTAGE_REG = 0x01;
const INA219_BUS_VOLTAGE_REG = 0x02;
const INA219_POWER_REG = 0x03;
const INA219_CURRENT_REG = 0x04;
const INA219_CALIBRATION_REG = 0x05;

// Configuration values
const INA219_CONFIG_BVOLTAGERANGE_32V = 0x2000; // 0-32V Range
const INA219_CONFIG_GAIN_8_320MV = 0x1800; // Gain 8, Range +/- 320mV
const INA219_CONFIG_BADCRES_12BIT = 0x0400; // 12-bit bus res = 0..4097
const INA219_CONFIG_SADCRES_12BIT_1S_532US = 0x0018; // 1 x 12-bit shunt sample
const INA219_CONFIG_MODE_SANDBVOLT_CONTINUOUS = 0x0007; // Continuous sampling

const I2C_PATH = "/dev/i2c-1";
const I2C_SLAVE = 0x0703;

const Readings = struct { x: f32, y: f32, z: f32 };
const ReadingsPacket = [44]u8;

pub const Mpu9250 = struct {
    i2c_fd: i32,
    mag_scale: [3]f32,
    mag_counter: u32 = 0,
    prev_mag: Readings = .{
        .x = 0.0,
        .y = 0.0,
        .z = 0.0,
    },

    const Self = @This();

    pub fn init() !?Self {
        const fd = I2C.openI2C(I2C_PATH) catch |err| {
            std.debug.print("Failed to open I2C for MPU9250: {any}\n", .{err});
            return null;
        };

        if (!I2C.isDevicePresent(fd, MPU9250_ADDR, I2C_SLAVE)) {
            std.debug.print("MPU9250 not detected on I2C bus\n", .{});
            _ = std.posix.close(fd);
            return null;
        }

        var self = Self{
            .i2c_fd = fd,
            .mag_scale = .{ 1.0, 1.0, 1.0 },
        };

        self.initSensor() catch |err| {
            std.debug.print("Failed to initialize MPU9250: {any}\n", .{err});
            _ = std.posix.close(fd);
            return null;
        };

        self.initMagnetometer() catch |err| {
            std.debug.print("Failed to initialize Magnetometer: {any}\n", .{err});
            // Continue even if magnetometer fails
        };

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

        // Disable all FIFOs and DMP
        try self.writeByte(MPU9250_ADDR, 0x6A, 0x00);

        // Configure low pass filter - DLPF_CFG = 1 (184Hz bandwidth)
        try self.writeByte(MPU9250_ADDR, CONFIG, 0x00);

        // Configure gyro (±250 dps)
        try self.writeByte(MPU9250_ADDR, GYRO_CONFIG, 0x00);

        // Configure accelerometer (±2g)
        try self.writeByte(MPU9250_ADDR, ACCEL_CONFIG, 0x00);
        try self.writeByte(MPU9250_ADDR, 0x1D, 0x00);

        try self.writeByte(MPU9250_ADDR, 0x19, 0x00);
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

    pub fn readAccelGyro(self: Self) !struct { accel: Readings, gyro: Readings } {
        var buffer: [14]u8 = undefined; // 6 bytes accel + 2 bytes temp + 6 bytes gyro

        // Read from ACCEL_XOUT_H through GYRO_ZOUT_L in one transaction
        try I2C.readBlock(self.i2c_fd, MPU9250_ADDR, ACCEL_XOUT_H, &buffer);

        const accel = Readings{
            .x = @as(f32, @floatFromInt(@as(i16, @bitCast([2]u8{ buffer[1], buffer[0] })))) / 16384.0,
            .y = @as(f32, @floatFromInt(@as(i16, @bitCast([2]u8{ buffer[3], buffer[2] })))) / 16384.0,
            .z = @as(f32, @floatFromInt(@as(i16, @bitCast([2]u8{ buffer[5], buffer[4] })))) / 16384.0,
        };

        const gyro = Readings{
            .x = @as(f32, @floatFromInt(@as(i16, @bitCast([2]u8{ buffer[9], buffer[8] })))) / 131.0,
            .y = @as(f32, @floatFromInt(@as(i16, @bitCast([2]u8{ buffer[11], buffer[10] })))) / 131.0,
            .z = @as(f32, @floatFromInt(@as(i16, @bitCast([2]u8{ buffer[13], buffer[12] })))) / 131.0,
        };

        return .{ .accel = accel, .gyro = gyro };
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
        const readings = try self.readAccelGyro();

        self.mag_counter +%= 1;

        if (self.mag_counter % 10 == 0) {
            self.prev_mag = self.readMag() catch |err| switch (err) {
                error.MagDataNotReady => self.prev_mag,
                error.MagOverflow => blk: {
                    std.debug.print("Mag Overflow Detected!\n", .{});
                    break :blk self.prev_mag;
                },
                else => {
                    std.debug.print("Unknown error reading from Magnetometer => {any}\n", .{err});
                    return err;
                },
            };
        }

        var packet: ReadingsPacket = undefined;
        std.mem.writeInt(u32, packet[0..4], @bitCast(readings.accel.x), .little);
        std.mem.writeInt(u32, packet[4..8], @bitCast(readings.accel.y), .little);
        std.mem.writeInt(u32, packet[8..12], @bitCast(readings.accel.z), .little);

        std.mem.writeInt(u32, packet[12..16], @bitCast(readings.gyro.x), .little);
        std.mem.writeInt(u32, packet[16..20], @bitCast(readings.gyro.y), .little);
        std.mem.writeInt(u32, packet[20..24], @bitCast(readings.gyro.z), .little);

        std.mem.writeInt(u32, packet[24..28], @bitCast(self.prev_mag.x), .little);
        std.mem.writeInt(u32, packet[28..32], @bitCast(self.prev_mag.y), .little);
        std.mem.writeInt(u32, packet[32..36], @bitCast(self.prev_mag.z), .little);

        return packet;
    }

    pub fn deinit(self: Self) void {
        _ = std.posix.close(self.i2c_fd);
    }
};

pub const Ina219 = struct {
    i2c_fd: i32,

    const Self = @This();

    pub fn init() !?Self {
        const fd = I2C.openI2C(I2C_PATH) catch |err| {
            std.debug.print("Failed to open I2C for INA219: {any}\n", .{err});
            return null;
        };

        // Check if INA219 is present
        if (!I2C.isDevicePresent(fd, INA219_ADDR, I2C_SLAVE)) {
            std.debug.print("INA219 not detected on I2C bus\n", .{});
            _ = std.posix.close(fd);
            return null;
        }

        var self = Self{
            .i2c_fd = fd,
        };

        self.initSensor() catch |err| {
            std.debug.print("Failed to initialize INA219: {any}\n", .{err});
            _ = std.posix.close(fd);
            return null;
        };

        return self;
    }

    fn initSensor(self: Self) !void {
        // Configure INA219 with default settings
        const config: u16 = INA219_CONFIG_BVOLTAGERANGE_32V |
            INA219_CONFIG_GAIN_8_320MV |
            INA219_CONFIG_BADCRES_12BIT |
            INA219_CONFIG_SADCRES_12BIT_1S_532US |
            INA219_CONFIG_MODE_SANDBVOLT_CONTINUOUS;

        try self.writeRegister16(INA219_CONFIG_REG, config);

        // Wait for ADC ready
        std.time.sleep(1 * std.time.ns_per_ms);
    }

    fn writeRegister16(self: Self, reg: u8, value: u16) !void {
        if (std.os.linux.ioctl(self.i2c_fd, I2C_SLAVE, INA219_ADDR) < 0) {
            return error.I2CSlaveSelectFailed;
        }

        var buf = [_]u8{
            reg,
            @intCast((value >> 8) & 0xFF), // High byte
            @intCast(value & 0xFF), // Low byte
        };

        if (try std.posix.write(self.i2c_fd, &buf) != 3) {
            return error.I2CWriteFailed;
        }
    }

    fn readRegister16(self: Self, reg: u8) !u16 {
        if (std.os.linux.ioctl(self.i2c_fd, I2C_SLAVE, INA219_ADDR) < 0) {
            return error.I2CSlaveSelectFailed;
        }

        // Write register address
        if (try std.posix.write(self.i2c_fd, &[_]u8{reg}) != 1) {
            return error.I2CWriteFailed;
        }

        // Read data
        var buf: [2]u8 = undefined;
        if (try std.posix.read(self.i2c_fd, &buf) != 2) {
            return error.I2CReadFailed;
        }

        // Convert from big-endian (INA219 format)
        return (@as(u16, buf[0]) << 8) | buf[1];
    }

    pub fn readBusVoltage(self: Self) !f32 {
        var raw_voltage = try self.readRegister16(INA219_BUS_VOLTAGE_REG);

        // The bus voltage register gives value in 4mV units, with first 3 bits being status flags
        raw_voltage = (raw_voltage >> 3) & 0x1FFF; // Shift right 3 and mask with 0x1FFF

        // Convert to volts
        return @as(f32, @floatFromInt(raw_voltage)) * 0.004;
    }

    pub fn readShuntVoltage(self: Self) !f32 {
        const raw_voltage = try self.readRegister16(INA219_SHUNT_VOLTAGE_REG);

        // The shunt voltage is a signed value in 10µV units
        const signed_value: i16 = @bitCast(raw_voltage);

        // Convert to volts
        return @as(f32, @floatFromInt(signed_value)) * 0.00001;
    }

    // Add current and power reading functions if needed
};

pub const UDP_Provider = struct {
    imu: ?Mpu9250,
    ina219: ?Ina219,
    socket: std.posix.socket_t,
    dest_addr: std.posix.sockaddr,
    dest_addr_len: std.posix.socklen_t,
    start_time: i64,
    packet_count: u32 = 0,
    last_stats_time: i64,

    const Self = @This();

    pub fn init(server_ip: []const u8, server_port: u16) !Self {
        const current_time = std.time.milliTimestamp();
        const socket = try std.posix.socket(
            std.posix.AF.INET,
            std.posix.SOCK.DGRAM,
            0,
        );

        const dest_addr = (try std.net.Address.parseIp4(server_ip, server_port)).any;

        const imu = try Mpu9250.init();
        const ina219 = try Ina219.init();

        return Self{
            .imu = imu,
            .ina219 = ina219,
            .socket = socket,
            .dest_addr = dest_addr,
            .dest_addr_len = @sizeOf(std.posix.sockaddr),
            .start_time = current_time,
            .last_stats_time = current_time,
            .packet_count = 0,
        };
    }

    pub fn sendBatteryCommand(battery_voltage: f32, server_ip: []const u8, server_port: u16) !void {
        // Create a UDP socket for sending commands
        const socket = try std.posix.socket(
            std.posix.AF.INET,
            std.posix.SOCK.DGRAM,
            0,
        );
        defer std.posix.close(socket);

        // Set up destination address (motor controller server)
        const dest_addr = (try std.net.Address.parseIp4(server_ip, server_port)).any;

        // Format battery command
        var cmd_buf: [64]u8 = undefined;
        const cmd_len = try std.fmt.bufPrint(&cmd_buf, "Battery {d:.3}", .{battery_voltage});

        // Send command
        _ = try std.posix.sendto(
            socket,
            cmd_len,
            0,
            &dest_addr,
            @sizeOf(std.posix.sockaddr),
        );
    }

    pub fn send(self: *Self) !void {
        const current_time = std.time.milliTimestamp();
        var packet = std.mem.zeroes(ReadingsPacket);
        if (self.imu) |*imu| {
            packet = imu.read() catch |err| blk: {
                std.debug.print("Error reading from IMU: {any}, continuing...\n", .{err});
                break :blk std.mem.zeroes(ReadingsPacket);
            };
        }

        // Add timestamp
        std.mem.writeInt(i64, packet[36..44], current_time * 1000, .little);

        _ = try std.posix.sendto(
            self.socket,
            &packet,
            0,
            &self.dest_addr,
            self.dest_addr_len,
        );

        // Update stats
        self.packet_count += 1;

        // Print stats every second without blocking
        if (current_time - self.last_stats_time >= 1000) {
            if (self.ina219) |ina219| {
                const bus_voltage = ina219.readBusVoltage() catch |err| blk: {
                    std.debug.print("Error reading bus voltage: {any}\n", .{err});
                    break :blk 0.0;
                };

                const shunt_voltage = ina219.readShuntVoltage() catch |err| blk: {
                    std.debug.print("Error reading shunt voltage: {any}\n", .{err});
                    break :blk 0.0;
                };

                std.debug.print("Battery: {d:.3} V, Shunt: {d:.6} V\n", .{ bus_voltage, shunt_voltage });

                sendBatteryCommand(bus_voltage, "10.42.0.219", 5000) catch |err| {
                    std.debug.print("Failed to send battery command: {any}\n", .{err});
                };
            } else {
                std.debug.print("INA219 not available\n", .{});
            }

            const rate = @as(f32, @floatFromInt(self.packet_count)) /
                (@as(f32, @floatFromInt(current_time - self.last_stats_time)) / 1000.0);

            std.debug.print("Rate: {d:.1} Hz\n", .{rate});

            // Status update about sensors
            if (self.imu == null) {
                std.debug.print("IMU not connected\n", .{});
            }

            // Reset counters
            self.packet_count = 0;
            self.last_stats_time = current_time;
        }
    }

    pub fn run(self: *Self) !void {
        if (self.imu == null and self.ina219 == null) {
            std.debug.print("No sensors detected. Exiting...\n", .{});
            return error.NoSensorsDetected;
        }

        std.debug.print("Starting UDP Provider with:\n", .{});
        if (self.imu != null) std.debug.print("- MPU9250 IMU active\n", .{});
        if (self.ina219 != null) std.debug.print("- INA219 voltage sensor active\n", .{});

        while (true) {
            try self.send();
        }
    }

    pub fn deinit(self: *Self) void {
        if (self.imu) |imu| {
            imu.deinit();
        }

        if (self.ina219) |ina219| {
            ina219.deinit();
        }

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
