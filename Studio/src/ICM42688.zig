//! Hardware driver for the user-picked MPU-9250 replacement pair (2026-09-12):
//!   - Icm42688p: TDK ICM-42688-P 6-axis IMU (accel+gyro), OSP-IC-0009
//!   - Mmc5983ma: MEMSIC MMC5983MA 3-axis magnetometer, OSP-SEN-0001
//! Both on the shared I2C bus (CONTRACTS.md amendment v1.1).
//! Register maps grounded in: TDK DS-000347 v1.6 (corroborated by the PX4
//! ICM42688P register header), MEMSIC MMC5983MA Rev A (corroborated by the
//! Linux kernel mmc5983 driver and bluerobotics datasheet mirror).
//! WHO_AM_I / PRODUCT_ID are checked at init, so a register-map mistake
//! fails loudly on real hardware instead of reading garbage.

const std = @import("std");
const I2C = @import("core/I2C.zig");

const I2C_SLAVE: u32 = 0x0703;
const I2C_PATH = "/dev/i2c-1";

pub const Readings = struct { x: f32, y: f32, z: f32 };

fn writeByte(fd: i32, addr: u8, reg: u8, value: u8) !void {
    if (std.os.linux.ioctl(fd, I2C_SLAVE, addr) < 0) {
        return error.I2CSlaveSelectFailed;
    }
    const buf = [_]u8{ reg, value };
    if (try std.posix.write(fd, &buf) != 2) {
        return error.I2CWriteFailed;
    }
}

fn readByte(fd: i32, addr: u8, reg: u8) !u8 {
    var b: [1]u8 = undefined;
    try I2C.readBlock(fd, addr, reg, &b);
    return b[0];
}

// ============================ ICM-42688-P ============================

pub const Icm42688p = struct {
    pub const ADDR: u8 = 0x68; // alt 0x69 (AD0 high)

    // Bank 0 registers (REG_BANK_SEL = 0x00; we never leave bank 0)
    const REG_DEVICE_CONFIG: u8 = 0x11;
    const REG_PWR_MGMT0: u8 = 0x4E;
    const REG_GYRO_CONFIG0: u8 = 0x4F;
    const REG_ACCEL_CONFIG0: u8 = 0x50;
    const REG_ACCEL_DATA_X1: u8 = 0x1F; // 12-byte burst: accel xyz + gyro xyz
    const REG_WHO_AM_I: u8 = 0x75;
    const REG_BANK_SEL: u8 = 0x76;
    const WHOAMI_VALUE: u8 = 0x47;

    // Config encoding (DS-000347):
    // GYRO_FS_SEL[7:5]: 000=2000dps ... ; GYRO_ODR[3:0]: 0110=1kHz.
    // ACCEL_FS_SEL[7:5]: 00=16g; ACCEL_ODR[3:0]: 0110=1kHz.
    const GYRO_CFG: u8 = (0b000 << 5) | 0b0110; // +/-2000 dps, 1 kHz
    const ACCEL_CFG: u8 = (0b00 << 5) | 0b0110; // +/-16 g, 1 kHz
    const PWR_LOW_NOISE_BOTH: u8 = 0x0F; // gyro+accel Low-Noise mode

    const GYRO_LSB_PER_DPS: f32 = 16.4; // +/-2000 dps
    const ACCEL_LSB_PER_G: f32 = 2048.0; // +/-16 g
    const DEG2RAD: f32 = std.math.pi / 180.0;
    const G_TO_MS2: f32 = 9.80665;

    i2c_fd: i32,

    const Self = @This();

    /// Probe, verify WHO_AM_I, soft-reset, configure. Returns null when the
    /// part is absent (mirrors IMU.zig init semantics).
    pub fn init() !?Self {
        const fd = I2C.openI2C(I2C_PATH) catch |err| {
            std.debug.print("Failed to open I2C for ICM-42688-P: {any}\n", .{err});
            return err;
        };
        if (!I2C.isDevicePresent(fd, ADDR, I2C_SLAVE)) {
            std.debug.print("ICM-42688-P not detected on I2C bus\n", .{});
            _ = std.posix.close(fd);
            return null;
        }
        const self = Self{ .i2c_fd = fd };
        try self.initSensor();
        return self;
    }

    fn initSensor(self: Self) !void {
        try writeByte(self.i2c_fd, ADDR, REG_BANK_SEL, 0x00);
        const who = try readByte(self.i2c_fd, ADDR, REG_WHO_AM_I);
        if (who != WHOAMI_VALUE) {
            std.debug.print("ICM-42688-P WHO_AM_I mismatch: got 0x{x}, want 0x{x}\n", .{ who, WHOAMI_VALUE });
            return error.WhoAmIMismatch;
        }
        // Soft reset, then wait for the part to come back (DS: 1ms typ).
        try writeByte(self.i2c_fd, ADDR, REG_DEVICE_CONFIG, 0x01);
        std.Thread.sleep(2 * std.time.ns_per_ms);
        try writeByte(self.i2c_fd, ADDR, REG_BANK_SEL, 0x00);
        // Re-verify after reset.
        const who2 = try readByte(self.i2c_fd, ADDR, REG_WHO_AM_I);
        if (who2 != WHOAMI_VALUE) return error.WhoAmIMismatch;
        // NOTE (design note from the part selection): the ICM-42688-P wants a
        // clean low-noise 3.3V rail; layout/decoupling care lives EE-side.
        try writeByte(self.i2c_fd, ADDR, REG_GYRO_CONFIG0, GYRO_CFG);
        try writeByte(self.i2c_fd, ADDR, REG_ACCEL_CONFIG0, ACCEL_CFG);
        try writeByte(self.i2c_fd, ADDR, REG_PWR_MGMT0, PWR_LOW_NOISE_BOTH);
        // Gyro/accel need ~30-45ms from standby to LN mode before data is valid.
        std.Thread.sleep(50 * std.time.ns_per_ms);
    }

    /// One 12-byte burst: accel xyz (m/s^2) + gyro xyz (rad/s), SI units.
    /// FIRMWARE NOTE (ICM-42688-P gyro spike artefact, Betaflight #12970):
    /// this part can emit rare single-sample gyro spikes; the consumer
    /// (estimator) should apply outlier rejection - the driver reports raw
    /// values honestly and does not hide them.
    pub fn readAccelGyro(self: Self) !struct { accel: Readings, gyro: Readings } {
        var buf: [12]u8 = undefined;
        try I2C.readBlock(self.i2c_fd, ADDR, REG_ACCEL_DATA_X1, &buf);
        const ax: i16 = @intCast((@as(u16, buf[0]) << 8) | buf[1]);
        const ay: i16 = @intCast((@as(u16, buf[2]) << 8) | buf[3]);
        const az: i16 = @intCast((@as(u16, buf[4]) << 8) | buf[5]);
        const gx: i16 = @intCast((@as(u16, buf[6]) << 8) | buf[7]);
        const gy: i16 = @intCast((@as(u16, buf[8]) << 8) | buf[9]);
        const gz: i16 = @intCast((@as(u16, buf[10]) << 8) | buf[11]);
        const f: f32 = G_TO_MS2 / ACCEL_LSB_PER_G;
        const g: f32 = DEG2RAD / GYRO_LSB_PER_DPS;
        return .{
            .accel = .{ .x = @as(f32, @floatFromInt(ax)) * f, .y = @as(f32, @floatFromInt(ay)) * f, .z = @as(f32, @floatFromInt(az)) * f },
            .gyro = .{ .x = @as(f32, @floatFromInt(gx)) * g, .y = @as(f32, @floatFromInt(gy)) * g, .z = @as(f32, @floatFromInt(gz)) * g },
        };
    }

    pub fn deinit(self: Self) void {
        _ = std.posix.close(self.i2c_fd);
    }
};

// ============================ MMC5983MA ============================

pub const Mmc5983ma = struct {
    pub const ADDR: u8 = 0x30;

    const REG_XOUT0: u8 = 0x00; // 7-byte burst: X(2) Y(2) Z(2) packed XYZ2
    const REG_TOUT: u8 = 0x07;
    const REG_STATUS: u8 = 0x08;
    const REG_CTRL0: u8 = 0x09;
    const REG_CTRL1: u8 = 0x0A;
    const REG_CTRL2: u8 = 0x0B;
    const REG_PRODUCT_ID: u8 = 0x2F;
    const PRODUCT_ID_VALUE: u8 = 0x30;

    // CTRL0 bits
    const CTRL0_TM_M: u8 = 1 << 0; // take magnetic measurement (auto-clears)
    const CTRL0_SET: u8 = 1 << 3; // SET pulse (degauss)
    const CTRL0_RESET: u8 = 1 << 4; // RESET pulse (degauss)
    // CTRL1 bits
    const CTRL1_SW_RST: u8 = 1 << 7;
    const CTRL1_BW_100HZ: u8 = 0b00; // BW=00: 8ms measurement, ~100Hz max ODR
    // STATUS bits
    const STATUS_MEAS_M_DONE: u8 = 1 << 0;

    // +/-8 G FSR, 18-bit offset-binary output: null field at 2^17.
    const COUNTS_PER_GAUSS: f32 = 16384.0;
    const NULL_FIELD: f32 = 131072.0;
    const GAUSS_TO_UT: f32 = 100.0;

    i2c_fd: i32,

    const Self = @This();

    pub fn init() !?Self {
        const fd = I2C.openI2C(I2C_PATH) catch |err| {
            std.debug.print("Failed to open I2C for MMC5983MA: {any}\n", .{err});
            return err;
        };
        if (!I2C.isDevicePresent(fd, ADDR, I2C_SLAVE)) {
            std.debug.print("MMC5983MA not detected on I2C bus\n", .{});
            _ = std.posix.close(fd);
            return null;
        }
        const self = Self{ .i2c_fd = fd };
        try self.initSensor();
        return self;
    }

    fn initSensor(self: Self) !void {
        const pid = try readByte(self.i2c_fd, ADDR, REG_PRODUCT_ID);
        if (pid != PRODUCT_ID_VALUE) {
            std.debug.print("MMC5983MA PRODUCT_ID mismatch: got 0x{x}, want 0x{x}\n", .{ pid, PRODUCT_ID_VALUE });
            return error.ProductIdMismatch;
        }
        // Soft reset, then BW=00 (best noise: 0.4 mG RMS per datasheet).
        try writeByte(self.i2c_fd, ADDR, REG_CTRL1, CTRL1_SW_RST);
        std.Thread.sleep(10 * std.time.ns_per_ms);
        try writeByte(self.i2c_fd, ADDR, REG_CTRL1, CTRL1_BW_100HZ);
    }

    /// One magnetic measurement in uT (offset-binary -> signed field).
    /// Single-shot path: TM_M, poll MEAS_M_DONE, burst-read 7 bytes.
    pub fn readMag(self: Self) !Readings {
        try writeByte(self.i2c_fd, ADDR, REG_CTRL0, CTRL0_TM_M);
        var tries: u32 = 0;
        while (tries < 100) : (tries += 1) {
            const st = try readByte(self.i2c_fd, ADDR, REG_STATUS);
            if (st & STATUS_MEAS_M_DONE != 0) break;
            std.Thread.sleep(1 * std.time.ns_per_ms);
        } else {
            return error.MeasurementTimeout;
        }
        var b: [7]u8 = undefined;
        try I2C.readBlock(self.i2c_fd, ADDR, REG_XOUT0, &b);
        const x: u32 = (@as(u32, b[0]) << 10) | (@as(u32, b[1]) << 2) | (@as(u32, b[6]) >> 6);
        const y: u32 = (@as(u32, b[2]) << 10) | (@as(u32, b[3]) << 2) | ((@as(u32, b[6]) >> 4) & 0x3);
        const z: u32 = (@as(u32, b[4]) << 10) | (@as(u32, b[5]) << 2) | ((@as(u32, b[6]) >> 2) & 0x3);
        const s: f32 = GAUSS_TO_UT / COUNTS_PER_GAUSS;
        return .{
            .x = (@as(f32, @floatFromInt(x)) - NULL_FIELD) * s,
            .y = (@as(f32, @floatFromInt(y)) - NULL_FIELD) * s,
            .z = (@as(f32, @floatFromInt(z)) - NULL_FIELD) * s,
        };
    }

    /// Degauss (SET/RESET): clears residual magnetization and null-field
    /// offset temp drift. Per the design note carried with the part pick,
    /// call periodically or on significant temperature change.
    pub fn degaussSet(self: Self) !void {
        try writeByte(self.i2c_fd, ADDR, REG_CTRL0, CTRL0_SET);
        std.Thread.sleep(1 * std.time.ns_per_ms);
    }

    pub fn degaussReset(self: Self) !void {
        try writeByte(self.i2c_fd, ADDR, REG_CTRL0, CTRL0_RESET);
        std.Thread.sleep(1 * std.time.ns_per_ms);
    }

    /// Offset-corrected read: SET pulse + read, RESET pulse + read, average.
    /// The average cancels both sensor offset and magnetization remnance.
    pub fn readMagDegaussed(self: Self) !Readings {
        try self.degaussSet();
        const a = try self.readMag();
        try self.degaussReset();
        const b = try self.readMag();
        return .{ .x = (a.x + b.x) / 2.0, .y = (a.y + b.y) / 2.0, .z = (a.z + b.z) / 2.0 };
    }

    pub fn deinit(self: Self) void {
        _ = std.posix.close(self.i2c_fd);
    }
};
