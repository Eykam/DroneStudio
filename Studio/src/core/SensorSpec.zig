//! dronestudio.sensor/1 - spec-driven sensor model loader (sim consumer).
//! One JSON per sensor part in sensors/; the sim reads ONLY the `dynamics`
//! block. physical/orientation/procurement/ee blocks are consumed by CAD and
//! EE tooling and are ignored here (ignore_unknown_fields).
//! Env-gated like ChassisManifest: DRONE_IMU_SPEC / DRONE_TOF_SPEC point at a
//! spec file; unset = compiled-in defaults, zero behavior change.

const std = @import("std");

pub const GyroDynamics = struct {
    noise_density_rad_per_s_rthz: f64,
    bias_walk_rad_per_s_rts: f64 = 2e-5,
    range_dps: f64 = 2000,
};

pub const AccelDynamics = struct {
    noise_density_m_per_s2_rthz: f64,
    bias_walk_m_per_s2_rts: f64 = 5e-4,
    range_g: f64 = 16,
};

pub const ImuDynamics = struct {
    sample_rate_hz: u32 = 1000,
    gyro: GyroDynamics,
    accel: AccelDynamics,
};

pub const TofDynamics = struct {
    max_range_m: f64 = 9.0,
    min_range_m: f64 = 0.05,
    sigma_base_mm: f64 = 5.0,
    sigma_k_mm_per_m2: f64 = 3.0,
    ambient_factor: f64 = 1.0,
    base_dropout: f64 = 0.01,
    far_dropout_start: f64 = 0.8,
    far_dropout_max: f64 = 0.5,
    graze_cos: f64 = 0.3,
    arm_block_p: f64 = 0.56,
    arm_lo_mm: f64 = 40.0,
    arm_mid_mm: f64 = 100.0,
    arm_hi_mm: f64 = 140.0,
};

pub const Dynamics = struct {
    imu: ?ImuDynamics = null,
    tof: ?TofDynamics = null,
};

pub const SensorSpec = struct {
    schema: []const u8,
    part_id: []const u8,
    category: []const u8,
    dynamics: ?Dynamics = null,

    pub fn load(alloc: std.mem.Allocator, path: []const u8) !std.json.Parsed(SensorSpec) {
        const bytes = try std.fs.cwd().readFileAlloc(alloc, path, 1 << 20);
        defer alloc.free(bytes);
        return std.json.parseFromSlice(SensorSpec, alloc, bytes, .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        });
    }
};
