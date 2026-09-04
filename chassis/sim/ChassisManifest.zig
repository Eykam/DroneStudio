//! dronestudio.chassis/1.1 manifest loader.
//! Draft for the chassis branch: replaces the hardcoded mass/inertia/motor
//! values in prefabs/Drone.zig with values authored by the CAD auto-researcher.
//! v1.1: reads the composed `dynamics` block directly - no derivation in the sim.
//! Compile-checked on the CAD box (zig build -Dcuda=false -Dpi=false) before merge.

const std = @import("std");

pub const Vec3 = [3]f64;
pub const InertiaTensor = struct { ixx: f64, iyy: f64, izz: f64, ixy: f64, ixz: f64, iyz: f64 };

pub const Motor = struct {
    id: u32,
    position_m: Vec3,
    axis: Vec3,
    direction: []const u8, // "cw" | "ccw"
    mass_kg: f64,
    prop_diameter_m: f64 = 0.127,
    max_thrust_n: f64,
    time_constant_s: f64,
    drag_ratio: f64,
};

pub const ChassisManifest = struct {
    schema: []const u8,
    name: []const u8,
    geometry: struct {
        file: []const u8,
        units: []const u8 = "meters",
        forward: []const u8 = "+X",
        up: []const u8 = "+Z",
    },
    material: struct { name: []const u8, density_kg_m3: f64, e_mpa: f64 = 0, yield_mpa: f64 = 0 },
    dynamics: struct {
        total_mass_kg: f64,
        com_m: Vec3,
        inertia_about_com_kgm2: InertiaTensor,
        composition: ?[]const struct {
            name: []const u8,
            mass_kg: f64,
            com_m: Vec3,
        } = null,
        note: ?[]const u8 = null,
    },
    aero: ?struct {
        projected_area_m2: struct { x: f64, y: f64, z: f64 },
        cd_flat_plate_estimate: f64 = 1.1,
        note: ?[]const u8 = null,
    } = null,
    collision: struct {
        type: []const u8, // "convex_hull" | "vhacd"
        fallback: ?[]const u8 = null,
        max_hulls: ?u32 = null,
    },
    motors: []Motor,
    imu: struct { position_m: Vec3, note: ?[]const u8 = null },
    stack: struct { pattern_mm: f64, hole_dia_mm: f64, z_bottom_m: f64 },

    pub fn load(alloc: std.mem.Allocator, path: []const u8) !std.json.Parsed(ChassisManifest) {
        const bytes = try std.fs.cwd().readFileAlloc(alloc, path, 1 << 20);
        defer alloc.free(bytes);
        return std.json.parseFromSlice(ChassisManifest, alloc, bytes, .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        });
    }

    /// Total rigid-body mass (frame + motors + payload), kg.
    pub fn totalMassKg(self: *const ChassisManifest) f32 {
        return @floatCast(self.dynamics.total_mass_kg);
    }

    /// Diagonal inertia about the composed CoM, kg*m^2 -> RigidBodyComponent.setInertia.
    pub fn diagonalInertia(self: *const ChassisManifest) [3]f32 {
        const t = self.dynamics.inertia_about_com_kgm2;
        return .{ @floatCast(t.ixx), @floatCast(t.iyy), @floatCast(t.izz) };
    }

    /// Center of mass (sim should offset the body origin by this), meters.
    pub fn comM(self: *const ChassisManifest) [3]f32 {
        const c = self.dynamics.com_m;
        return .{ @floatCast(c[0]), @floatCast(c[1]), @floatCast(c[2]) };
    }

    /// Center -> motor axis distance (quad symmetric), meters. Feeds FlightController.motor_arm_length.
    pub fn armLengthM(self: *const ChassisManifest) f32 {
        if (self.motors.len == 0) return 0;
        const p = self.motors[0].position_m;
        return @floatCast(@sqrt(p[0] * p[0] + p[1] * p[1]));
    }

    pub fn maxThrustPerMotorN(self: *const ChassisManifest) f32 {
        if (self.motors.len == 0) return 0;
        return @floatCast(self.motors[0].max_thrust_n);
    }

    pub fn motorTimeConstantS(self: *const ChassisManifest) f32 {
        if (self.motors.len == 0) return 0.04;
        return @floatCast(self.motors[0].time_constant_s);
    }

    pub fn motorDragRatio(self: *const ChassisManifest) f32 {
        if (self.motors.len == 0) return 0.15;
        return @floatCast(self.motors[0].drag_ratio);
    }
};
