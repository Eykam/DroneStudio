//! dronestudio.chassis/1 manifest loader.
//! Draft for the chassis branch: replaces the hardcoded mass/inertia/motor
//! values in prefabs/Drone.zig with values authored by the CAD auto-researcher.
//! Compile-checked on the CAD box (zig build -Dcuda=false -Dpi=false) before merge.

const std = @import("std");

pub const Motor = struct {
    id: u32,
    position_m: [3]f64,
    axis: [3]f64,
    direction: []const u8, // "cw" | "ccw"
    mass_kg: f64,
    max_thrust_n: f64,
    time_constant_s: f64,
    drag_ratio: f64,
};

pub const InertiaTriplet = struct { ixx: f64, iyy: f64, izz: f64 };
pub const InertiaTensor = struct { ixx: f64, iyy: f64, izz: f64, ixy: f64, ixz: f64, iyz: f64 };

pub const ChassisManifest = struct {
    schema: []const u8,
    name: []const u8,
    geometry: struct {
        file: []const u8,
        units: []const u8 = "meters",
        forward: []const u8 = "+X",
        up: []const u8 = "+Z",
    },
    material: struct { name: []const u8, density_kg_m3: f64 },
    inertial: struct {
        frame_mass_kg: f64,
        frame_com_m: [3]f64,
        frame_inertia_kgm2: InertiaTensor,
        motor_inertia_add_kgm2: InertiaTriplet,
        note: ?[]const u8 = null,
    },
    collision: struct {
        type: []const u8, // "convex_hull" | "vhacd"
        fallback: ?[]const u8 = null,
        max_hulls: ?u32 = null,
    },
    motors: []Motor,
    imu: struct { position_m: [3]f64 },
    stack: struct { pattern_mm: f64, hole_dia_mm: f64, z_bottom_m: f64 },

    pub fn load(alloc: std.mem.Allocator, path: []const u8) !std.json.Parsed(ChassisManifest) {
        const bytes = try std.fs.cwd().readFileAlloc(alloc, path, 1 << 20);
        defer alloc.free(bytes);
        return std.json.parseFromSlice(ChassisManifest, alloc, bytes, .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        });
    }

    /// Frame + mounted motors. Payload (battery/stack/cameras) is composed by the caller.
    pub fn totalMassKg(self: *const ChassisManifest) f32 {
        var m: f64 = self.inertial.frame_mass_kg;
        for (self.motors) |mot| m += mot.mass_kg;
        return @floatCast(m);
    }

    /// Diagonal inertia (frame full tensor diagonal + motor point-mass adds), kg*m^2.
    pub fn diagonalInertia(self: *const ChassisManifest) [3]f32 {
        const f = self.inertial.frame_inertia_kgm2;
        const a = self.inertial.motor_inertia_add_kgm2;
        return .{ @floatCast(f.ixx + a.ixx), @floatCast(f.iyy + a.iyy), @floatCast(f.izz + a.izz) };
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
