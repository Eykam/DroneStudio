//! dronestudio.chassis/1.2 manifest loader (sim side, auto-researcher branch).
//! Ported from the CAD teams chassis/sim/ChassisManifest.zig (1.1 draft) and
//! extended for schema 1.2: imu rotation/offset-from-com and cameras[].
//! Unknown fields are ignored, so 1.1 manifests still parse.
//! Consumers: prefabs/Drone.zig (dynamics + IMU pose), render cameras (FOV/poses).

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

/// Schema 1.2: camera lens pose + FOV in the GLB frame (+X fwd, +Z up).
pub const Camera = struct {
    id: []const u8,
    lens_origin_m: Vec3,
    lens_axis: Vec3,
    hfov_deg: f64,
    vfov_deg: f64,
};

pub const ChassisManifest = struct {
    schema: []const u8,
    name: []const u8,
    geometry: struct {
        file: []const u8,
        units: []const u8 = "meters",
        forward: []const u8 = "+X",
        up: []const u8 = "+Z",
        // CAD pipeline emits <variant>.sim.glb alongside the pretty GLB:
        // uncompressed (GLTF.zig-compatible), Z-up -> Y-up baked in.
        sim_file: ?[]const u8 = null,
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
    /// Schema 1.2 IMU mount. rotation_quat_xyzw is the accel/gyro die frame;
    /// the AK8963 mag die is rotated in-package - estimator applies the mag remap.
    imu: struct {
        position_m: Vec3,
        rotation_quat_xyzw: [4]f64 = .{ 0, 0, 0, 1 },
        offset_from_com_m: ?Vec3 = null,
        note: ?[]const u8 = null,
    },
    /// Schema 1.2 camera list (absent in 1.1 -> empty slice).
    cameras: []const Camera = &.{},
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
    /// GLB to load for simulation: the uncompressed Y-up sim variant when
    /// the CAD pipeline emitted one, else the primary geometry file.
    pub fn simGeometryFile(self: *const ChassisManifest) []const u8 {
        return self.geometry.sim_file orelse self.geometry.file;
    }

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

    /// IMU offset from CoM, meters. Uses the manifests offset_from_com_m when
    /// present (1.2), else derives position_m - com_m (1.1-compatible).
    pub fn imuOffsetFromComM(self: *const ChassisManifest) [3]f32 {
        const o = self.imu.offset_from_com_m orelse blk: {
            const p = self.imu.position_m;
            const c = self.dynamics.com_m;
            break :blk Vec3{ p[0] - c[0], p[1] - c[1], p[2] - c[2] };
        };
        return .{ @floatCast(o[0]), @floatCast(o[1]), @floatCast(o[2]) };
    }

    /// IMU mount orientation quaternion (x, y, z, w), body->sensor frame.
    pub fn imuRotationQuatXyzw(self: *const ChassisManifest) [4]f32 {
        const q = self.imu.rotation_quat_xyzw;
        return .{ @floatCast(q[0]), @floatCast(q[1]), @floatCast(q[2]), @floatCast(q[3]) };
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

test "parse 1.2 manifest: imu offset/quat + cameras" {
    const json =
        \\{
        \\ "schema": "dronestudio.chassis/1.2",
        \\ "name": "v22-g21a",
        \\ "geometry": {"file": "chassis.glb", "sim_file": "chassis.sim.glb"},
        \\ "material": {"name": "pla-cf", "density_kg_m3": 1240},
        \\ "dynamics": {"total_mass_kg": 1.1, "com_m": [-0.0257, 0.0, 0.0099],
        \\   "inertia_about_com_kgm2": {"ixx": 0.01, "iyy": 0.01, "izz": 0.02, "ixy": 0, "ixz": 0, "iyz": 0}},
        \\ "collision": {"type": "convex_hull"},
        \\ "motors": [{"id": 0, "position_m": [0.1, 0.1, 0.02], "axis": [0,0,1], "direction": "cw",
        \\   "mass_kg": 0.05, "max_thrust_n": 8.0, "time_constant_s": 0.04, "drag_ratio": 0.15}],
        \\ "imu": {"position_m": [0, 0, 0.028], "rotation_quat_xyzw": [0, 0, 0, 1],
        \\   "offset_from_com_m": [0.025739, -0.000001, 0.018061]},
        \\ "cameras": [{"id": "pi_camera_3#left", "lens_origin_m": [0.085, -0.028, 0.0145],
        \\   "lens_axis": [1, 0, 0], "hfov_deg": 66.3, "vfov_deg": 41.6}],
        \\ "stack": {"pattern_mm": 30.5, "hole_dia_mm": 3.2, "z_bottom_m": 0.01}
        \\}
    ;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "m.json", .data = json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "m.json");
    defer std.testing.allocator.free(path);
    const parsed = try ChassisManifest.load(std.testing.allocator, path);
    defer parsed.deinit();
    const m = parsed.value;
    try std.testing.expectEqualStrings("dronestudio.chassis/1.2", m.schema);
    try std.testing.expectEqual(@as(usize, 1), m.cameras.len);
    try std.testing.expectApproxEqAbs(@as(f32, 66.3), @as(f32, @floatCast(m.cameras[0].hfov_deg)), 1e-4);
    const off = m.imuOffsetFromComM();
    try std.testing.expectApproxEqAbs(@as(f32, 0.025739), off[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.018061), off[2], 1e-6);
    try std.testing.expectEqualStrings("chassis.sim.glb", m.simGeometryFile());
}
