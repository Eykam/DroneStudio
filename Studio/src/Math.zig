const std = @import("std");
const math = std.math;
const Sensors = @import("Sensors.zig");
const SensorState = Sensors.SensorState;
const TypeId = std.builtin.TypeId;

fn isNumericType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .float => true,
        else => false,
    };
}

pub fn Vec3(comptime T: type) type {
    // Verify T is a numeric type
    comptime {
        const type_id = @typeInfo(T);
        switch (type_id) {
            .int, .float => {},
            else => @compileError("Vec3 requires a numeric type (integer or float), got " ++ @typeName(T)),
        }
    }

    const can_use_simd = switch (@typeInfo(T)) {
        .float => true,
        .int => |int_info| int_info.bits <= 32,
        else => false,
    };

    return struct {
        const Self = @This();
        const SimdVector = if (can_use_simd) @Vector(4, T) else void;

        // Fields
        x: T,
        y: T,
        z: T,
        simd: if (can_use_simd) SimdVector else void,

        // Initialize a new Vec3
        pub fn init(x: T, y: T, z: T) Self {
            if (can_use_simd) {
                return .{
                    .x = x,
                    .y = y,
                    .z = z,
                    .simd = SimdVector{ x, y, z, 0 },
                };
            } else {
                return .{
                    .x = x,
                    .y = y,
                    .z = z,
                    .simd = undefined,
                };
            }
        }

        // Create a zero vector
        pub fn zero() Self {
            return Self.init(0, 0, 0);
        }

        // Create a unit vector along a primary axis
        pub fn unit(axis: enum { x, y, z }) Self {
            return switch (axis) {
                .x => Self.init(1, 0, 0),
                .y => Self.init(0, 1, 0),
                .z => Self.init(0, 0, 1),
            };
        }

        // Add two vectors
        pub fn add(a: Self, b: Self) Self {
            if (can_use_simd) {
                const result = a.simd + b.simd;
                return .{
                    .x = result[0],
                    .y = result[1],
                    .z = result[2],
                    .simd = result,
                };
            }
            return Self.init(
                a.x + b.x,
                a.y + b.y,
                a.z + b.z,
            );
        }

        // Subtract two vectors
        pub fn sub(a: Self, b: Self) Self {
            if (can_use_simd) {
                const result = a.simd - b.simd;
                return .{
                    .x = result[0],
                    .y = result[1],
                    .z = result[2],
                    .simd = result,
                };
            }
            return Self.init(
                a.x - b.x,
                a.y - b.y,
                a.z - b.z,
            );
        }

        // Scale vector by scalar
        pub fn scale(self: Self, scalar: T) Self {
            if (can_use_simd) {
                const scale_vec = @as(scalar, @splat(4));
                const result = self.simd * scale_vec;
                return .{
                    .x = result[0],
                    .y = result[1],
                    .z = result[2],
                    .simd = result,
                };
            }
            return Self.init(
                self.x * scalar,
                self.y * scalar,
                self.z * scalar,
            );
        }

        // Dot product
        pub fn dot(a: Self, b: Self) T {
            if (can_use_simd) {
                const prod = a.simd * b.simd;
                var sum: T = 0;

                inline for (0..3) |i| {
                    sum += prod[i];
                }
                return sum;
            }
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        // Cross product
        pub fn cross(a: Self, b: Self) Self {
            if (can_use_simd) {
                // Use SIMD shuffle operations for cross product
                const a_yzx = @shuffle(T, a.simd, undefined, @Vector(4, i32){ 1, 2, 0, 3 });
                const b_yzx = @shuffle(T, b.simd, undefined, @Vector(4, i32){ 1, 2, 0, 3 });
                const a_zxy = @shuffle(T, a.simd, undefined, @Vector(4, i32){ 2, 0, 1, 3 });
                const b_zxy = @shuffle(T, b.simd, undefined, @Vector(4, i32){ 2, 0, 1, 3 });
                const result = (a_yzx * b_zxy) - (a_zxy * b_yzx);

                return .{
                    .x = result[0],
                    .y = result[1],
                    .z = result[2],
                    .simd = result,
                };
            }

            return Self.init(
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x,
            );
        }

        // Length squared
        pub fn lengthSquared(self: Self) T {
            return self.dot(self);
        }

        // Length
        pub fn length(self: Self) T {
            return switch (@typeInfo(T)) {
                .float => @sqrt(self.lengthSquared()),
                .int => @intFromFloat(@sqrt(@as(f64, @floatFromInt(self.lengthSquared())))),
                else => unreachable,
            };
        }

        // Normalize vector (only for floating point types)
        pub fn normalize(self: Self) Self {
            if (@typeInfo(T) != .float) {
                @compileError("normalize() is only available for floating point vectors");
            }

            const l = self.length();
            if (l == 0) {
                std.debug.print("Warning: Attempting to normalize zero vector\n", .{});
                return self;
            }
            return self.scale(1.0 / l);
        }

        // Distance between two points
        pub fn distance(a: Self, b: Self) T {
            return a.sub(b).length();
        }

        // Linear interpolation between vectors
        // pub fn lerp(a: Self, b: Self, t: T) Self {
        //     if (@typeInfo(T) != .Float) {
        //         @compileError("lerp() is only available for floating point vectors");
        //     }

        //     if (can_use_simd) {
        //         const t_vec = @as(T., @splat(4));
        //         const one_minus_t = @as@splat(4, @as(T, 1.0) - t);
        //         const result = (a.simd * one_minus_t) + (b.simd * t_vec);
        //         return .{
        //             .x = result[0],
        //             .y = result[1],
        //             .z = result[2],
        //             .simd = result,
        //         };
        //     }

        //     return a.scale(1 - t).add(b.scale(t));
        // }

        // Rotate vector around axis by angle (radians, only for floating point)
        pub fn rotate(self: Self, axis: Self, _angle: T) Self {
            if (@typeInfo(T) != .float) {
                @compileError("rotate() is only available for floating point vectors");
            }

            const normalized_axis = axis.normalize();
            const sin_angle = @sin(_angle);
            const cos_angle = @cos(_angle);
            const one_minus_cos = 1 - cos_angle;

            // Rodrigues rotation formula
            const dot_prod = self.dot(normalized_axis) * one_minus_cos;
            const cross_prod = normalized_axis.cross(self).scale(sin_angle);
            const parallel = normalized_axis.scale(dot_prod);
            const perpendicular = self.scale(cos_angle);

            return parallel.add(perpendicular).add(cross_prod);
        }

        // Angle between two vectors (in radians, only for floating point)
        pub fn angle(a: Self, b: Self) T {
            if (@typeInfo(T) != .float) {
                @compileError("angle() is only available for floating point vectors");
            }

            const dot_prod = a.dot(b);
            const lengths_prod = a.length() * b.length();
            if (lengths_prod == 0) return 0;

            // Clamp to avoid floating point errors
            const cos_theta = @max(@min(dot_prod / lengths_prod, 1), -1);
            return std.math.acos(cos_theta);
        }

        pub fn from_angles(yaw_deg: T, pitch_deg: T) Self {
            if (@typeInfo(T) != .float) {
                @compileError("from_angles() is only available for floating point vectors");
            }

            const yaw = radians(yaw_deg);
            const pitch = radians(pitch_deg);

            return Self.init(
                @cos(yaw) * @cos(pitch),
                @sin(pitch),
                @sin(yaw) * @cos(pitch),
            ).normalize();
        }

        // Convert to array
        pub fn toArray(self: Self) [3]T {
            return .{ self.x, self.y, self.z };
        }

        // Format for printing
        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("Vec3({d}, {d}, {d})", .{ self.x, self.y, self.z });
        }

        // Test whether two vectors are approximately equal (for floating point)
        pub fn approxEqual(a: Self, b: Self, tolerance: T) bool {
            if (@typeInfo(T) != .float) {
                @compileError("approxEqual() is only available for floating point vectors");
            }
            return math.approxEqAbs(T, a.x, b.x, tolerance) and
                math.approxEqAbs(T, a.y, b.y, tolerance) and
                math.approxEqAbs(T, a.z, b.z, tolerance);
        }
    };
}

pub fn Vec4(comptime T: type) type {
    if (!isNumericType(T)) {
        @compileError("Vec4 requires a numeric type (integer or float), got " ++ @typeName(T));
    }

    // Determine if we can use SIMD
    const can_use_simd = switch (@typeInfo(T)) {
        .float => true,
        .int => |int_info| int_info.bits <= 32,
        else => false,
    };

    const Vec4Vector = if (can_use_simd) @Vector(4, T) else void;

    return struct {
        const Self = @This();

        x: T,
        y: T,
        z: T,
        w: T,

        // Internal SIMD data if available
        simd_data: if (can_use_simd) Vec4Vector else void = undefined,

        pub fn init(x: T, y: T, z: T, w: T) Self {
            var result = Self{ .x = x, .y = y, .z = z, .w = w, .simd_data = undefined };
            if (can_use_simd) {
                result.simd_data = Vec4Vector{ x, y, z, w };
            }
            return result;
        }

        pub fn zero() Self {
            return Self.init(0, 0, 0, 0);
        }

        pub fn one() Self {
            return Self.init(1, 1, 1, 1);
        }

        pub fn add(a: Self, b: Self) Self {
            if (can_use_simd) {
                const result = a.simd_data + b.simd_data;
                return Self{
                    .x = result[0],
                    .y = result[1],
                    .z = result[2],
                    .w = result[3],
                    .simd_data = result,
                };
            }
            return Self.init(
                a.x + b.x,
                a.y + b.y,
                a.z + b.z,
                a.w + b.w,
            );
        }

        pub fn sub(a: Self, b: Self) Self {
            if (can_use_simd) {
                const result = a.simd_data - b.simd_data;
                return Self{
                    .x = result[0],
                    .y = result[1],
                    .z = result[2],
                    .w = result[3],
                    .simd_data = result,
                };
            }
            return Self.init(
                a.x - b.x,
                a.y - b.y,
                a.z - b.z,
                a.w - b.w,
            );
        }

        pub fn scale(self: Self, scalar: T) Self {
            if (can_use_simd) {
                const result = self.simd_data * @as(scalar, @splat(4));
                return Self{
                    .x = result[0],
                    .y = result[1],
                    .z = result[2],
                    .w = result[3],
                    .simd_data = result,
                };
            }
            return Self.init(
                self.x * scalar,
                self.y * scalar,
                self.z * scalar,
                self.w * scalar,
            );
        }

        pub fn dot(a: Self, b: Self) T {
            if (can_use_simd) {
                const prod = a.simd_data * b.simd_data;
                return @reduce(.Add, prod);
            }
            return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }

        pub fn length(self: Self) T {
            return switch (@typeInfo(T)) {
                .float => @sqrt(self.dot(self)),
                .int => @as(T, @intFromFloat(@sqrt(@as(f64, @floatFromInt(self.dot(self)))))),
                else => unreachable,
            };
        }

        pub fn normalize(self: Self) Self {
            if (@typeInfo(T) != .float) {
                @compileError("normalize() is only available for floating point vectors");
            }

            const l = self.length();
            if (l == 0) {
                std.debug.print("Warning: Attempting to normalize zero vector\n", .{});
                return self;
            }
            return self.scale(1.0 / l);
        }

        pub fn lerp(a: Self, b: Self, t: T) Self {
            if (@typeInfo(T) != .float) {
                @compileError("lerp() is only available for floating point vectors");
            }
            return a.scale(1 - t).add(b.scale(t));
        }

        pub fn toArray(self: Self) [4]T {
            return .{ self.x, self.y, self.z, self.w };
        }
    };
}

pub const Mat4 = [16]f32;

// Todo: Use @Vector instead for SIMD
pub const Quaternion = struct {
    x: f32,
    y: f32,
    z: f32,
    w: f32,

    pub fn identity() Quaternion {
        return Quaternion{ .w = 1, .x = 0, .y = 0, .z = 0 };
    }

    pub fn fromAxisAngle(axis: Vec3, angle: f32) Quaternion {
        const normalized_axis = axis.normalize();

        // Calculate the sine and cosine of half the angle
        const half_angle = angle * 0.5;
        const sin_half = @sin(half_angle);
        const cos_half = @cos(half_angle);

        return Quaternion{
            .x = normalized_axis.x * sin_half,
            .y = normalized_axis.y * sin_half,
            .z = normalized_axis.z * sin_half,
            .w = cos_half,
        };
    }

    pub fn fromEuler(pitch: f32, yaw: f32, roll: f32) Quaternion {
        const cy = @cos(yaw * 0.5);
        const sy = @sin(yaw * 0.5);
        const cp = @cos(pitch * 0.5);
        const sp = @sin(pitch * 0.5);
        const cr = @cos(roll * 0.5);
        const sr = @sin(roll * 0.5);

        return Quaternion{
            .w = cr * cp * cy + sr * sp * sy,
            .x = sr * cp * cy - cr * sp * sy,
            .y = cr * sp * cy + sr * cp * sy,
            .z = cr * cp * sy - sr * sp * cy,
        };
    }

    pub fn normalize(q: Quaternion) Quaternion {
        const mag = @sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);

        if (mag == 0) {
            std.debug.print("Quaternion with 0 Length Detected", .{});
            return q;
        }

        return Quaternion{
            .x = q.x / mag,
            .y = q.y / mag,
            .z = q.z / mag,
            .w = q.w / mag,
        };
    }

    pub fn toMatrix(q: Quaternion) Mat4 {
        const xx = q.x * q.x;
        const yy = q.y * q.y;
        const zz = q.z * q.z;
        const xy = q.x * q.y;
        const xz = q.x * q.z;
        const yz = q.y * q.z;
        const wx = q.w * q.x;
        const wy = q.w * q.y;
        const wz = q.w * q.z;

        return Mat4{
            1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),     0,
            2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),     0,
            2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy), 0,
            0,                 0,                 0,                 1,
        };
    }
    pub fn add(self: Quaternion, other: Quaternion) Quaternion {
        return Quaternion{
            .x = self.x + other.x,
            .y = self.y + other.y,
            .z = self.z + other.z,
            .w = self.w + other.w,
        };
    }

    pub fn multiply(a: Quaternion, b: Quaternion) Quaternion {
        const q = Quaternion{
            .w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            .x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            .y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            .z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        };

        return q;
    }

    pub fn scale(self: Quaternion, scalar: f32) Quaternion {
        return Quaternion{
            .x = self.x * scalar,
            .y = self.y * scalar,
            .z = self.z * scalar,
            .w = self.w * scalar,
        };
    }

    pub fn conjugate(self: Quaternion) Quaternion {
        return Quaternion{
            .w = self.w,
            .x = -self.x,
            .y = -self.y,
            .z = -self.z,
        };
    }

    pub fn slerp(a: Quaternion, b: Quaternion, t: f32) Quaternion {
        var cos_half_theta = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;

        var b_copy = b;
        if (cos_half_theta < 0.0) {
            b_copy = Quaternion{ .x = -b.x, .y = -b.y, .z = -b.z, .w = -b.w };
            cos_half_theta = -cos_half_theta;
        }

        if (cos_half_theta > 0.9995) {
            return Quaternion.normalize(Quaternion{
                .x = a.x + t * (b_copy.x - a.x),
                .y = a.y + t * (b_copy.y - a.y),
                .z = a.z + t * (b_copy.z - a.z),
                .w = a.w + t * (b_copy.w - a.w),
            });
        } else {
            const half_theta = std.math.acos(cos_half_theta);
            const sin_half_theta = @sin(half_theta);
            const ratio_a = @sin((1 - t) * half_theta) / sin_half_theta;
            const ratio_b = @sin(t * half_theta) / sin_half_theta;

            return Quaternion{
                .x = a.x * ratio_a + b_copy.x * ratio_b,
                .y = a.y * ratio_a + b_copy.y * ratio_b,
                .z = a.z * ratio_a + b_copy.z * ratio_b,
                .w = a.w * ratio_a + b_copy.w * ratio_b,
            };
        }
    }
};

pub fn identity() Mat4 {
    return .{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    };
}

pub fn rotate_x(angle_deg: f32) Mat4 {
    const angle_rad = angle_deg * (std.math.pi / 180.0);
    const c = @cos(angle_rad);
    const s = @sin(angle_rad);

    return .{
        1.0, 0.0, 0.0, 0.0,
        0.0, c,   -s,  0.0,
        0.0, s,   c,   0.0,
        0.0, 0.0, 0.0, 1.0,
    };
}

pub fn translate(matrix: Mat4, x: f32, y: f32, z: f32) Mat4 {
    // Create translation matrix
    var result = matrix;

    // Translation components go in the last column (indices 12,13,14)
    // In column-major order: matrix[12] = x, matrix[13] = y, matrix[14] = z
    result[12] = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    result[13] = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    result[14] = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
    result[15] = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];

    return result;
}

pub fn scale(matrix: Mat4, x: f32, y: f32, z: f32) Mat4 {
    var result = matrix;

    // Scale the basis vectors
    result[0] *= x;
    result[1] *= x;
    result[2] *= x;
    result[3] *= x;

    result[4] *= y;
    result[5] *= y;
    result[6] *= y;
    result[7] *= y;

    result[8] *= z;
    result[9] *= z;
    result[10] *= z;
    result[11] *= z;

    return result;
}

pub fn rotate_y(angle_deg: f32) Mat4 {
    const angle_rad = radians(angle_deg);
    const c = std.math.cos(angle_rad);
    const s = std.math.sin(angle_rad);

    return .{
        c,   0.0, s,   0.0,
        0.0, 1.0, 0.0, 0.0,
        -s,  0.0, c,   0.0,
        0.0, 0.0, 0.0, 1.0,
    };
}

pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) Mat4 {
    return .{
        2.0 / (right - left), 0.0,                  0.0,                 -(right + left) / (right - left),
        0.0,                  2.0 / (top - bottom), 0.0,                 -(top + bottom) / (top - bottom),
        0.0,                  0.0,                  -2.0 / (far - near), -(far + near) / (far - near),
        0.0,                  0.0,                  0.0,                 1.0,
    };
}

pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) Mat4 {
    const rad = radians(fov);
    const tan_half_fov = @tan(rad / 2.0);
    var mat: Mat4 = .{0} ** 16;
    mat[0] = 1.0 / (aspect * tan_half_fov);
    mat[5] = 1.0 / (tan_half_fov);
    mat[10] = -(far + near) / (far - near);
    mat[11] = -1.0;
    mat[14] = -(2.0 * far * near) / (far - near);
    return mat;
}

// Function to create a simple view matrix (camera at (0,0,5) looking at origin)
pub fn lookAt(eye: Vec3, center: Vec3, up: Vec3) Mat4 {
    const f = Vec3.normalize(Vec3.sub(center, eye));
    const s = Vec3.normalize(Vec3.cross(f, up));
    const u = Vec3.cross(s, f);

    return .{
        s.x,               u.x,               -f.x,             0.0,
        s.y,               u.y,               -f.y,             0.0,
        s.z,               u.z,               -f.z,             0.0,
        -Vec3.dot(s, eye), -Vec3.dot(u, eye), Vec3.dot(f, eye), 1.0,
    };
}

pub fn multiply_matrices(a: Mat4, b: Mat4) Mat4 {
    var result: Mat4 = .{0} ** 16;
    for (0..4) |row| {
        for (0..4) |col| {
            var sum: f32 = 0.0;
            for (0..4) |i| {
                sum += a[row * 4 + i] * b[i * 4 + col];
            }
            result[row * 4 + col] = sum;
        }
    }
    return result;
}

pub fn radians(_degrees: f32) f32 {
    return _degrees * (std.math.pi / 180.0);
}

fn degrees(_radians: f32) f32 {
    return _radians / (std.math.pi / 180.0);
}

pub const MadgwickFilter = struct {
    const Self = @This();
    q: Quaternion,
    initial_orientation: ?Quaternion = null,
    err: [3]f32,
    beta: f32,
    zeta: f32,
    bx: f32,
    bz: f32,
    w_bx: f32 = 0,
    w_by: f32 = 0,
    w_bz: f32 = 0,

    pub fn init() Self {
        // These gains significantly affect stability
        // const gyroMeasError = std.math.pi * (0.001 / 180.0); // increased from 5.0
        // const gyroMeasDrift = std.math.pi * (0.0001 / 180.0); // kept same

        return Self{
            .q = Quaternion.identity(),
            .beta = 0.0000020, // rad/s
            .zeta = 0.0000001,
            // .beta = std.math.sqrt(3.0 / 4.0) * gyroMeasError,
            // .zeta = std.math.sqrt(3.0 / 4.0) * gyroMeasDrift,
            .err = [_]f32{ 0.0, 0.0, 0.0 },
            .bx = 1.0,
            .bz = 0.0,
        };
    }

    pub fn update(
        self: *Self,
        gyro: Vec3,
        accel: Vec3,
        mag: Vec3,
        delta_t: f32,
    ) void {
        // Pre-compute quantities used multiple times
        var q_conj = self.q.conjugate();

        var q1 = self.q.w;
        var q2 = self.q.x;
        var q3 = self.q.y;
        var q4 = self.q.z;

        var q1q2 = q1 * q2;
        var q1q3 = q1 * q3;
        var q1q4 = q1 * q4;
        var q2q2 = q2 * q2;
        var q2q3 = q2 * q3;
        var q2q4 = q2 * q4;
        var q3q3 = q3 * q3;
        var q3q4 = q3 * q4;
        var q4q4 = q4 * q4;

        const a_norm = accel.normalize();

        const m_norm = mag.normalize();

        // Gradient decent algorithm corrective step
        const F = [6]f32{
            2.0 * (q2q4 - q1q3) - a_norm.x,
            2.0 * (q1q2 + q3q4) - a_norm.y,
            2.0 * (0.5 - q2q2 - q3q3) - a_norm.z,
            2.0 * self.bx * (0.5 - q3q3 - q4q4) + 2.0 * self.bz * (q2q4 - q1q3) - m_norm.x,
            2.0 * self.bx * (q2q3 - q1q4) + 2.0 * self.bz * (q1q2 + q3q4) - m_norm.y,
            2.0 * self.bx * (q1q3 + q2q4) + 2.0 * self.bz * (0.5 - q2q2 - q3q3) - m_norm.z,
        };

        const J_t = [6][4]f32{
            [4]f32{ -2.0 * q3, 2.0 * q4, -2.0 * q1, 2.0 * q2 },
            [4]f32{ 2.0 * q2, 2.0 * q1, 2.0 * q4, 2.0 * q3 },
            [4]f32{ 0.0, -4.0 * q2, -4.0 * q3, 0.0 },
            [4]f32{ -2.0 * self.bz * q3, 2.0 * self.bz * q4, -4.0 * self.bx * q3 - 2.0 * self.bz * q1, -4.0 * self.bx * q4 + 2.0 * self.bz * q2 },
            [4]f32{ -2.0 * self.bx * q4 + 2.0 * self.bz * q2, 2.0 * self.bx * q3 + 2.0 * self.bz * q1, 2.0 * self.bx * q2 + 2.0 * self.bz * q4, -2.0 * self.bx * q1 + 2.0 * self.bz * q3 },
            [4]f32{ 2.0 * self.bx * q3, 2.0 * self.bx * q4 - 4.0 * self.bz * q2, 2.0 * self.bx * q1 - 4.0 * self.bz * q3, 2.0 * self.bx * q2 },
        };

        var step = [4]f32{ 0, 0, 0, 0 };

        // Compute gradient (matrix multiplication)
        for (0..4) |j| {
            for (0..6) |i| {
                step[j] += J_t[i][j] * F[i];
            }
        }

        // Normalize step magnitude
        var step_vector = Vec3{
            .x = step[1],
            .y = step[2],
            .z = step[3],
        };
        step_vector = step_vector.normalize();

        var w_err = Vec3{
            .x = q_conj.x,
            .y = q_conj.y,
            .z = q_conj.z,
        };
        w_err = w_err.cross(step_vector).scale(2.0);

        // **Gyroscope Bias Correction**
        // Update gyro biases based on step and zeta
        self.w_bx += self.zeta * w_err.x * delta_t;
        self.w_by += self.zeta * w_err.y * delta_t;
        self.w_bz += self.zeta * w_err.z * delta_t;

        const gyro_corrected = Vec3{
            .x = gyro.x - self.w_bx,
            .y = gyro.y - self.w_by,
            .z = gyro.z - self.w_bz,
        };

        // Compute quaternion derivative from gyroscope data
        const q_dot_gyro = self.q.multiply(Quaternion{
            .w = 0.0,
            .x = gyro_corrected.x,
            .y = gyro_corrected.y,
            .z = gyro_corrected.z,
        }).scale(0.5);

        // Compute quaternion derivative from gradient step
        const q_dot_step = Quaternion{
            .w = 0.0, // No scalar component
            .x = -self.beta * step_vector.x, // Derived from step_vector
            .y = -self.beta * step_vector.y, // Derived from step_vector
            .z = -self.beta * step_vector.z, // Derived from step_vector
        };

        // Combine derivatives
        const q_dot = Quaternion{
            .w = q_dot_gyro.w + q_dot_step.w,
            .x = q_dot_gyro.x + q_dot_step.x,
            .y = q_dot_gyro.y + q_dot_step.y,
            .z = q_dot_gyro.z + q_dot_step.z,
        };

        // Integrate to yield new quaternion
        self.q.w += q_dot.w * delta_t;
        self.q.x += q_dot.x * delta_t;
        self.q.y += q_dot.y * delta_t;
        self.q.z += q_dot.z * delta_t;

        self.q = self.q.normalize();

        q1 = self.q.w;
        q2 = self.q.x;
        q3 = self.q.y;
        q4 = self.q.z;

        // Update reference direction of flux
        // Reference direction of Earth's magnetic field
        q1q2 = q1 * q2;
        q1q3 = q1 * q3;
        q1q4 = q1 * q4;
        q2q2 = q2 * q2;
        q2q3 = q2 * q3;
        q2q4 = q2 * q4;
        q3q3 = q3 * q3;
        q3q4 = q3 * q4;
        q4q4 = q4 * q4;

        const mag_q = Quaternion{
            .w = 0.0,
            .x = m_norm.x,
            .y = m_norm.y,
            .z = m_norm.z,
        };

        q_conj = self.q.conjugate();

        const h = self.q.multiply(mag_q).normalize().multiply(q_conj).normalize();

        const bx = std.math.sqrt(h.x * h.x + h.y * h.y);
        const bz = h.z;

        self.bx = bx;
        self.bz = bz;
    }
};

pub fn updateModelMatrix(
    accel: Vec3,
    gyro: Vec3,
    mag: Vec3,
    declination: f32,
    delta_time: f32,
    sensor_state: *SensorState,
) Quaternion {
    const accel_gl = Vec3{
        .x = accel.x, // Right
        .y = accel.y, // Up
        .z = accel.z, // Back
    };

    const gyro_gl = Vec3{
        .x = gyro.x,
        .y = gyro.y,
        .z = gyro.z,
    };

    const mag_gl = Vec3{
        .x = mag.y, // Right
        .y = mag.x, // Up
        .z = -mag.z, // Back
    };

    if (sensor_state.filter == null) {
        sensor_state.filter = MadgwickFilter.init();
    }

    _ = declination;

    sensor_state.filter.?.update(
        gyro_gl,
        accel_gl,
        mag_gl,
        delta_time,
    );

    var q = Quaternion{
        .w = sensor_state.filter.?.q.w,
        .x = sensor_state.filter.?.q.x,
        .y = sensor_state.filter.?.q.y,
        .z = sensor_state.filter.?.q.z,
    };

    if (sensor_state.filter.?.initial_orientation == null) {
        sensor_state.filter.?.initial_orientation = sensor_state.filter.?.q;
    }

    const initial_conj = sensor_state.filter.?.initial_orientation.?.conjugate();

    q = initial_conj.multiply(q);
    q = Quaternion{
        .w = q.w,
        .x = -q.x,
        .y = -q.z,
        .z = q.y,
    };

    const rotation = q.toMatrix();
    const accel_corrected = Vec3{
        .x = accel_gl.x - (rotation[1] * 1),
        .y = accel_gl.z - (rotation[5] * 1),
        .z = accel_gl.y - (rotation[9] * 1),
    };

    sensor_state.velocity = accel_corrected.scale(delta_time).add(sensor_state.velocity);
    sensor_state.position = sensor_state.velocity.scale(delta_time).add(sensor_state.position);

    std.debug.print("Position: {d:.3} =>  Velocity: {d:.3}\n", .{
        [_]f32{
            sensor_state.position.x,
            @max(sensor_state.position.y, 0.0),
            sensor_state.position.z,
        },
        [_]f32{
            sensor_state.velocity.x,
            sensor_state.velocity.y,
            sensor_state.velocity.z,
        },
    });

    // std.debug.print("Q: {d:.5}\n", .{
    //     [_]f32{
    //         q.w,
    //         q.x,
    //         q.y,
    //         q.z,
    //     },
    // });

    return q;
}
