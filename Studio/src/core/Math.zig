const std = @import("std");
const math = std.math;
const Sensors = @import("Sensors.zig");

pub fn radians(_degrees: f32) f32 {
    return _degrees * (std.math.pi / 180.0);
}

pub fn degrees(_radians: f32) f32 {
    return _radians / (std.math.pi / 180.0);
}

pub fn clamp(x: f32, low: f32, high: f32) f32 {
    return if (x < low) low else if (x > high) high else x;
}

pub const Vec3 = struct {
    const Self = @This();

    data: @Vector(3, f32),

    pub inline fn x(self: Self) f32 {
        return self.data[0];
    }

    pub inline fn y(self: Self) f32 {
        return self.data[1];
    }

    pub inline fn z(self: Self) f32 {
        return self.data[2];
    }

    pub inline fn set_x(self: *Self, value: f32) void {
        self.data[0] = value;
    }

    pub inline fn set_y(self: *Self, value: f32) void {
        self.data[1] = value;
    }

    pub inline fn set_z(self: *Self, value: f32) void {
        self.data[2] = value;
    }

    pub fn init(_x: f32, _y: f32, _z: f32) Self {
        return .{ .data = .{ _x, _y, _z } };
    }

    pub fn zero() Self {
        return .{ .data = .{ 0, 0, 0 } };
    }

    pub fn add(a: Self, b: Self) Self {
        const result: @Vector(3, f32) = a.data + b.data;
        return .{ .data = result };
    }

    pub fn add_inplace(self: *Self, other: Self) void {
        const result = self.data + other.data;
        self.data = result;
    }

    pub fn sub(a: Self, b: Self) Self {
        const result = a.data - b.data;

        return .{ .data = result };
    }

    pub fn sub_inplace(self: *Self, other: Self) void {
        const result = self.data - other.data;
        self.data = result;
    }

    pub fn multiply(a: Self, b: Self) Self {
        const result = a.data * b.data;
        return .{ .data = result };
    }

    pub fn scale(self: Self, scalar: f32) Self {
        const s: @Vector(3, f32) = @splat(scalar);
        const result = self.data * s;

        return .{ .data = result };
    }

    pub fn scale_inplace(self: *Self, scalar: f32) void {
        const s: @Vector(3, f32) = @splat(scalar);
        const result = self.data * s;
        self.data = result;
    }

    pub fn dot(a: Self, b: Self) f32 {
        const product = a.data * b.data;
        return @reduce(.Add, product);
    }

    pub fn cross(a: Self, b: Self) Self {
        return .{
            .data = .{
                a.y() * b.z() - a.z() * b.y(),
                a.z() * b.x() - a.x() * b.z(),
                a.x() * b.y() - a.y() * b.x(),
            },
        };
    }

    pub fn length(self: Self) f32 {
        return @sqrt(self.dot(self));
    }

    pub fn lengthSquared(self: Self) f32 {
        return self.dot(self);
    }

    //TODO: Return error if check is false
    pub fn normalize(self: Self) Self {
        const len = self.length();

        if (math.approxEqAbs(f32, len, 0.0, 1e-6)) {
            std.debug.print("Vec3 with 0 Length Detected", .{});
            return self;
        }

        return self.scale(1.0 / len);
    }

    //TODO: Return error if check is false
    pub fn normalize_inplace(self: *Self) void {
        const len = self.length();

        if (math.approxEqAbs(f32, len, 0.0, 1e-6)) {
            std.debug.print("Vec3 with 0 Length Detected", .{});
            return;
        }

        self.scale_inplace(1.0 / len);
    }

    pub fn lerp(a: Self, b: Self, t: f32) Self {
        const vt: @Vector(3, f32) = @splat(t);

        // a + t * (b - a)
        const diff = b.data - a.data;
        const scaled = diff * vt;
        const result = a.data + scaled;

        return .{ .data = result };
    }

    /// Create vector from angles
    pub fn from_angles(yaw_deg: f32, pitch_deg: f32) Self {
        const yaw = radians(yaw_deg);
        const pitch = radians(pitch_deg);

        const front = Self.init(@cos(yaw) * @cos(pitch), @sin(pitch), @sin(yaw) * @cos(pitch));
        return front.normalize();
    }

    /// Check if two vectors are approximately equal
    pub fn approx_eq(a: Self, b: Self, tolerance: f32) bool {
        const diff = @abs(a.data - b.data);
        const tol: @Vector(3, f32) = @splat(tolerance);

        // Check if all components are within tolerance
        const within_tolerance: @Vector(3, bool) = diff <= tol;
        return @reduce(.And, within_tolerance);
    }

    pub fn rotate_by_quaternion(v: Self, q: Quaternion) Self {
        const v_quat = Quaternion.init(v.x(), v.y(), v.z(), 0.0);

        // q * v * q^-1
        const result = q.multiply(v_quat).multiply(q.conjugate());

        return Vec3.init(result.x(), result.y(), result.z());
    }

    /// Convert to string representation (for debugging)
    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("Vec3({d:.6}, {d:.6}, {d:.6})", .{ self.x(), self.y(), self.z() });
    }
};

pub fn Matrix(comptime N: usize) type {
    return struct {
        data: [N * N]f32,

        const Self = @This();

        pub fn identity() Self {
            var result = Self{ .data = undefined };
            @memset(&result.data, 0);

            var i: usize = 0;
            while (i < N) : (i += 1) {
                result.data[i * N + i] = 1.0;
            }
            return result;
        }

        pub fn zero() Self {
            var result = Self{ .data = undefined };
            @memset(&result.data, 0);
            return result;
        }

        pub inline fn at(self: Self, i: usize, j: usize) f32 {
            return self.data[i * N + j];
        }

        pub inline fn set(self: *Self, i: usize, j: usize, value: f32) void {
            self.data[i * N + j] = value;
        }

        pub fn add(self: Self, other: Self) Self {
            var result = Self{ .data = undefined };

            for (0..N * N) |i| {
                result.data[i] = self.data[i] + other.data[i];
            }

            return result;
        }

        pub fn subtract(self: Self, other: Self) Self {
            var result = Self{ .data = undefined };

            for (0..N * N) |i| {
                result.data[i] = self.data[i] - other.data[i];
            }

            return result;
        }

        pub fn scale(self: Self, scalar: f32) Self {
            var result = Self{ .data = undefined };

            for (0..N * N) |i| {
                result.data[i] = self.data[i] * scalar;
            }

            return result;
        }

        pub fn multiply(self: Self, other: Self) Self {
            var result = Self.zero();

            for (0..N) |i| {
                for (0..N) |j| {
                    var sum: f32 = 0.0;
                    for (0..N) |k| {
                        sum += self.data[i * N + k] * other.data[k * N + j];
                    }
                    result.data[i * N + j] = sum;
                }
            }

            return result;
        }

        pub fn transpose(self: Self) Self {
            var result = Self{ .data = undefined };

            for (0..N) |i| {
                for (0..N) |j| {
                    result.data[j * N + i] = self.data[i * N + j];
                }
            }

            return result;
        }

        pub fn to_array(self: Self) [N * N]f32 {
            return self.data;
        }

        pub fn from_array(arr: [N * N]f32) Self {
            return Self{ .data = arr };
        }
    };
}

pub const Mat2 = struct {
    base: Matrix(2),

    const Self = @This();

    /// Create from generic Matrix(2)
    pub fn from_matrix(m: [2 * 2]f32) Self {
        return Self{ .base = Matrix(2).from_array(m) };
    }

    /// Create identity matrix
    pub fn identity() Self {
        return Self{ .base = Matrix(2).identity() };
    }

    /// Create zero matrix
    pub fn zero() Self {
        return Self{ .base = Matrix(2).zero() };
    }

    /// Create rotation matrix
    pub fn rotation(angle_deg: f32) Self {
        const angle_rad = angle_deg * (math.pi / 180.0);
        const c = @cos(angle_rad);
        const s = @sin(angle_rad);

        var result = Self{ .base = undefined };
        result.base.data[0] = c;
        result.base.data[1] = -s;
        result.base.data[2] = s;
        result.base.data[3] = c;

        return result;
    }

    /// Create scaling matrix
    pub fn scaling(x: f32, y: f32) Self {
        var result = Self{ .base = Matrix(2).identity() };
        result.base.data[0] = x;
        result.base.data[3] = y;

        return result;
    }

    /// Pass-through methods to base matrix
    pub fn add(self: Self, other: Self) Self {
        return Self{ .base = self.base.add(other.base) };
    }

    pub fn subtract(self: Self, other: Self) Self {
        return Self{ .base = self.base.subtract(other.base) };
    }

    pub fn multiply(self: Self, other: Self) Self {
        return Self{ .base = self.base.multiply(other.base) };
    }

    pub fn scale(self: Self, scalar: f32) Self {
        return Self{ .base = self.base.scale(scalar) };
    }

    pub fn transpose(self: Self) Self {
        return Self{ .base = self.base.transpose() };
    }
};

/// Specialized 3x3 matrix type with additional methods
pub const Mat3 = struct {
    base: Matrix(3),

    const Self = @This();

    pub fn to_array(self: Self) [3 * 3]f32 {
        return self.base.to_array();
    }

    /// Create from generic Matrix(3)
    pub fn from_array(m: [3 * 3]f32) Self {
        return Self{ .base = Matrix(3).from_array(m) };
    }

    /// Create identity matrix
    pub fn identity() Self {
        return Self{ .base = Matrix(3).identity() };
    }

    /// Create zero matrix
    pub fn zero() Self {
        return Self{ .base = Matrix(3).zero() };
    }

    /// Create rotation matrix around X axis
    pub fn rotation_x(angle_deg: f32) Self {
        const angle_rad = angle_deg * (math.pi / 180.0);
        const c = @cos(angle_rad);
        const s = @sin(angle_rad);

        var result = Self{ .base = Matrix(3).identity() };
        result.base.data[4] = c; // [1,1]
        result.base.data[5] = -s; // [1,2]
        result.base.data[7] = s; // [2,1]
        result.base.data[8] = c; // [2,2]

        return result;
    }

    /// Create rotation matrix around Y axis
    pub fn rotation_y(angle_deg: f32) Self {
        const angle_rad = angle_deg * (math.pi / 180.0);
        const c = @cos(angle_rad);
        const s = @sin(angle_rad);

        var result = Self{ .base = Matrix(3).identity() };
        result.base.data[0] = c; // [0,0]
        result.base.data[2] = s; // [0,2]
        result.base.data[6] = -s; // [2,0]
        result.base.data[8] = c; // [2,2]

        return result;
    }

    /// Create rotation matrix around Z axis
    pub fn rotation_z(angle_deg: f32) Self {
        const angle_rad = angle_deg * (math.pi / 180.0);
        const c = @cos(angle_rad);
        const s = @sin(angle_rad);

        var result = Self{ .base = Matrix(3).identity() };
        result.base.data[0] = c; // [0,0]
        result.base.data[1] = -s; // [0,1]
        result.base.data[3] = s; // [1,0]
        result.base.data[4] = c; // [1,1]

        return result;
    }

    /// Create scaling matrix
    pub fn scaling(x: f32, y: f32, z: f32) Self {
        var result = Self{ .base = Matrix(3).identity() };
        result.base.data[0] = x;
        result.base.data[4] = y;
        result.base.data[8] = z;

        return result;
    }

    /// Create 3x3 rotation matrix from Quaternion
    pub fn from_quaternion(q: Quaternion) Self {
        const qx = q.x();
        const qy = q.y();
        const qz = q.z();
        const qw = q.w();

        const xx = qx * qx;
        const yy = qy * qy;
        const zz = qz * qz;
        const xy = qx * qy;
        const xz = qx * qz;
        const yz = qy * qz;
        const wx = qw * qx;
        const wy = qw * qy;
        const wz = qw * qz;

        var result = Self{ .base = undefined };

        // Row 0
        result.base.data[0] = 1 - 2 * (yy + zz);
        result.base.data[1] = 2 * (xy - wz);
        result.base.data[2] = 2 * (xz + wy);

        // Row 1
        result.base.data[3] = 2 * (xy + wz);
        result.base.data[4] = 1 - 2 * (xx + zz);
        result.base.data[5] = 2 * (yz - wx);

        // Row 2
        result.base.data[6] = 2 * (xz - wy);
        result.base.data[7] = 2 * (yz + wx);
        result.base.data[8] = 1 - 2 * (xx + yy);

        return result;
    }

    // TODO: Add inplace operations
    pub fn add(self: Self, other: Self) Self {
        return Self{ .base = self.base.add(other.base) };
    }

    pub fn subtract(self: Self, other: Self) Self {
        return Self{ .base = self.base.subtract(other.base) };
    }

    pub fn multiply(self: Self, other: Self) Self {
        return Self{ .base = self.base.multiply(other.base) };
    }

    pub fn scale(self: Self, scalar: f32) Self {
        return Self{ .base = self.base.scale(scalar) };
    }
};

// TODO: Add inplace operations
pub const Mat4 = struct {
    base: Matrix(4),

    const Self = @This();

    pub fn to_array(self: Self) [4 * 4]f32 {
        return self.base.to_array();
    }

    pub fn from_array(m: [4 * 4]f32) Self {
        return Self{ .base = Matrix(4).from_array(m) };
    }

    pub fn identity() Self {
        return Self{ .base = Matrix(4).identity() };
    }

    pub fn zero() Self {
        return Self{ .base = Matrix(4).zero() };
    }

    /// Create translation matrix
    pub fn translation(x: f32, y: f32, z: f32) Self {
        var result = Self{ .base = Matrix(4).identity() };
        result.base.data[12] = x; // [0,3]
        result.base.data[13] = y; // [1,3]
        result.base.data[14] = z; // [2,3]

        return result;
    }

    /// Apply translation to existing matrix
    pub fn translate(self: Self, x: f32, y: f32, z: f32) Self {
        var result = self;
        const m = self.base.data;

        // Translation components in last column
        result.base.data[12] = m[0] * x + m[4] * y + m[8] * z + m[12];
        result.base.data[13] = m[1] * x + m[5] * y + m[9] * z + m[13];
        result.base.data[14] = m[2] * x + m[6] * y + m[10] * z + m[14];
        result.base.data[15] = m[3] * x + m[7] * y + m[11] * z + m[15];

        return result;
    }

    /// Create scaling matrix
    pub fn scaling(x: f32, y: f32, z: f32) Self {
        var result = Self{ .base = Matrix(4).identity() };
        result.base.data[0] = x;
        result.base.data[5] = y;
        result.base.data[10] = z;

        return result;
    }

    /// Apply scaling to existing matrix
    pub fn scale(self: Self, x: f32, y: f32, z: f32) Self {
        var result = self;

        // Scale the basis vectors
        for (0..4) |i| {
            result.base.data[i] *= x; // First row
            result.base.data[4 + i] *= y; // Second row
            result.base.data[8 + i] *= z; // Third row
        }

        return result;
    }

    /// Create rotation matrix around X axis
    pub fn rotation_x(angle_deg: f32) Self {
        const angle_rad = angle_deg * (math.pi / 180.0);
        const c = @cos(angle_rad);
        const s = @sin(angle_rad);

        var result = Self{ .base = Matrix(4).identity() };
        result.base.data[5] = c; // [1,1]
        result.base.data[6] = -s; // [1,2]
        result.base.data[9] = s; // [2,1]
        result.base.data[10] = c; // [2,2]

        return result;
    }

    /// Create rotation matrix around Y axis
    pub fn rotation_y(angle_deg: f32) Self {
        const angle_rad = radians(angle_deg);
        const c = @cos(angle_rad);
        const s = @sin(angle_rad);

        var result = Self{ .base = Matrix(4).identity() };
        result.base.data[0] = c; // [0,0]
        result.base.data[2] = s; // [0,2]
        result.base.data[8] = -s; // [2,0]
        result.base.data[10] = c; // [2,2]

        return result;
    }

    /// Create rotation matrix around Z axis
    pub fn rotation_z(angle_deg: f32) Self {
        const angle_rad = angle_deg * (math.pi / 180.0);
        const c = @cos(angle_rad);
        const s = @sin(angle_rad);

        var result = Self{ .base = Matrix(4).identity() };
        result.base.data[0] = c; // [0,0]
        result.base.data[1] = -s; // [0,1]
        result.base.data[4] = s; // [1,0]
        result.base.data[5] = c; // [1,1]

        return result;
    }

    /// Create perspective projection matrix
    pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) Self {
        const rad = radians(fov);
        const tan_half_fov = @tan(rad / 2.0);

        var result = Self{ .base = Matrix(4).zero() };
        result.base.data[0] = 1.0 / (aspect * tan_half_fov);
        result.base.data[5] = 1.0 / tan_half_fov;
        result.base.data[10] = -(far + near) / (far - near);
        result.base.data[11] = -1.0;
        result.base.data[14] = -(2.0 * far * near) / (far - near);

        return result;
    }

    /// Create orthographic projection matrix
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) Self {
        var result = Self{ .base = Matrix(4).identity() };

        result.base.data[0] = 2.0 / (right - left);
        result.base.data[5] = 2.0 / (top - bottom);
        result.base.data[10] = -2.0 / (far - near);

        result.base.data[3] = -(right + left) / (right - left);
        result.base.data[7] = -(top + bottom) / (top - bottom);
        result.base.data[11] = -(far + near) / (far - near);

        return result;
    }

    /// Create look-at view matrix
    pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) Self {
        // Calculate forward vector (normalized eye to center)
        var f: Vec3 = center.sub(eye).normalize();

        // Calculate right vector (normalized cross product of forward and up)
        var s = Vec3.cross(f, up).normalize();

        // Calculate camera up vector (cross product of right and forward)
        const u = Vec3.cross(s, f);

        var result = Self{ .base = Matrix(4).identity() };

        // Row 0
        result.base.data[0] = s.x();
        result.base.data[1] = u.x();
        result.base.data[2] = -f.x();

        result.base.data[4] = s.y();
        result.base.data[5] = u.y();
        result.base.data[6] = -f.y();

        result.base.data[8] = s.z();
        result.base.data[9] = u.z();
        result.base.data[10] = -f.z();

        result.base.data[12] = -Vec3.dot(s, eye);
        result.base.data[13] = -Vec3.dot(u, eye);
        result.base.data[14] = Vec3.dot(f, eye);

        return result;
    }

    /// Create 4x4 matrix from Quaternion
    pub fn from_quaternion(q: Quaternion) Self {
        const qx = q.x();
        const qy = q.y();
        const qz = q.z();
        const qw = q.w();

        const xx = qx * qx;
        const yy = qy * qy;
        const zz = qz * qz;
        const xy = qx * qy;
        const xz = qx * qz;
        const yz = qy * qz;
        const wx = qw * qx;
        const wy = qw * qy;
        const wz = qw * qz;

        var result = Self{ .base = undefined };

        // Row 0
        result.base.data[0] = 1 - 2 * (yy + zz);
        result.base.data[1] = 2 * (xy - wz);
        result.base.data[2] = 2 * (xz + wy);
        result.base.data[3] = 0;

        // Row 1
        result.base.data[4] = 2 * (xy + wz);
        result.base.data[5] = 1 - 2 * (xx + zz);
        result.base.data[6] = 2 * (yz - wx);
        result.base.data[7] = 0;

        // Row 2
        result.base.data[8] = 2 * (xz - wy);
        result.base.data[9] = 2 * (yz + wx);
        result.base.data[10] = 1 - 2 * (xx + yy);
        result.base.data[11] = 0;

        // Row 3
        result.base.data[12] = 0;
        result.base.data[13] = 0;
        result.base.data[14] = 0;
        result.base.data[15] = 1;

        return result;
    }

    /// Create a Mat4 from Mat3 with 0 padding
    pub fn from_mat3(m: Mat3) Mat4 {
        const data: [4 * 4]f32 = undefined;

        data[0] = m.data[0];
        data[1] = m.data[1];
        data[2] = m.data[2];
        data[4] = m.data[3];
        data[5] = m.data[4];
        data[6] = m.data[5];
        data[8] = m.data[6];
        data[9] = m.data[7];
        data[10] = m.data[8];

        return Self.from_array(data);
    }

    /// Extract the upper 3x3 portion of the matrix
    pub fn to_mat3(self: Self) Mat3 {
        var result = Matrix(3){ .data = undefined };

        // Copy upper-left 3x3 submatrix
        result.data[0] = self.base.data[0];
        result.data[1] = self.base.data[1];
        result.data[2] = self.base.data[2];
        result.data[3] = self.base.data[4];
        result.data[4] = self.base.data[5];
        result.data[5] = self.base.data[6];
        result.data[6] = self.base.data[8];
        result.data[7] = self.base.data[9];
        result.data[8] = self.base.data[10];

        return Mat3{ .base = result };
    }

    /// Gets the right basis vectors in row-major order
    pub inline fn get_right(self: Self) Vec3 {
        const right_x = self.base.data[0];
        const right_y = self.base.data[1];
        const right_z = self.base.data[2];

        return Vec3.init(right_x, right_y, right_z).normalize();
    }

    /// Gets the up basis vectors in row-major order
    pub inline fn get_up(self: Self) Vec3 {
        const up_x = self.base.data[4];
        const up_y = self.base.data[5];
        const up_z = self.base.data[6];

        return Vec3.init(up_x, up_y, up_z).normalize();
    }

    /// Gets the forward basis vectors in row-major order
    pub inline fn get_forward(self: Self) Vec3 {
        const forward_x = self.base.data[8];
        const forward_y = self.base.data[9];
        const forward_z = self.base.data[10];

        return Vec3.init(forward_x, forward_y, forward_z).normalize();
    }

    /// Pass-through methods to base matrix
    pub fn add(self: Self, other: Self) Self {
        return Self{ .base = self.base.add(other.base) };
    }

    pub fn subtract(self: Self, other: Self) Self {
        return Self{ .base = self.base.subtract(other.base) };
    }

    pub fn multiply(self: Self, other: Self) Self {
        return Self{ .base = self.base.multiply(other.base) };
    }

    pub fn scale_uniform(self: Self, scalar: f32) Self {
        return Self{ .base = self.base.scale(scalar) };
    }

    pub fn transpose(self: Self) Self {
        return Self{ .base = self.base.transpose() };
    }

    pub fn determinant(self: Self) f32 {
        return self.base.determinant();
    }

    pub fn inverse(self: Self) ?Self {
        if (self.base.inverse()) |inv| {
            return Self{ .base = inv };
        }
        return null;
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        const m = self.base.data;
        try writer.print("Mat4(\n{d:.6}\n{d:.6}\n{d:.6}\n{d:.6}\n)\n", .{
            m[0..4],
            m[4..8],
            m[8..12],
            m[12..16],
        });
    }
};

// Todo: Use @Vector instead for SIMD
pub const Quaternion = struct {
    const Self = @This();

    data: @Vector(4, f32), // [x, y, z, w]

    pub fn init(_x: f32, _y: f32, _z: f32, _w: f32) Self {
        return Self{ .data = .{ _x, _y, _z, _w } };
    }

    pub inline fn x(self: Quaternion) f32 {
        return self.data[0];
    }
    pub inline fn y(self: Quaternion) f32 {
        return self.data[1];
    }
    pub inline fn z(self: Quaternion) f32 {
        return self.data[2];
    }
    pub inline fn w(self: Quaternion) f32 {
        return self.data[3];
    }

    pub fn identity() Quaternion {
        return Quaternion.init(0, 0, 0, 1);
    }

    pub fn add(a: Self, b: Self) Self {
        const result = a.data + b.data;
        return .{ .data = result };
    }

    pub fn add_inplace(a: Self, b: Self) Self {
        _ = a;
        _ = b;
        @panic("Not Implemented!");
    }

    pub fn sub(a: Self, b: Self) Self {
        const result = a.data - b.data;
        return .{ .data = result };
    }

    pub fn sub_inplace(a: Self, b: Self) Self {
        _ = a;
        _ = b;
        @panic("Not Implemented!");
    }

    pub fn multiply(a: Self, b: Self) Self {
        return Self.init(
            a.w() * b.x() + a.x() * b.w() + a.y() * b.z() - a.z() * b.y(),
            a.w() * b.y() - a.x() * b.z() + a.y() * b.w() + a.z() * b.x(),
            a.w() * b.z() + a.x() * b.y() - a.y() * b.x() + a.z() * b.w(),
            a.w() * b.w() - a.x() * b.x() - a.y() * b.y() - a.z() * b.z(),
        );
    }

    pub fn multiply_inplace(a: Self, b: Self) Self {
        _ = a;
        _ = b;
        @panic("Not Implemented!");
    }

    pub fn dot(a: Self, b: Self) f32 {
        const product = a.data * b.data;
        const result = @reduce(.Add, product);
        return result;
    }

    pub fn dot_inplace(a: Self, b: Self) f32 {
        _ = a;
        _ = b;
        @panic("Not Implemented!");
    }

    pub fn scale(self: Self, scalar: f32) Self {
        const s: @Vector(4, f32) = @splat(scalar);
        const result = self.data * s;

        return .{ .data = result };
    }

    pub fn scale_inplace(self: Self, scalar: f32) Self {
        _ = self;
        _ = scalar;
        @panic("Not Implemented!");
    }

    pub fn length(self: Self) f32 {
        const _dot = self.dot(self);
        return @sqrt(_dot);
    }

    //TODO: Return error if check is false
    pub fn normalize(q: Self) Self {
        const len = q.length();

        if (math.approxEqAbs(f32, len, 0.0, 1e-6)) {
            std.debug.print("Quaternion with 0 Length Detected", .{});
            return q;
        }

        return q.scale(1.0 / len);
    }

    //TODO: Return error if check is false
    pub fn normalize_inplace(self: *Self) void {
        _ = self;
        @panic("Not Implemented!");
    }

    pub fn conjugate(self: Self) Self {
        // Conjugate: keep w, negate x, y, z
        const mask: @Vector(4, f32) = .{ -1, -1, -1, 1 };
        const result = self.data * mask;

        return .{ .data = result };
    }

    pub fn from_axis_angle(axis: Vec3, angle: f32) Self {
        const normalized_axis = axis.normalize();

        // Calculate the sine and cosine of half the angle
        const half_angle = radians(angle * 0.5);
        const sin_half = @sin(half_angle);
        const cos_half = @cos(half_angle);

        return Self.init(
            normalized_axis.x() * sin_half,
            normalized_axis.y() * sin_half,
            normalized_axis.z() * sin_half,
            cos_half,
        );
    }

    pub fn to_euler(self: Self) [3]f32 {
        const qx = self.x();
        const qy = self.y();
        const qz = self.z();
        const qw = self.w();

        // Pitch (rotation around X-axis)
        const sin_pitch = 2.0 * (qw * qx + qy * qz);
        const cos_pitch = 1.0 - 2.0 * (qx * qx + qy * qy);
        const pitch = std.math.atan2(sin_pitch, cos_pitch);

        // Yaw (rotation around Y-axis)
        const sin_yaw = 2.0 * (qw * qy - qz * qx);
        // Clamp siny to ensure the value is in the valid range for asin
        const yaw = std.math.asin(clamp(sin_yaw, -1.0, 1.0));

        // Roll (rotation around Z-axis)
        const sin_roll = 2.0 * (qw * qz + qx * qy);
        const cos_roll = 1.0 - 2.0 * (qy * qy + qz * qz);
        const roll = std.math.atan2(sin_roll, cos_roll);

        return [3]f32{ pitch, yaw, roll };
    }

    pub fn from_euler(pitch: f32, yaw: f32, roll: f32) Self {
        const qx = from_axis_angle(Vec3.init(1, 0, 0), pitch);
        const qy = from_axis_angle(Vec3.init(0, 1, 0), yaw);
        const qz = from_axis_angle(Vec3.init(0, 0, 1), roll);

        // roll → pitch → yaw (Extrinsic)
        return qz.multiply(qy).multiply(qx).normalize();
    }

    pub fn from_mat3(_mat: Mat3) Self {
        const mat = _mat.base.data;
        const trace = mat[0] + mat[4] + mat[8];
        var result = Self{
            .w = 0,
            .x = 0,
            .y = 0,
            .z = 0,
        };

        if (trace > 0) {
            const S = @sqrt(trace + 1.0) * 2;
            result.w = 0.25 * S;
            result.x = (mat[7] - mat[5]) / S;
            result.y = (mat[2] - mat[6]) / S;
            result.z = (mat[3] - mat[1]) / S;
        } else if ((mat[0] > mat[4]) and (mat[0] > mat[8])) {
            const S = @sqrt(1.0 + mat[0] - mat[4] - mat[8]) * 2;
            result.w = (mat[7] - mat[5]) / S;
            result.x = 0.25 * S;
            result.y = (mat[1] + mat[3]) / S;
            result.z = (mat[2] + mat[6]) / S;
        } else if (mat[4] > mat[8]) {
            const S = @sqrt(1.0 + mat[4] - mat[0] - mat[8]) * 2;
            result.w = (mat[2] - mat[6]) / S;
            result.x = (mat[1] + mat[3]) / S;
            result.y = 0.25 * S;
            result.z = (mat[5] + mat[7]) / S;
        } else {
            const S = @sqrt(1.0 + mat[8] - mat[0] - mat[4]) * 2;
            result.w = (mat[3] - mat[1]) / S;
            result.x = (mat[2] + mat[6]) / S;
            result.y = (mat[5] + mat[7]) / S;
            result.z = 0.25 * S;
        }

        // Normalize the quaternion
        return result.normalize();
    }

    pub fn to_mat4(_q: Self) Mat4 {
        const q = _q.normalize();

        const qx = q.x();
        const qy = q.y();
        const qz = q.z();
        const qw = q.w();

        const xx = qx * qx;
        const yy = qy * qy;
        const zz = qz * qz;
        const xy = qx * qy;
        const xz = qx * qz;
        const yz = qy * qz;
        const wx = qw * qx;
        const wy = qw * qy;
        const wz = qw * qz;

        const data = [4 * 4]f32{
            1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),     0,
            2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),     0,
            2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy), 0,
            0,                 0,                 0,                 1,
        };

        return Mat4.from_array(data);
    }

    pub fn slerp(a: Self, b: Self, t: f32) Self {
        const ax = a.x();
        const ay = a.y();
        const az = a.z();
        const aw = a.w();

        const bx = b.x();
        const by = b.y();
        const bz = b.z();
        const bw = b.w();

        var cos_half_theta = aw * bw + ax * bx + ay * by + az * bz;

        var b_copy: Self = undefined;
        if (cos_half_theta < 0.0) {
            b_copy = Self.init(-bx, -by, -bz, -bw);
            cos_half_theta = -cos_half_theta;
        }

        const bcx = b_copy.x();
        const bcy = b_copy.y();
        const bcz = b_copy.z();
        const bcw = b_copy.w();

        if (cos_half_theta > 0.9995) {
            return Self.normalize(Self.init(
                ax + t * (bcx - ax),
                ay + t * (bcy - ay),
                az + t * (bcz - az),
                aw + t * (bcw - aw),
            ));
        } else {
            const half_theta = std.math.acos(cos_half_theta);
            const sin_half_theta = @sin(half_theta);
            const ratio_a = @sin((1 - t) * half_theta) / sin_half_theta;
            const ratio_b = @sin(t * half_theta) / sin_half_theta;

            return Self.init(
                ax * ratio_a + bcx * ratio_b,
                ay * ratio_a + bcy * ratio_b,
                az * ratio_a + bcz * ratio_b,
                aw * ratio_a + bcw * ratio_b,
            );
        }
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        const m = self.data;
        try writer.print("Quat({d:.6})\n", .{m});
    }
};
