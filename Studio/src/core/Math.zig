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
        _ = std.builtin;
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

    pub fn from_array(arr: [3]f32) Self {
        return Self{ .data = arr };
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

        pub fn determinant(self: Self) f32 {
            if (N == 2) {
                return self.data[0] * self.data[3] - self.data[1] * self.data[2];
            } else if (N == 3) {
                return self.data[0] * (self.data[4] * self.data[8] - self.data[5] * self.data[7]) -
                    self.data[1] * (self.data[3] * self.data[8] - self.data[5] * self.data[6]) +
                    self.data[2] * (self.data[3] * self.data[7] - self.data[4] * self.data[6]);
            } else if (N == 4) {
                // 4x4 determinant using cofactor expansion along first row
                const m = self.data;
                const a00 = m[0];
                const a01 = m[4];
                const a02 = m[8];
                const a03 = m[12];
                const a10 = m[1];
                const a11 = m[5];
                const a12 = m[9];
                const a13 = m[13];
                const a20 = m[2];
                const a21 = m[6];
                const a22 = m[10];
                const a23 = m[14];
                const a30 = m[3];
                const a31 = m[7];
                const a32 = m[11];
                const a33 = m[15];

                return a00 * (a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)) -
                    a01 * (a10 * (a22 * a33 - a23 * a32) - a12 * (a20 * a33 - a23 * a30) + a13 * (a20 * a32 - a22 * a30)) +
                    a02 * (a10 * (a21 * a33 - a23 * a31) - a11 * (a20 * a33 - a23 * a30) + a13 * (a20 * a31 - a21 * a30)) -
                    a03 * (a10 * (a21 * a32 - a22 * a31) - a11 * (a20 * a32 - a22 * a30) + a12 * (a20 * a31 - a21 * a30));
            } else {
                @compileError("Determinant not implemented for matrices larger than 4x4");
            }
        }

        pub fn inverse(self: Self) ?Self {
            if (N == 4) {
                const m = self.data;
                const a00 = m[0];
                const a01 = m[4];
                const a02 = m[8];
                const a03 = m[12];
                const a10 = m[1];
                const a11 = m[5];
                const a12 = m[9];
                const a13 = m[13];
                const a20 = m[2];
                const a21 = m[6];
                const a22 = m[10];
                const a23 = m[14];
                const a30 = m[3];
                const a31 = m[7];
                const a32 = m[11];
                const a33 = m[15];

                const det = self.determinant();
                if (@abs(det) < 1e-8) return null; // Matrix is singular

                const inv_det = 1.0 / det;

                var result = Self{ .data = undefined };

                // Calculate adjugate matrix and divide by determinant
                result.data[0] = inv_det * (a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31));
                result.data[4] = inv_det * -(a01 * (a22 * a33 - a23 * a32) - a02 * (a21 * a33 - a23 * a31) + a03 * (a21 * a32 - a22 * a31));
                result.data[8] = inv_det * (a01 * (a12 * a33 - a13 * a32) - a02 * (a11 * a33 - a13 * a31) + a03 * (a11 * a32 - a12 * a31));
                result.data[12] = inv_det * -(a01 * (a12 * a23 - a13 * a22) - a02 * (a11 * a23 - a13 * a21) + a03 * (a11 * a22 - a12 * a21));

                result.data[1] = inv_det * -(a10 * (a22 * a33 - a23 * a32) - a12 * (a20 * a33 - a23 * a30) + a13 * (a20 * a32 - a22 * a30));
                result.data[5] = inv_det * (a00 * (a22 * a33 - a23 * a32) - a02 * (a20 * a33 - a23 * a30) + a03 * (a20 * a32 - a22 * a30));
                result.data[9] = inv_det * -(a00 * (a12 * a33 - a13 * a32) - a02 * (a10 * a33 - a13 * a30) + a03 * (a10 * a32 - a12 * a30));
                result.data[13] = inv_det * (a00 * (a12 * a23 - a13 * a22) - a02 * (a10 * a23 - a13 * a20) + a03 * (a10 * a22 - a12 * a20));

                result.data[2] = inv_det * (a10 * (a21 * a33 - a23 * a31) - a11 * (a20 * a33 - a23 * a30) + a13 * (a20 * a31 - a21 * a30));
                result.data[6] = inv_det * -(a00 * (a21 * a33 - a23 * a31) - a01 * (a20 * a33 - a23 * a30) + a03 * (a20 * a31 - a21 * a30));
                result.data[10] = inv_det * (a00 * (a11 * a33 - a13 * a31) - a01 * (a10 * a33 - a13 * a30) + a03 * (a10 * a31 - a11 * a30));
                result.data[14] = inv_det * -(a00 * (a11 * a23 - a13 * a21) - a01 * (a10 * a23 - a13 * a20) + a03 * (a10 * a21 - a11 * a20));

                result.data[3] = inv_det * -(a10 * (a21 * a32 - a22 * a31) - a11 * (a20 * a32 - a22 * a30) + a12 * (a20 * a31 - a21 * a30));
                result.data[7] = inv_det * (a00 * (a21 * a32 - a22 * a31) - a01 * (a20 * a32 - a22 * a30) + a02 * (a20 * a31 - a21 * a30));
                result.data[11] = inv_det * -(a00 * (a11 * a32 - a12 * a31) - a01 * (a10 * a32 - a12 * a30) + a02 * (a10 * a31 - a11 * a30));
                result.data[15] = inv_det * (a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20));

                return result;
            } else {
                @compileError("Matrix inversion not implemented for matrices other than 4x4");
            }
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

//TODO: add a to_quaternion function
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

    pub fn transpose(self: Self) Self {
        return Self{ .base = self.base.transpose() };
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

    /// Transform a Vec3 by this matrix
    pub fn transformVec3(self: Self, v: Vec3) Vec3 {
        const m = self.base.data;
        return Vec3.init(
            m[0] * v.x() + m[1] * v.y() + m[2] * v.z(), // row 0
            m[3] * v.x() + m[4] * v.y() + m[5] * v.z(), // row 1
            m[6] * v.x() + m[7] * v.y() + m[8] * v.z(), // row 2
        );
    }

    /// Transform a 3-element array by this matrix
    pub fn transformArray(self: Self, arr: [3]f32) [3]f32 {
        const m = self.base.data;
        return [3]f32{
            m[0] * arr[0] + m[1] * arr[1] + m[2] * arr[2], // row 0
            m[3] * arr[0] + m[4] * arr[1] + m[5] * arr[2], // row 1
            m[6] * arr[0] + m[7] * arr[1] + m[8] * arr[2], // row 2
        };
    }
};

pub const TRS = struct {
    translation: [3]f32,
    rotation: Quaternion,
    scale: [3]f32,
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
        result.base.data[12] = x; // [3,0]
        result.base.data[13] = y; // [3,1]
        result.base.data[14] = z; // [3,2]

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

    pub fn decomposeTRS(self: Self) TRS {
        const mat = self.base.data;
        //  Extract translation.
        // In a column-major matrix, the translation is in mat[12], mat[13], mat[14].
        // mat[15] is typically 1.0 in an affine transform.
        const _translation = [3]f32{ mat[12], mat[13], mat[14] };

        // Extract scale by taking the length of each of the first 3 columns.
        // Matrices are stored row-major, so column 0 is at [0,4,8], column 1 at [1,5,9], column 2 at [2,6,10]
        const sx = math.sqrt(mat[0] * mat[0] + mat[4] * mat[4] + mat[8] * mat[8]);
        const sy = math.sqrt(mat[1] * mat[1] + mat[5] * mat[5] + mat[9] * mat[9]);
        const sz = math.sqrt(mat[2] * mat[2] + mat[6] * mat[6] + mat[10] * mat[10]);

        const _scale = [3]f32{ sx, sy, sz };

        // Build a 3×3 rotation matrix by dividing out the scale from each column.
        const r00 = mat[0] / sx;
        const r01 = mat[1] / sy;
        const r02 = mat[2] / sz;
        const r10 = mat[4] / sx;
        const r11 = mat[5] / sy;
        const r12 = mat[6] / sz;
        const r20 = mat[8] / sx;
        const r21 = mat[9] / sy;
        const r22 = mat[10] / sz;

        // Convert this 3×3 rotation to a quaternion.
        const trace = r00 + r11 + r22;

        var qx: f32 = 0;
        var qy: f32 = 0;
        var qz: f32 = 0;
        var qw: f32 = 0;

        if (trace > 0) {
            const s = 0.5 / math.sqrt(trace + 1.0);
            qw = 0.25 / s;
            qx = (r21 - r12) * s;
            qy = (r02 - r20) * s;
            qz = (r10 - r01) * s;
        } else {
            // If not in the “trace > 0” case, find which diagonal is largest and proceed.
            if (r00 > r11 and r00 > r22) {
                const s = 2.0 * math.sqrt(1.0 + r00 - r11 - r22);
                qw = (r21 - r12) / s;
                qx = 0.25 * s;
                qy = (r01 + r10) / s;
                qz = (r02 + r20) / s;
            } else if (r11 > r22) {
                const s = 2.0 * math.sqrt(1.0 + r11 - r00 - r22);
                qw = (r02 - r20) / s;
                qx = (r01 + r10) / s;
                qy = 0.25 * s;
                qz = (r12 + r21) / s;
            } else {
                const s = 2.0 * math.sqrt(1.0 + r22 - r00 - r11);
                qw = (r10 - r01) / s;
                qx = (r02 + r20) / s;
                qy = (r12 + r21) / s;
                qz = 0.25 * s;
            }
        }

        // Optional: normalize the quaternion to avoid floating error drift
        var q = Quaternion.init(qx, qy, qz, qw);
        q = q.normalize();

        return .{ .translation = _translation, .rotation = q, .scale = _scale };
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
        const data: [4 * 4]f32 = .{0} ** 16;

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

    pub inline fn get_position(self: Self) Vec3 {
        return Vec3.init(self.base.data[12], self.base.data[13], self.base.data[14]);
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

pub const Quaternion = struct {
    const Self = @This();

    data: @Vector(4, f32), // [x, y, z, w]

    pub fn init(_x: f32, _y: f32, _z: f32, _w: f32) Self {
        return Self{ .data = .{ _x, _y, _z, _w } };
    }

    pub fn init_from_arr(arr: [4]f32) Self {
        return Self{ .data = arr };
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

    pub inline fn set_x(self: *Quaternion, _x: f32) void {
        self.data[0] = _x;
    }
    pub inline fn set_y(self: *Quaternion, _y: f32) void {
        self.data[1] = _y;
    }
    pub inline fn set_z(self: *Quaternion, _z: f32) void {
        self.data[2] = _z;
    }
    pub inline fn set_w(self: *Quaternion, _w: f32) void {
        self.data[3] = _w;
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

    pub fn from_mat3(m: Mat3) Self {
        const mat = m.base.data;
        const trace = mat[0] + mat[4] + mat[8];
        var result = Self{ .data = .{ 0, 0, 0, 0 } };

        if (trace > 0) {
            const S = @sqrt(trace + 1.0) * 2;
            result.data[0] = (mat[7] - mat[5]) / S;
            result.data[1] = (mat[2] - mat[6]) / S;
            result.data[2] = (mat[3] - mat[1]) / S;
            result.data[3] = 0.25 * S;
        } else if ((mat[0] > mat[4]) and (mat[0] > mat[8])) {
            const S = @sqrt(1.0 + mat[0] - mat[4] - mat[8]) * 2;
            result.data[0] = 0.25 * S;
            result.data[1] = (mat[1] + mat[3]) / S;
            result.data[2] = (mat[2] + mat[6]) / S;
            result.data[3] = (mat[7] - mat[5]) / S;
        } else if (mat[4] > mat[8]) {
            const S = @sqrt(1.0 + mat[4] - mat[0] - mat[8]) * 2;
            result.data[0] = (mat[1] + mat[3]) / S;
            result.data[1] = 0.25 * S;
            result.data[2] = (mat[5] + mat[7]) / S;
            result.data[3] = (mat[2] - mat[6]) / S;
        } else {
            const S = @sqrt(1.0 + mat[8] - mat[0] - mat[4]) * 2;
            result.data[0] = (mat[2] + mat[6]) / S;
            result.data[1] = (mat[5] + mat[7]) / S;
            result.data[2] = 0.25 * S;
            result.data[3] = (mat[3] - mat[1]) / S;
        }

        // Normalize the quaternion
        return result.normalize();
    }

    pub fn to_mat3(self: Self) Mat3 {
        _ = self;
        @panic("Not Implemented");
    }

    pub fn from_mat4(m: Mat4) Self {
        _ = m;
        @panic("Not Implemented");
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

        var bx = b.x();
        var by = b.y();
        var bz = b.z();
        var bw = b.w();

        var cos_half_theta = aw * bw + ax * bx + ay * by + az * bz;

        // Flip b if on opposite hemisphere to ensure shortest path
        if (cos_half_theta < 0.0) {
            bx = -bx;
            by = -by;
            bz = -bz;
            bw = -bw;
            cos_half_theta = -cos_half_theta;
        }

        if (cos_half_theta > 0.9995) {
            // Quaternions are very close, use linear interpolation
            return Self.normalize(Self.init(
                ax + t * (bx - ax),
                ay + t * (by - ay),
                az + t * (bz - az),
                aw + t * (bw - aw),
            ));
        } else {
            const half_theta = std.math.acos(cos_half_theta);
            const sin_half_theta = @sin(half_theta);
            const ratio_a = @sin((1 - t) * half_theta) / sin_half_theta;
            const ratio_b = @sin(t * half_theta) / sin_half_theta;

            return Self.init(
                ax * ratio_a + bx * ratio_b,
                ay * ratio_a + by * ratio_b,
                az * ratio_a + bz * ratio_b,
                aw * ratio_a + bw * ratio_b,
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
