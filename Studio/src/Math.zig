const std = @import("std");
const Sensors = @import("Sensors.zig");
const SensorState = Sensors.SensorState;

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

//Todo: Use @Vector for SIMD
pub const Vec3 = struct {
    x: f32,
    y: f32,
    z: f32,

    pub fn normalize(self: Vec3) Vec3 {
        const length = @sqrt(self.x * self.x + self.y * self.y + self.z * self.z);

        if (length == 0) {
            std.debug.print("Vec3 with 0 Length Detected", .{});
            return self;
        }

        return Vec3{
            .x = self.x / length,
            .y = self.y / length,
            .z = self.z / length,
        };
    }

    pub fn add(a: Vec3, b: Vec3) Vec3 {
        return Vec3{
            .x = a.x + b.x,
            .y = a.y + b.y,
            .z = a.z + b.z,
        };
    }

    pub fn sub(a: Vec3, b: Vec3) Vec3 {
        return Vec3{
            .x = a.x - b.x,
            .y = a.y - b.y,
            .z = a.z - b.z,
        };
    }

    pub fn scale(self: Vec3, scalar: f32) Vec3 {
        return Vec3{
            .x = self.x * scalar,
            .y = self.y * scalar,
            .z = self.z * scalar,
        };
    }

    pub fn cross(a: Vec3, b: Vec3) Vec3 {
        return Vec3{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }

    pub fn dot(a: Vec3, b: Vec3) f32 {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    pub fn from_angles(yaw_deg: f32, pitch_deg: f32) Vec3 {
        const yaw = radians(yaw_deg);
        const pitch = radians(pitch_deg);

        const front = Vec3{
            .x = @cos(yaw) * @cos(pitch),
            .y = @sin(pitch),
            .z = @sin(yaw) * @cos(pitch),
        };
        return Vec3.normalize(front);
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
