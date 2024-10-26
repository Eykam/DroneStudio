const std = @import("std");
const Mesh = @import("Shape.zig").Mesh;

pub const Vec3 = struct {
    x: f32,
    y: f32,
    z: f32,

    pub fn normalize(self: Vec3) Vec3 {
        const length = @sqrt(self.x * self.x + self.y * self.y + self.z * self.z);
        if (length == 0) return self;
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

pub const KalmanState = struct {
    angle: f32, // Current angle estimate
    bias: f32, // Gyro bias estimate
    P: [2][2]f32, // Error covariance matrix
    Q_angle: f32, // Process noise for angle
    Q_bias: f32, // Process noise for bias
    R_measure: f32, // Measurement noise
    dt: f32, // Time step
};

pub fn initKalmanState(initial_angle: f32, initial_bias: f32) KalmanState {
    return KalmanState{
        .angle = initial_angle,
        .bias = initial_bias,
        .P = .{
            .{ 0.0001, 0.0 },
            .{ 0.0, 0.0001 },
        },
        .Q_angle = 0.001,
        .Q_bias = 0.003,
        .R_measure = 0.03,
        .dt = 1.0 / 1000.0, // Assuming 100Hz update rate
    };
}

pub fn updateKalman(state: *KalmanState, accel_angle: f32, gyro_rate: f32) f32 {
    // Predict step
    const rate = gyro_rate - state.bias;
    state.angle += state.dt * rate;

    // Update error covariance matrix
    state.P[0][0] += state.dt * (state.dt * state.P[1][1] - state.P[0][1] - state.P[1][0] + state.Q_angle);
    state.P[0][1] -= state.dt * state.P[1][1];
    state.P[1][0] -= state.dt * state.P[1][1];
    state.P[1][1] += state.Q_bias * state.dt;

    // Calculate Kalman gain
    const S = state.P[0][0] + state.R_measure;
    const K = .{
        state.P[0][0] / S,
        state.P[1][0] / S,
    };

    // Update step
    const y = accel_angle - state.angle;
    state.angle += K[0] * y;
    state.bias += K[1] * y;

    // Update error covariance matrix
    const P00_temp = state.P[0][0];
    const P01_temp = state.P[0][1];

    state.P[0][0] -= K[0] * P00_temp;
    state.P[0][1] -= K[0] * P01_temp;
    state.P[1][0] -= K[1] * P00_temp;
    state.P[1][1] -= K[1] * P01_temp;

    return state.angle;
}

// const pitch: [16]f32 = [_]f32{};
//
pub fn createRotationMatrix(yaw: f32, pitch: f32, roll: f32) [16]f32 {
    const cy = @cos(yaw); // cos of Y-axis rotation (yaw)
    const sy = @sin(yaw); // sin of Y-axis rotation (yaw)
    const cp = @cos(pitch); // cos of X-axis rotation (pitch)
    const sp = @sin(pitch); // sin of X-axis rotation (pitch)
    const cr = @cos(roll); // cos of Z-axis rotation (roll)
    const sr = @sin(roll); // sin of Z-axis rotation (roll)

    // Combined rotation matrix (in row-major order)
    return .{
        // First column
        cy * cr,                 -cy * sr,                sy,       0,
        // Second column
        sp * sy * cr + cp * sr,  -sp * sy * sr + cp * cr, -sp * cy, 0,
        // Third column
        -cp * sy * cr + sp * sr, cp * sy * sr + sp * cr,  cp * cy,  0,
        // Fourth column
        0,                       0,                       0,        1,
    };
}
pub fn updateModelMatrix(mesh: *Mesh, accel: Vec3, gyro: Vec3) !void {
    // const sensitivity: f32 = 1.0;

    // const norm_accel = accel.normalize();

    // Scale down the gyro values as well
    const scaled_gyro = Vec3{
        .x = radians(gyro.x),
        .y = radians(gyro.y),
        .z = radians(gyro.z),
    };

    const accel_roll = angleRoll(accel);
    // const accel_pitch = anglePitch(scaled_accel);

    // Update Kalman filter with scaled values
    const filtered_roll = updateKalman(&mesh.rollKalman, accel_roll, scaled_gyro.x);
    // const filtered_pitch = updateKalman(&mesh.pitchKalman, accel_pitch, scaled_gyro.y);

    //   mesh.yaw += scaled_gyro.z * mesh.rollKalman.dt; // Increment yaw using gyro z-axis data

    mesh.modelMatrix = createRotationMatrix(0.0, // Yaw (rotation around Y-axis)
        0.0, // Pitch (rotation around X-axis)
        filtered_roll // Roll (rotation around Z-axis)
    );
}

pub fn identity() [16]f32 {
    return .{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    };
}

pub fn rotate_x(angle_deg: f32) [16]f32 {
    const angle_rad = angle_deg * (std.math.pi / 180.0);
    const c = std.math.cos(angle_rad);
    const s = std.math.sin(angle_rad);

    return .{
        1.0, 0.0, 0.0, 0.0,
        0.0, c,   -s,  0.0,
        0.0, s,   c,   0.0,
        0.0, 0.0, 0.0, 1.0,
    };
}

pub fn rotate_y(angle_deg: f32) [16]f32 {
    const angle_rad = angle_deg * (std.math.pi / 180.0);
    const c = std.math.cos(angle_rad);
    const s = std.math.sin(angle_rad);

    return .{
        c,   0.0, s,   0.0,
        0.0, 1.0, 0.0, 0.0,
        -s,  0.0, c,   0.0,
        0.0, 0.0, 0.0, 1.0,
    };
}

pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) [16]f32 {
    return .{
        2.0 / (right - left), 0.0,                  0.0,                 -(right + left) / (right - left),
        0.0,                  2.0 / (top - bottom), 0.0,                 -(top + bottom) / (top - bottom),
        0.0,                  0.0,                  -2.0 / (far - near), -(far + near) / (far - near),
        0.0,                  0.0,                  0.0,                 1.0,
    };
}

pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) [16]f32 {
    const rad = radians(fov);
    const tan_half_fov = tan(rad / 2.0);
    var mat: [16]f32 = .{0} ** 16;
    mat[0] = 1.0 / (aspect * tan_half_fov);
    mat[5] = 1.0 / (tan_half_fov);
    mat[10] = -(far + near) / (far - near);
    mat[11] = -1.0;
    mat[14] = -(2.0 * far * near) / (far - near);
    return mat;
}

// Function to create a simple view matrix (camera at (0,0,5) looking at origin)
pub fn lookAt(eye: Vec3, center: Vec3, up: Vec3) [16]f32 {
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

pub fn multiply_matrices(a: [16]f32, b: [16]f32) [16]f32 {
    var result: [16]f32 = .{0} ** 16;
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

fn tan(x: f32) f32 {
    return @tan(x);
}

fn atan(x: f32) f32 {
    return std.math.atan(x);
}

fn atan2(x: f32, y: f32) f32 {
    return std.math.atan2(x, y);
}

pub fn angleRoll(angles: Vec3) f32 {
    // Roll (rotation around X axis)
    // In OpenGL: Y is up, Z is backward
    return atan2(angles.y, angles.x);
}

pub fn anglePitch(angles: Vec3) f32 {
    // Pitch (rotation around Z axis)
    // In OpenGL: Y is up, Z is backward
    return atan2(-angles.x, @sqrt(angles.y * angles.y + angles.z * angles.z));
}
