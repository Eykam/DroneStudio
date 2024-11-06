const std = @import("std");
const Mesh = @import("Mesh.zig");
const Sensors = @import("Sensors.zig");
const SensorState = Sensors.SensorState;

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
    return atan(x);
}

fn atan2(x: f32, y: f32) f32 {
    return std.math.atan2(x, y);
}

pub inline fn angleRoll(accel: Vec3) f32 {
    // More stable roll calculation from accelerometer
    return atan2(-accel.x, @sqrt(accel.y * accel.y + accel.z * accel.z));
}

pub inline fn anglePitch(angles: Vec3) f32 {
    // Pitch (rotation around Z axis)
    // In OpenGL: Y is up, Z is backward
    return atan2(angles.z, @sqrt(angles.x * angles.x + angles.y * angles.y));
}

pub inline fn angleYaw(mag: Vec3, pitch: f32, roll: f32) f32 {
    const sin_roll = @sin(roll);
    const cos_roll = @cos(roll);
    const sin_pitch = @sin(pitch);
    const cos_pitch = @cos(pitch);

    // Tilt compensated magnetic field X component
    const bx = mag.x * cos_pitch +
        mag.y * sin_roll * sin_pitch -
        mag.z * cos_roll * sin_pitch;

    // Tilt compensated magnetic field Z component
    const bz = mag.y * cos_roll +
        mag.z * sin_roll;

    // Calculate yaw
    return atan2(bz, bx);
}

/// Creates a rotation matrix around the Z axis (Roll)
pub inline fn createRollMatrix(angle_rad: f32) [16]f32 {
    const c = @cos(angle_rad);
    const s = @sin(angle_rad);

    return .{
        c,   -s,  0.0, 0.0,
        s,   c,   0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    };
}

/// Creates a rotation matrix around the X axis (Pitch)
pub inline fn createPitchMatrix(angle_rad: f32) [16]f32 {
    const c = @cos(angle_rad);
    const s = @sin(angle_rad);

    return .{
        1.0, 0.0, 0.0, 0.0,
        0.0, c,   -s,  0.0,
        0.0, s,   c,   0.0,
        0.0, 0.0, 0.0, 1.0,
    };
}

/// Creates a rotation matrix around the Y axis (Yaw)
pub inline fn createYawMatrix(angle_rad: f32) [16]f32 {
    const c = @cos(angle_rad);
    const s = @sin(angle_rad);

    return .{
        c,   0.0, s,   0.0,
        0.0, 1.0, 0.0, 0.0,
        -s,  0.0, c,   0.0,
        0.0, 0.0, 0.0, 1.0,
    };
}
pub inline fn createRotationMatrix(yaw: f32, pitch: f32, roll: f32) [16]f32 {
    const rotation = multiply_matrices(
        createRollMatrix(roll),
        multiply_matrices(
            createPitchMatrix(pitch),
            createYawMatrix(yaw),
        ),
    );

    return rotation;
}

fn calculateAdaptiveAlpha(mag: Vec3, accel: Vec3) f32 {
    const MAG_STRENGTH_NOMINAL = 30; // Adjust based on your magnetometer's typical readings
    const ACCEL_MAGNITUDE_NOMINAL = 1; // Standard gravity

    // Calculate magnetic field strength
    const mag_strength = @sqrt(mag.x * mag.x + mag.y * mag.y + mag.z * mag.z);
    const mag_error = @abs(mag_strength - MAG_STRENGTH_NOMINAL) / MAG_STRENGTH_NOMINAL;

    // Calculate acceleration magnitude to detect motion
    const accel_magnitude = @sqrt(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z);
    const accel_error = @abs(accel_magnitude - ACCEL_MAGNITUDE_NOMINAL) / ACCEL_MAGNITUDE_NOMINAL;

    // Combine errors to determine filter weight
    const total_error = mag_error + accel_error;
    const base_alpha = 0.98; // Base gyro weight
    const min_alpha = 0.75; // Minimum gyro weight

    return @max(min_alpha, base_alpha - total_error);
}

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
        .dt = 1.0 / 850.0, // Assuming 950hz update rate
    };
}

pub fn updateKalman(state: *KalmanState, accel_angle: f32, gyro_rate: f32, delta_time: f32) f32 {
    // Predict step
    const rate = gyro_rate - state.bias;
    // _ = delta_time;
    state.dt = delta_time;
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

pub fn updateModelMatrix(
    mesh: *Mesh,
    accel: Vec3,
    gyro: Vec3,
    mag: Vec3,
    delta_time: f32,
    sensor_state: *SensorState,
) !void {
    const scaled_gyro = Vec3{
        .x = radians(gyro.x),
        .y = radians(gyro.y) * -1.0,
        .z = radians(gyro.z),
    };

    const accel_roll = angleRoll(accel);
    const accel_pitch = anglePitch(accel);

    // Update Kalman filter with scaled values
    const filtered_roll = updateKalman(&sensor_state.rollKalman, accel_roll, scaled_gyro.x, delta_time);
    const filtered_pitch = updateKalman(&sensor_state.pitchKalman, accel_pitch, scaled_gyro.z, delta_time);

    sensor_state.gyro_integrated_yaw += scaled_gyro.y * delta_time;

    if (sensor_state.mag_updated) {
        // Normalize magnetometer readings before processing
        const mag_length = @sqrt(mag.x * mag.x + mag.y * mag.y + mag.z * mag.z);
        const normalized_mag = Vec3{
            .x = mag.x / mag_length,
            .y = mag.y / mag_length,
            .z = mag.z / mag_length,
        };

        const mag_yaw = angleYaw(normalized_mag, filtered_pitch, filtered_roll);

        if (!sensor_state.mag_valid) {
            // sensor_state.gyro_integrated_yaw = mag_yaw;
            // sensor_state.previous_yaw = mag_yaw;
            sensor_state.gyro_integrated_yaw = 0.0;
            sensor_state.mag_valid = true;
            std.debug.print("Initial mag yaw: {d}\n", .{degrees(mag_yaw)});
        } else {
            // Handle wraparound for magnetic yaw
            var yaw_diff = mag_yaw - sensor_state.previous_yaw;
            if (yaw_diff > std.math.pi) {
                yaw_diff -= 2.0 * std.math.pi;
            } else if (yaw_diff < -std.math.pi) {
                yaw_diff += 2.0 * std.math.pi;
            }

            // Debug print the yaw values
            std.debug.print("Mag: {d:.2}, Gyro: {d:.2}, Diff: {d:.2}\n", .{
                degrees(mag_yaw),
                degrees(sensor_state.gyro_integrated_yaw),
                degrees(yaw_diff),
            });

            // Complementary filter with adaptive weights
            // const alpha = calculateAdaptiveAlpha(normalized_mag, accel);
            // sensor_state.gyro_integrated_yaw = alpha * sensor_state.gyro_integrated_yaw +
            //     (1.0 - alpha) * (sensor_state.previous_yaw + yaw_diff);
            // sensor_state.previous_yaw = mag_yaw; // Store the raw mag reading for next diff calculation
        }
    }

    mesh.modelMatrix = createRotationMatrix(
        sensor_state.gyro_integrated_yaw, // Yaw (rotation around Y-axis)
        filtered_pitch, // Pitch (rotation around X-axis)
        filtered_roll, // Roll (rotation around Z-axis)
    );
}
