const std = @import("std");

pub fn identity() [16]f32 {
    return .{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
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
pub fn lookAt(eye: [3]f32, center: [3]f32, up: [3]f32) [16]f32 {
    const f = normalize(subtract(center, eye));
    const s = normalize(cross(f, up));
    const u = cross(s, f);

    return .{
        s[0],         u[0],         -f[0],       0.0,
        s[1],         u[1],         -f[1],       0.0,
        s[2],         u[2],         -f[2],       0.0,
        -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0,
    };
}

// Helper functions
pub fn subtract(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

pub fn normalize(v: [3]f32) [3]f32 {
    const length = std.math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return .{ v[0] / length, v[1] / length, v[2] / length };
}

pub fn cross(a: [3]f32, b: [3]f32) [3]f32 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn dot(a: [3]f32, b: [3]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn radians(degrees: f32) f32 {
    return degrees * (std.math.pi / 180.0);
}

fn tan(x: f32) f32 {
    return @tan(x);
}
