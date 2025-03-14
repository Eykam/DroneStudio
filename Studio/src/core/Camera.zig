const std = @import("std");
const Math = @import("Math.zig");

const Mat4 = Math.Mat4;
const Vec3 = Math.Vec3;

const Self = @This();

position: Vec3,
front: Vec3,
up: Vec3,
right: Vec3,
yaw: f32, // Rotation around Y-axis in degrees
pitch: f32, // Rotation around X-axis in degrees
speed: f32, // Movement speed
sensitivity: f32, // Mouse sensitivity

// Initialize a new Camera
pub fn init(position: ?Vec3, front: ?Vec3) Self {
    var camera = Self{
        .position = position orelse Vec3.init(0.0, 1.0, 5.0),
        .front = front orelse Vec3.init(0.0, 0.0, -1.0),
        .up = Vec3.init(0.0, 1.0, 0.0),
        .right = Vec3.init(1.0, 0.0, 0.0),
        .yaw = 0.0,
        .pitch = 0.0,
        .speed = 2.5,
        .sensitivity = 0.1,
    };

    camera.update_direction();
    return camera;
}

// Update the front, right, and up vectors based on current yaw and pitch
pub fn update_direction(self: *Self) void {
    self.front = Vec3.from_angles(self.yaw, self.pitch);
    self.right = Vec3.cross(self.front, self.up).normalize();
}

//TODO: Use Quaternions instead? Removes the need for constraining Euler angles
// Process mouse movement to update yaw and pitch
pub fn process_mouse_movement(self: *Self, xoffset: f64, yoffset: f64, aspectRatio: f32, constrain_pitch: bool) void {
    self.yaw += @as(f32, @floatCast(xoffset)) * self.sensitivity * aspectRatio;
    self.pitch += @as(f32, @floatCast(yoffset)) * self.sensitivity;

    if (constrain_pitch) {
        if (self.pitch > 89.0) {
            self.pitch = 89.0;
        }
        if (self.pitch < -89.0) {
            self.pitch = -89.0;
        }
    }

    self.update_direction();
}

// Generate the view matrix using the lookAt function
pub fn get_view_matrix(self: *Self) Mat4 {
    return Mat4.look_at(self.position, Vec3.add(self.position, self.front), self.up);
}
