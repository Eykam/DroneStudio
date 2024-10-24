const std = @import("std");
const Transformations = @import("Transformations.zig");
const Vec3 = Transformations.Vec3;

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
        .position = position orelse Vec3{ .x = 0.0, .y = 1.0, .z = 5.0 },
        .front = front orelse Vec3{ .x = 0.0, .y = 0.0, .z = -1.0 }, // Default front
        .up = Vec3{ .x = 0.0, .y = 1.0, .z = 0.0 }, // Default up
        .right = Vec3{ .x = 1.0, .y = 0.0, .z = 0.0 },
        .yaw = 0.0,
        .pitch = 0.0,
        .speed = 2.5,
        .sensitivity = 0.1,
    };
    camera.update_vectors();
    return camera;
}

// Update the front, right, and up vectors based on current yaw and pitch
pub fn update_vectors(self: *Self) void {
    self.front = Vec3.from_angles(self.yaw, self.pitch);
    self.right = Vec3.normalize(Vec3.cross(self.front, self.up));
}

// Move the camera forward
pub fn move_forward(self: *Self, delta_time: f32, sprinting: bool, debug: bool) void {
    var multiplier: f32 = 1;
    if (sprinting) multiplier *= 2;

    const velocity = self.speed * delta_time * multiplier;

    if (debug) {
        std.debug.print("==============\nForward\n", .{});
        std.debug.print("Initial: {d:.6}\n", .{[_]f32{
            self.position.x,
            self.position.y,
            self.position.z,
        }});
    }

    self.position = Vec3.add(self.position, Vec3{
        .x = self.front.x * velocity,
        .y = 0.0,
        .z = self.front.z * velocity,
    });

    if (debug) std.debug.print("New: {d:.6}\n", .{[_]f32{
        self.position.x,
        self.position.y,
        self.position.z,
    }});
}

// Move the camera backward
pub fn move_backward(self: *Self, delta_time: f32, sprinting: bool, debug: bool) void {
    var multiplier: f32 = 1;
    if (sprinting) multiplier *= 2;

    const velocity = self.speed * delta_time * multiplier;

    if (debug) {
        std.debug.print("==============\nBack\n", .{});
        std.debug.print("Initial: {d:.6}\n", .{[_]f32{
            self.position.x,
            self.position.y,
            self.position.z,
        }});
    }

    self.position = Vec3.sub(self.position, Vec3{
        .x = self.front.x * velocity,
        .y = 0.0,
        .z = self.front.z * velocity,
    });

    if (debug) std.debug.print("New: {d:.6}\n", .{[_]f32{
        self.position.x,
        self.position.y,
        self.position.z,
    }});
}

// Move the camera to the right
pub fn move_right(self: *Self, delta_time: f32, sprinting: bool, debug: bool) void {
    var multiplier: f32 = 1;
    if (sprinting) multiplier *= 2;

    const velocity = self.speed * delta_time * multiplier;

    if (debug) {
        std.debug.print("==============\nRight\n", .{});
        std.debug.print("Initial: {d:.6}\n", .{[_]f32{
            self.position.x,
            self.position.y,
            self.position.z,
        }});
    }

    self.position = Vec3.add(self.position, Vec3{
        .x = self.right.x * velocity,
        .y = 0.0,
        .z = self.right.z * velocity,
    });

    if (debug) std.debug.print("New: {d:.6}\n", .{[_]f32{
        self.position.x,
        self.position.y,
        self.position.z,
    }});
}

// Move the camera to the left
pub fn move_left(self: *Self, delta_time: f32, sprinting: bool, debug: bool) void {
    var multiplier: f32 = 1;
    if (sprinting) multiplier *= 2;

    const velocity = self.speed * delta_time * multiplier;

    if (debug) {
        std.debug.print("==============\nLeft\n", .{});
        std.debug.print("Initial: {d:.6}\n", .{[_]f32{
            self.position.x,
            self.position.y,
            self.position.z,
        }});
    }

    self.position = Vec3.sub(self.position, Vec3{
        .x = self.right.x * velocity,
        .y = 0.0,
        .z = self.right.z * velocity,
    });

    if (debug) std.debug.print("New: {d:.6}\n", .{[_]f32{
        self.position.x,
        self.position.y,
        self.position.z,
    }});
}

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

    self.update_vectors();
}

// Generate the view matrix using the lookAt function
pub fn get_view_matrix(self: *Self) [16]f32 {
    return Transformations.lookAt(self.position, Vec3.add(self.position, self.front), self.up);
}
