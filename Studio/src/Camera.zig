const std = @import("std");
const Transformations = @import("Transformations.zig");
const Vec3 = Transformations.Vec3; // Adjust the import path accordingly

pub const Camera = struct {
    position: Vec3,
    front: Vec3,
    up: Vec3,
    right: Vec3,
    world_up: Vec3,
    yaw: f32, // Rotation around Y-axis in degrees
    pitch: f32, // Rotation around X-axis in degrees
    speed: f32, // Movement speed
    sensitivity: f32, // Mouse sensitivity

    // Initialize a new Camera
    pub fn init(position: Vec3, up: Vec3, yaw: f32, pitch: f32) Camera {
        var camera = Camera{
            .position = position,
            .world_up = up,
            .yaw = yaw,
            .pitch = pitch,
            .speed = 5.0,
            .sensitivity = 0.1,
            .front = Vec3{ .x = 0.0, .y = 0.0, .z = -1.0 }, // Default front
            .up = Vec3{ .x = 0.0, .y = 1.0, .z = 0.0 }, // Default up
            .right = Vec3{ .x = 1.0, .y = 0.0, .z = 0.0 }, // Default right
        };
        camera.update_vectors();
        return camera;
    }

    // Update the front, right, and up vectors based on current yaw and pitch
    pub fn update_vectors(self: *Camera) void {
        self.front = Vec3.from_angles(self.yaw, self.pitch);
        self.right = Vec3.normalize(Vec3.cross(self.front, self.world_up));
        self.up = Vec3.normalize(Vec3.cross(self.right, self.front));
    }

    // Move the camera forward
    pub fn move_forward(self: *Camera, delta_time: f32) void {
        const velocity = self.speed * delta_time;
        self.position = Vec3.add(self.position, Vec3{
            .x = self.front.x * velocity,
            .y = self.front.y * velocity,
            .z = self.front.z * velocity,
        });
    }

    // Move the camera backward
    pub fn move_backward(self: *Camera, delta_time: f32) void {
        const velocity = self.speed * delta_time;
        self.position = Vec3.sub(self.position, Vec3{
            .x = self.front.x * velocity,
            .y = self.front.y * velocity,
            .z = self.front.z * velocity,
        });
    }

    // Move the camera to the right
    pub fn move_right(self: *Camera, delta_time: f32) void {
        const velocity = self.speed * delta_time;
        self.position = Vec3.add(self.position, Vec3{
            .x = self.right.x * velocity,
            .y = self.right.y * velocity,
            .z = self.right.z * velocity,
        });
    }

    // Move the camera to the left
    pub fn move_left(self: *Camera, delta_time: f32) void {
        const velocity = self.speed * delta_time;
        self.position = Vec3.sub(self.position, Vec3{
            .x = self.right.x * velocity,
            .y = self.right.y * velocity,
            .z = self.right.z * velocity,
        });
    }

    // Process mouse movement to update yaw and pitch
    pub fn process_mouse_movement(self: *Camera, xoffset: f32, yoffset: f32, constrain_pitch: bool) void {
        self.yaw += xoffset * self.sensitivity;
        self.pitch += yoffset * self.sensitivity;

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
    pub fn get_view_matrix(self: *Camera) [16]f32 {
        return Transformations.lookAt(self.position, Vec3.add(self.position, self.front), self.up);
    }
};
