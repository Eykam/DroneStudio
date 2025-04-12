const std = @import("std");
const Node = @import("Node.zig");
const Mesh = @import("Mesh.zig");
const Math = @import("Math.zig");
const Pipeline = @import("Pipeline.zig");
const gl = @import("bindings/gl.zig");

const glfw = gl.glfw;
const glad = gl.glad;
const Viewport = Pipeline.Viewport;
const Mat3 = Math.Mat3;
const Mat4 = Math.Mat4;
const Vec3 = Math.Vec3;
const Quaternion = Math.Quaternion;

pub const CameraManager = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    scene_width: f32,
    scene_height: f32,
    viewports: std.StringArrayHashMap(Viewport) = undefined,
    cameras: std.StringArrayHashMap(*Camera),
    active_cameras: std.StringArrayHashMap(bool),
    main_camera: ?*Camera = null,

    pub fn init(allocator: std.mem.Allocator, width: f32, height: f32) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .cameras = std.StringArrayHashMap(*Camera).init(allocator),
            .active_cameras = std.StringArrayHashMap(bool).init(allocator),
            .viewports = std.StringArrayHashMap(Viewport).init(allocator),
            .scene_width = width,
            .scene_height = height,
        };

        return self;
    }

    pub fn deinit(self: Self) void {
        self.cameras.deinit();
        self.active_cameras.deinit();
    }

    pub fn register_camera(self: *Self, camera: *Camera) !void {
        const camera_name = camera.get_base().name;

        try self.cameras.put(camera_name, camera);
        try self.active_cameras.put(camera_name, true);

        if (self.main_camera == null) {
            self.main_camera = camera;
        }

        const fbo_width: i32 = @intFromFloat(self.scene_width); // Main camera: 95% of screen width, max 1920px
        const fbo_height: i32 = @intFromFloat(self.scene_height); // Main camera: 70% of screen height, max 1080px

        const viewport = try Viewport.init(
            self.allocator,
            camera_name,
            fbo_width,
            fbo_height,
        );
        try self.viewports.put(camera_name, viewport);
    }

    pub fn set_active(self: *Self, name: []const u8) void {
        self.main_camera = self.cameras.get(name);
        self.main_camera.?.get_base().reset();
    }

    pub fn update_viewports(self: *Self) !bool {
        // Get the window manager to find the SensorManagementWindow instance
        // Clear existing viewports
        var viewport_it = self.viewports.iterator();
        while (viewport_it.next()) |entry| {
            var viewport = entry.value_ptr;
            viewport.deinit();
        }
        self.viewports.clearRetainingCapacity();

        // Get active viewport selections
        var active_viewport_count: usize = 0;
        var it = self.active_cameras.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.*) {
                active_viewport_count += 1;
            }
        }

        if (active_viewport_count == 0) {
            // No active viewports, nothing to do
            return false;
        }

        // Create viewports for each active camera with appropriate resolutions
        it = self.active_cameras.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.*) {
                const camera_name = entry.key_ptr.*;

                // Use higher resolution for main camera, reasonable resolution for others
                const fbo_width: i32 = @intFromFloat(self.scene_width); // Main camera: 95% of screen width, max 1920px
                const fbo_height: i32 = @intFromFloat(self.scene_height); // Main camera: 70% of screen height, max 1080px

                // Create the viewport with appropriate resolution
                const viewport = try Viewport.init(
                    self.allocator,
                    camera_name,
                    fbo_width,
                    fbo_height,
                );

                try self.viewports.put(camera_name, viewport);
            }
        }

        return false;
    }
};

pub const Camera = union(enum) {
    const Self = @This();

    Free: FreeCamera,
    DroneControl: DroneControlCamera,
    SensorCamera: SensorCamera,

    pub fn get_base(self: *Self) *CameraBase {
        return switch (self.*) {
            .Free => |*fc| &fc.base,
            .DroneControl => |*dc| &dc.base,
            .SensorCamera => |*sc| &sc.base,
        };
    }

    pub fn process_key_input(self: *Self) void {
        switch (self.*) {
            .Free => |*fc| fc.process_key_input(),
            .DroneControl => |*dc| dc.process_key_input(),
            else => {},
        }
    }

    pub fn process_mouse_input(self: *Self, xoffset: f64, yoffset: f64) void {
        switch (self.*) {
            .Free => |*fc| fc.process_mouse_input(xoffset, yoffset),
            .DroneControl => |*dc| dc.process_mouse_input(xoffset, yoffset),
            else => {},
        }
    }

    pub fn get_view_matrix(self: *Self) Mat4 {
        return switch (self.*) {
            .Free => |*fc| fc.get_view_matrix(),
            .DroneControl => |*dc| dc.get_view_matrix(),
            .SensorCamera => |*sc| sc.get_view_matrix(),
        };
    }

    pub fn get_projection_matrix(self: *Self) Mat4 {
        return switch (self.*) {
            .Free => |*fc| fc.base.get_projection(),
            .DroneControl => |*dc| dc.base.get_projection(),
            .SensorCamera => |*sc| sc.get_projection_matrix(),
        };
    }

    pub fn process_scroll_wheel(self: *Self, zoom: f32) void {
        switch (self.*) {
            .Free => |*fc| {
                fc.process_scroll_wheel(zoom);
            },
            else => {},
        }
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .Free => |fc| {
                fc.deinit(allocator);
            },
            .DroneControl => |dc| {
                dc.deinit(allocator);
            },
            .SensorCamera => |sc| {
                sc.deinit(allocator);
            },
        }
    }
};

pub const FreeCamera = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    base: CameraBase,
    speed: f32, // Movement speed
    sensitivity: f32, // Mouse sensitivity
    yaw: f32 = 0.0,
    pitch: f32 = 0.0,

    pub fn init(
        allocator: std.mem.Allocator,
        name: []const u8,
        aspect_ratio: f32,
        position: ?Vec3,
        speed: ?f32,
        sensitivity: ?f32,
    ) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .base = try CameraBase.init(allocator, name, position, aspect_ratio),
            .speed = speed orelse 2.5,
            .sensitivity = sensitivity orelse 0.1,
        };
        return self;
    }

    pub fn deinit(
        self: *Self,
    ) void {
        self.base.deinit(self.allocator);
    }

    /// For identical logic, just call the base:
    pub fn get_view_matrix(self: *Self) Mat4 {
        return Mat4.look_at(self.base.position, Vec3.add(self.base.position, self.base.front), self.base.up);
    }

    // Update the front, right, and up vectors based on current yaw and pitch
    pub fn update_direction(self: *Self) void {
        self.base.front = Vec3.from_angles(self.yaw, self.pitch).normalize();
        self.base.right = Vec3.cross(self.base.front, self.base.world_up).normalize();
        self.base.up = Vec3.cross(self.base.right, self.base.front).normalize();
    }

    /// Overridden method, different from first-person
    pub fn process_key_input(self: *Self) void {
        if (self.base.node.scene == null) return;

        const appState = self.base.node.scene.?.appState;
        const sprinting = appState.keys[@as(usize, glfw.GLFW_KEY_LEFT_SHIFT)];
        const velocity = self.speed * appState.delta_time *
            (if (sprinting) @as(f32, 2.0) else @as(f32, 1.0));

        var movement: *Vec3 = @constCast(&Vec3.zero());

        if (appState.keys[@as(usize, glfw.GLFW_KEY_W)]) {
            movement.add_inplace(self.base.front.scale(velocity));
        }
        if (appState.keys[@as(usize, glfw.GLFW_KEY_S)]) {
            movement.sub_inplace(self.base.front.scale(velocity));
        }
        if (appState.keys[@as(usize, glfw.GLFW_KEY_D)]) {
            movement.add_inplace(self.base.right.scale(velocity));
        }
        if (appState.keys[@as(usize, glfw.GLFW_KEY_A)]) {
            movement.sub_inplace(self.base.right.scale(velocity));
        }

        if (appState.fly) {
            self.base.position = self.base.position.add(movement.*);
            if (appState.keys[@as(usize, glfw.GLFW_KEY_SPACE)]) {
                self.base.position.set_y(self.base.position.y() + velocity);
            }
        } else {
            const grounded = Vec3.init(
                self.base.position.x(),
                1.75,
                self.base.position.z(),
            );

            const grounded_movement = Vec3.init(
                movement.x(),
                0,
                movement.z(),
            );

            self.base.position = grounded.add(grounded_movement);
        }
    }

    pub fn process_mouse_input(self: *Self, xoffset: f64, yoffset: f64) void {
        self.yaw += @as(f32, @floatCast(xoffset)) * self.sensitivity;
        self.pitch += @as(f32, @floatCast(yoffset)) * self.sensitivity;

        // if (constrain_pitch) {
        //     if (self.pitch > 89.0) {
        //         self.pitch = 89.0;
        //     }
        //     if (self.pitch < -89.0) {
        //         self.pitch = -89.0;
        //     }
        // }

        self.update_direction();
        self.base.update_frustum();
    }

    pub fn process_scroll_wheel(self: *Self, zoom: f32) void {
        self.base.fov = zoom;
    }
};

pub const DroneControlCamera = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    base: CameraBase,
    speed: f32, // Movement speed
    sensitivity: f32, // Mouse sensitivity

    pub fn init(
        allocator: std.mem.Allocator,
        name: []const u8,
        aspect_ratio: f32,
        position: ?Vec3,
        speed: ?f32,
        sensitivity: ?f32,
    ) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .base = try CameraBase.init(allocator, name, position, aspect_ratio),
            .speed = speed orelse 2.5,
            .sensitivity = sensitivity orelse 0.1,
        };
        return self;
    }

    pub fn deinit(
        self: *Self,
    ) void {
        self.base.deinit(self.allocator);
    }

    /// For identical logic, just call the base:
    pub fn get_view_matrix(self: *Self) Mat4 {
        const world_transform = self.base.node.world_transform.transpose();

        // Extract position from world transform matrix
        const m = world_transform.base.data;
        const world_position = Vec3.init(m[3], m[7], m[11]);

        // Extract orientation axes from world transform matrix
        const camera_front = Vec3.init(m[2], m[6], m[10]).scale(-1).normalize();
        const camera_up = Vec3.init(m[1], m[5], m[9]).normalize();

        // Create view matrix using lookAt
        return Mat4.look_at(world_position, world_position.add(camera_front), camera_up);
    }

    /// Overridden method, different from first-person
    pub fn process_key_input(self: *Self) void {
        if (self.base.node.parent == null or self.base.node.scene == null) return;

        const delta_time = self.base.node.scene.?.appState.delta_time;

        // Control rates
        const move_distance = 5.0 * delta_time;
        const yaw_speed = 120.0 * delta_time; // degrees per second

        const node = self.base.node.parent.?;
        const appState = self.base.node.scene.?.appState;

        // Move up along local Y-axis
        var vertical_movement: f32 = 0.0;
        const local_up = node.world_transform.get_up();
        if (appState.keys[@as(usize, glfw.GLFW_KEY_W)]) vertical_movement += move_distance;
        if (appState.keys[@as(usize, glfw.GLFW_KEY_S)]) vertical_movement -= move_distance;

        var drone_translation = Vec3.zero();
        if (vertical_movement != 0.0) {
            drone_translation = local_up.scale(vertical_movement);
        }

        // Yaw (A/D keys)
        var yaw_quat = Quaternion.identity();
        if (appState.keys[@as(usize, glfw.GLFW_KEY_A)]) {
            yaw_quat = Quaternion.from_axis_angle(local_up, -yaw_speed);
        }
        if (appState.keys[@as(usize, glfw.GLFW_KEY_D)]) {
            yaw_quat = Quaternion.from_axis_angle(local_up, yaw_speed);
        }

        // Update the drone's rotation
        node.translate(drone_translation);
        node.rotateWithQuaternion(yaw_quat);
    }

    pub fn process_mouse_input(self: *Self, xoffset: f64, yoffset: f64) void {
        if (self.base.node.parent == null) return;

        const node = self.base.node.parent.?;

        // Get basis vectors from matrix
        const local_forward = node.world_transform.get_forward().scale(-1);
        const local_right = node.world_transform.get_right();

        // Yaw (mouse left/right) - around world up axis
        const roll_angle = -@as(f32, @floatCast(xoffset)) * self.sensitivity;
        const roll_quat = Quaternion.from_axis_angle(local_forward, roll_angle);

        // Pitch (mouse up/down) - around local right axis
        const pitch_angle = @as(f32, @floatCast(yoffset)) * self.sensitivity;
        const pitch_quat = Quaternion.from_axis_angle(local_right, pitch_angle);

        var relative_rotation = Quaternion.identity();
        relative_rotation = relative_rotation.multiply(pitch_quat);
        relative_rotation = relative_rotation.multiply(roll_quat);
        relative_rotation = relative_rotation.normalize();

        node.rotateWithQuaternion(relative_rotation);
    }
};

pub const SensorCamera = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    base: CameraBase,

    // Camera intrinsics
    focal_length_mm: f32, // Focal length in millimeters
    sensor_width_mm: f32, // Sensor width in millimeters
    sensor_height_mm: f32, // Sensor height in millimeters
    resolution_width: u32, // Resolution width in pixels
    resolution_height: u32, // Resolution height in pixels

    // Derived properties
    focal_length_px: f32, // Focal length in pixels
    fov_horizontal: f32, // Horizontal field of view in degrees
    fov_vertical: f32, // Vertical field of view in degrees
    principal_point_x: f32, // Principal point x-coordinate (typically center of image)
    principal_point_y: f32, // Principal point y-coordinate (typically center of image)

    pub fn init(
        allocator: std.mem.Allocator,
        name: []const u8,
        aspect_ratio: f32,
        position: ?Vec3,
        focal_length_mm: ?f32,
        sensor_width_mm: ?f32,
        sensor_height_mm: ?f32,
        resolution_width: ?u32,
        resolution_height: ?u32,
    ) !*Self {
        const self = try allocator.create(Self);

        // Set defaults or use provided values
        const fl_mm = focal_length_mm orelse 35.0;
        const sens_width = sensor_width_mm orelse 36.0;
        const sens_height = sensor_height_mm orelse (sens_width / aspect_ratio);
        const res_width = resolution_width orelse 1920;
        const res_height = resolution_height orelse 1080;

        // Calculate derived properties
        const fl_px = fl_mm * @as(f32, @floatFromInt(res_width)) / sens_width;
        const fov_h = 2.0 * Math.degrees(std.math.atan(sens_width / (2.0 * fl_mm)));
        const fov_v = 2.0 * Math.degrees(std.math.atan(sens_height / (2.0 * fl_mm)));

        // Initialize the camera with calculated properties
        self.* = Self{
            .allocator = allocator,
            .base = try CameraBase.init(
                allocator,
                name,
                position,
                aspect_ratio,
            ),
            .focal_length_mm = fl_mm,
            .sensor_width_mm = sens_width,
            .sensor_height_mm = sens_height,
            .resolution_width = res_width,
            .resolution_height = res_height,
            .focal_length_px = fl_px,
            .fov_horizontal = fov_h,
            .fov_vertical = fov_v,
            .principal_point_x = @as(f32, @floatFromInt(res_width)) / 2.0,
            .principal_point_y = @as(f32, @floatFromInt(res_height)) / 2.0,
        };

        // Update the camera's frustum to match the calculated FOV
        self.base.fov = self.fov_vertical;
        self.base.update_frustum();

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.base.deinit(self.allocator);
    }

    /// Get the camera matrix (intrinsic parameters)
    pub fn get_camera_matrix(self: *Self) Mat3 {
        var camera_matrix = Mat3.zero();

        // Set the focal lengths
        camera_matrix.base.data[0] = self.focal_length_px; // fx
        camera_matrix.base.data[4] = self.focal_length_px; // fy (assuming square pixels)

        // Set the principal point
        camera_matrix.base.data[2] = self.principal_point_x; // cx
        camera_matrix.base.data[5] = self.principal_point_y; // cy

        // Set the identity part
        camera_matrix.base.data[8] = 1.0; // bottom-right element

        return camera_matrix;
    }

    /// Get the projection matrix with intrinsics
    pub fn get_projection_matrix(self: *Self) Mat4 {
        // Create a projection matrix that incorporates the camera intrinsics

        // Calculate parameters for OpenGL-style projection matrix
        const n = self.base.near_plane;
        const f = self.base.far_plane;
        const aspect = @as(f32, @floatFromInt(self.resolution_width)) / @as(f32, @floatFromInt(self.resolution_height));

        // Note: we use the vertical FOV for the projection matrix
        return Mat4.perspective(self.fov_vertical, aspect, n, f);
    }

    /// Get view matrix based on camera's world transform
    pub fn get_view_matrix(self: *Self) Mat4 {
        const world_transform = self.base.node.world_transform;

        // Extract position from world transform matrix (assuming correct matrix layout)
        const m = world_transform.base.data;
        const world_position = Vec3.init(m[12], m[13], m[14]);

        // Extract orientation axes from world transform matrix
        const camera_up = Vec3.init(m[4], m[5], m[6]).normalize();
        const camera_front = Vec3.init(m[8], m[9], m[10]).normalize();

        // Create view matrix using lookAt with negative front direction
        // (because camera looks down negative z-axis in OpenGL convention)
        return Mat4.look_at(world_position, world_position.add(camera_front.scale(-1)), camera_up);
    }

    /// Calculate and set FOV based on current focal length and sensor size
    pub fn update_fov_from_intrinsics(self: *Self) void {
        self.fov_horizontal = 2.0 * Math.degrees(std.math.atan(self.sensor_width_mm / (2.0 * self.focal_length_mm)));
        self.fov_vertical = 2.0 * Math.degrees(std.math.atan(self.sensor_height_mm / (2.0 * self.focal_length_mm)));

        // Update the base camera's FOV
        self.base.fov = self.fov_vertical;
        self.base.update_frustum();
    }

    /// Set a new focal length and update derived properties
    pub fn set_focal_length(self: *Self, focal_length_mm: f32) void {
        self.focal_length_mm = focal_length_mm;
        self.focal_length_px = focal_length_mm * @as(f32, @floatFromInt(self.resolution_width)) / self.sensor_width_mm;
        self.update_fov_from_intrinsics();
    }
};

const CameraBase = struct {
    const Self = @This();

    name: []const u8,
    node: *Node,
    position: Vec3,
    front: Vec3 = Vec3.init(0.0, 0.0, -1.0),
    up: Vec3 = Vec3.init(0.0, 1.0, 0.0),
    right: Vec3 = Vec3.init(1.0, 0.0, 0.0),

    start_position: Vec3,
    start_front: Vec3,
    start_up: Vec3,
    start_right: Vec3,

    world_up: Vec3 = Vec3.init(0.0, 1.0, 0.0),
    fov: f32 = 45.0, // Field of view in degrees
    aspect_ratio: f32 = 1.0, // Aspect ratio of the viewport
    near_plane: f32 = 0.1, // Near clipping plane
    far_plane: f32 = 100.0, // Far clipping plane
    frustum_debug_near: f32 = 0.1,
    frustum_debug_far: f32 = 1.0,

    frustum_node: ?*Node = null,
    debug_mode: bool = false,

    pub const Cameras = union(enum) {};

    // Initialize a new Camera
    pub fn init(allocator: std.mem.Allocator, name: []const u8, _position: ?Vec3, aspect_ratio: f32) !Self {
        const position = _position orelse Vec3.init(0.0, 1.0, 5.0);
        const node = try Node.init(allocator, null, null, null);
        node.setPosition(position.x(), position.y(), position.z());

        // Create the CameraNode
        var camera_node = Self{
            .name = name,
            .node = node,
            .position = position,
            .aspect_ratio = aspect_ratio,
            .start_position = undefined,
            .start_front = undefined,
            .start_up = undefined,
            .start_right = undefined,
        };

        camera_node.start_position = camera_node.position;
        camera_node.start_front = camera_node.front;
        camera_node.start_up = camera_node.up;
        camera_node.start_right = camera_node.right;

        // Create frustum visualization if in debug mode
        if (camera_node.debug_mode) {
            try camera_node.create_frustum_visualization(allocator);
        }

        return camera_node;
    }

    pub fn reset(self: *Self) void {
        self.position = self.start_position;
        self.front = self.start_front;
        self.up = self.start_up;
        self.right = self.start_right;
    }

    pub fn process_zoom(self: *Self, yoffset: f64) void {
        self.fov -= @as(f32, @floatCast(yoffset));

        if (self.fov < 1.0) {
            self.fov = 1.0;
        }
        if (self.fov > 120.0) {
            self.fov = 120.0;
        }

        self.update_frustum();
    }

    pub fn set_aspect_ratio(self: *Self, width: f32, height: f32) void {
        self.aspect_ratio = width / height;
        self.update_frustum();
    }

    pub fn create_frustum_visualization(self: *Self, allocator: std.mem.Allocator) !void {
        // Create a mesh for the frustum visualization
        const vertices = try self.generate_frustum_vertices(allocator);
        defer allocator.free(vertices);

        const frustum_node = try Node.init(allocator, vertices, null, Mesh.gen_draw(glad.GL_LINES));
        frustum_node.mesh.?.drawType = glad.GL_LINES;

        // Set frustum color
        for (frustum_node.mesh.?.vertices) |*vertex| {
            vertex.color = .{ 0.0, 1.0, 1.0 }; // Cyan color for camera frustum
        }

        // Add the frustum node as a child to the camera node
        try self.node.addChild(frustum_node);

        self.frustum_node = frustum_node;
    }

    fn generate_frustum_vertices(self: *Self, allocator: std.mem.Allocator) ![]Mesh.Vertex {
        // Calculate frustum corners in view space
        const tan_half_fov = @tan(Math.radians(self.fov / 2.0));
        const near_height = 2.0 * tan_half_fov * self.frustum_debug_near;
        const near_width = near_height * self.aspect_ratio;
        const far_height = 2.0 * tan_half_fov * self.frustum_debug_far;
        const far_width = far_height * self.aspect_ratio;

        // Calculate the 8 corners of the frustum
        // Near plane corners
        const near_top_left = Vec3.init(-near_width / 2.0, near_height / 2.0, -self.frustum_debug_near);
        const near_top_right = Vec3.init(near_width / 2.0, near_height / 2.0, -self.frustum_debug_near);
        const near_bottom_left = Vec3.init(-near_width / 2.0, -near_height / 2.0, -self.frustum_debug_near);
        const near_bottom_right = Vec3.init(near_width / 2.0, -near_height / 2.0, -self.frustum_debug_near);

        // Far plane corners
        const far_top_left = Vec3.init(-far_width / 2.0, far_height / 2.0, -self.frustum_debug_far);
        const far_top_right = Vec3.init(far_width / 2.0, far_height / 2.0, -self.frustum_debug_far);
        const far_bottom_left = Vec3.init(-far_width / 2.0, -far_height / 2.0, -self.frustum_debug_far);
        const far_bottom_right = Vec3.init(far_width / 2.0, -far_height / 2.0, -self.frustum_debug_far);

        // Create vertices for the lines representing the frustum edges
        var vertices = try allocator.alloc(Mesh.Vertex, 24);

        // Near plane
        vertices[0] = .{ .position = .{ near_top_left.x(), near_top_left.y(), near_top_left.z() }, .color = .{ 0.0, 1.0, 1.0 } };
        vertices[1] = .{ .position = .{ near_top_right.x(), near_top_right.y(), near_top_right.z() }, .color = .{ 0.0, 1.0, 1.0 } };

        vertices[2] = .{ .position = .{ near_top_right.x(), near_top_right.y(), near_top_right.z() }, .color = .{ 0.0, 1.0, 1.0 } };
        vertices[3] = .{ .position = .{ near_bottom_right.x(), near_bottom_right.y(), near_bottom_right.z() }, .color = .{ 0.0, 1.0, 1.0 } };

        vertices[4] = .{ .position = .{ near_bottom_right.x(), near_bottom_right.y(), near_bottom_right.z() }, .color = .{ 0.0, 1.0, 1.0 } };
        vertices[5] = .{ .position = .{ near_bottom_left.x(), near_bottom_left.y(), near_bottom_left.z() }, .color = .{ 0.0, 1.0, 1.0 } };

        vertices[6] = .{ .position = .{ near_bottom_left.x(), near_bottom_left.y(), near_bottom_left.z() }, .color = .{ 0.0, 1.0, 1.0 } };
        vertices[7] = .{ .position = .{ near_top_left.x(), near_top_left.y(), near_top_left.z() }, .color = .{ 0.0, 1.0, 1.0 } };

        // Far plane
        vertices[8] = .{ .position = .{ far_top_left.x(), far_top_left.y(), far_top_left.z() }, .color = .{ 0.0, 1.0, 1.0 } };
        vertices[9] = .{ .position = .{ far_top_right.x(), far_top_right.y(), far_top_right.z() }, .color = .{ 0.0, 1.0, 1.0 } };

        vertices[10] = .{ .position = .{ far_top_right.x(), far_top_right.y(), far_top_right.z() }, .color = .{ 0.0, 1.0, 1.0 } };
        vertices[11] = .{ .position = .{ far_bottom_right.x(), far_bottom_right.y(), far_bottom_right.z() }, .color = .{ 0.0, 1.0, 1.0 } };

        vertices[12] = .{ .position = .{ far_bottom_right.x(), far_bottom_right.y(), far_bottom_right.z() }, .color = .{ 0.0, 1.0, 1.0 } };
        vertices[13] = .{ .position = .{ far_bottom_left.x(), far_bottom_left.y(), far_bottom_left.z() }, .color = .{ 0.0, 1.0, 1.0 } };

        vertices[14] = .{ .position = .{ far_bottom_left.x(), far_bottom_left.y(), far_bottom_left.z() }, .color = .{ 0.0, 1.0, 1.0 } };
        vertices[15] = .{ .position = .{ far_top_left.x(), far_top_left.y(), far_top_left.z() }, .color = .{ 0.0, 1.0, 1.0 } };

        // Connections between near and far planes
        vertices[16] = .{ .position = .{ near_top_left.x(), near_top_left.y(), near_top_left.z() }, .color = .{ 0.0, 1.0, 1.0 } };
        vertices[17] = .{ .position = .{ far_top_left.x(), far_top_left.y(), far_top_left.z() }, .color = .{ 0.0, 1.0, 1.0 } };

        vertices[18] = .{ .position = .{ near_top_right.x(), near_top_right.y(), near_top_right.z() }, .color = .{ 0.0, 1.0, 1.0 } };
        vertices[19] = .{ .position = .{ far_top_right.x(), far_top_right.y(), far_top_right.z() }, .color = .{ 0.0, 1.0, 1.0 } };

        vertices[20] = .{ .position = .{ near_bottom_right.x(), near_bottom_right.y(), near_bottom_right.z() }, .color = .{ 0.0, 1.0, 1.0 } };
        vertices[21] = .{ .position = .{ far_bottom_right.x(), far_bottom_right.y(), far_bottom_right.z() }, .color = .{ 0.0, 1.0, 1.0 } };

        vertices[22] = .{ .position = .{ near_bottom_left.x(), near_bottom_left.y(), near_bottom_left.z() }, .color = .{ 0.0, 1.0, 1.0 } };
        vertices[23] = .{ .position = .{ far_bottom_left.x(), far_bottom_left.y(), far_bottom_left.z() }, .color = .{ 0.0, 1.0, 1.0 } };

        return vertices;
    }

    // Update the frustum visualization when camera parameters change
    pub fn update_frustum(self: *Self) void {
        if (self.debug_mode and self.frustum_node != null) {
            // Generate new frustum vertices
            const vertices = self.generate_frustum_vertices(self.node.allocator) catch return;
            defer self.node.allocator.free(vertices);

            // Update the frustum mesh
            const mesh = self.frustum_node.?.mesh.?;
            for (0..mesh.vertices.len) |i| {
                if (i < vertices.len) {
                    mesh.vertices[i] = vertices[i];
                }
            }

            // Update VBO
            glad.glBindBuffer(glad.GL_ARRAY_BUFFER, mesh.meta.VBO);
            glad.glBufferData(
                glad.GL_ARRAY_BUFFER,
                @intCast(mesh.vertices.len * @sizeOf(Mesh.Vertex)),
                mesh.vertices.ptr,
                glad.GL_STATIC_DRAW,
            );
        }
    }

    // Toggle debug visualization
    pub fn toggle_debug_mode(self: *Self, allocator: std.mem.Allocator) !void {
        self.debug_mode = !self.debug_mode;

        if (self.debug_mode) {
            if (self.frustum_node == null) {
                try self.create_frustum_visualization(allocator);
            }
        } else {
            if (self.frustum_node != null) {
                // TODO: Remove frustum node from children
                self.frustum_node = null;
            }
        }
    }

    // Deinitialize the camera node
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.node.deinit();
        allocator.destroy(self);
    }

    pub fn get_projection(self: *Self) Mat4 {
        return Mat4.perspective(self.fov, self.aspect_ratio, self.near_plane, self.far_plane);
    }
};
