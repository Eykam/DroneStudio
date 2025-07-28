const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const gl = @import("../../bindings/gl.zig");
const ECSManager = @import("../ECSManager.zig");
const Controller = @import("../components/Controller.zig");
const Transform = @import("../components/Transform.zig");
const Collisions = @import("../components/Collisions.zig");
const Renderer = @import("../components/Renderer.zig");
const Camera = @import("../components/Camera.zig");
const DroneCamera = @import("DroneCamera.zig");
const Frustum = @import("Frustum.zig");
const Mesh = @import("../../Mesh.zig");

const glfw = gl.glfw;
const Vec3 = Math.Vec3;
const CollisionSystem = Collisions.CollisionSystem;

/// ---------------------------------------------------------------------------
/// Tunable constants
const Defaults = struct {
    const thrust: f32 = 20.0; // m/s
    const yawRate: f32 = 2; // °/s
    const rollRate: f32 = 60;
    const pitchRate: f32 = 60; // ° per pixel
};

// Physics-aware movement: use forces if physics is available, otherwise direct transform
inline fn move(
    eid: Core.EntityID,
    tf: *Transform.TransformComponent,
    rigid_body: ?*Collisions.RigidBodyComponent,
    collision_system: ?*CollisionSystem,
    dir: Vec3,
    speed: f32,
    dt: f32,
) void {
    if (rigid_body != null and collision_system != null and rigid_body.?.mass > 0.0) {
        // Use physics force for dynamic bodies
        const force = dir.scale(speed * rigid_body.?.mass); // Scale by mass for consistent feel
        std.debug.print("Applying force to eid {d}: [{d:.2}, {d:.2}, {d:.2}]\n", .{ eid.id, force.data[0], force.data[1], force.data[2] });
        collision_system.?.applyCentralForce(eid, force.data);
    } else {
        // Fallback to direct transform manipulation for static/kinematic or non-physics entities
        std.debug.print("Using direct transform for eid {d} (rigid_body: {any})\n", .{ eid.id, rigid_body });
        tf.translate(dir.scale(speed * dt).data);
    }
}

// Physics-aware rotation: use torque if physics is available, otherwise direct transform
inline fn rotate(
    eid: Core.EntityID,
    tf: *Transform.TransformComponent,
    rigid_body: ?*Collisions.RigidBodyComponent,
    collision_system: ?*CollisionSystem,
    axis: Vec3,
    deg: f32,
    dt: f32,
) void {
    if (rigid_body != null and collision_system != null and rigid_body.?.mass > 0.0) {
        // Use physics torque for dynamic bodies - increased multiplier to match mouse sensitivity
        const torque = axis.scale(Math.radians(deg) * rigid_body.?.mass); // Scale by mass and convert to radians
        std.debug.print("Applying torque to eid {d}: [{d:.2}, {d:.2}, {d:.2}]\n", .{ eid.id, torque.data[0], torque.data[1], torque.data[2] });
        collision_system.?.applyTorque(eid, torque.data);
    } else {
        // Fallback to direct transform manipulation
        std.debug.print("Using direct rotation for eid {d}\n", .{eid.id});
        const q = Math.Quaternion.from_axis_angle(axis, deg * dt);
        tf.rotate(q);
    }
}

/// Key‑binding callbacks --------------------------------------------------
fn up(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, dt: f32) void {
    move(eid, tf, rigid_body, collision_system, tf.local_transform.get_up(), Defaults.thrust, dt);
}
fn down(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, dt: f32) void {
    move(eid, tf, rigid_body, collision_system, tf.local_transform.get_up().scale(-1), Defaults.thrust, dt);
}
fn yawLeft(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, dt: f32) void {
    rotate(eid, tf, rigid_body, collision_system, tf.local_transform.get_up(), -Defaults.yawRate, dt);
}
fn yawRight(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, dt: f32) void {
    rotate(eid, tf, rigid_body, collision_system, tf.local_transform.get_up(), Defaults.yawRate, dt);
}

/// Mouse‑look (pitch / roll) ----------------------------------------------
fn mouseLook(
    eid: Core.EntityID,
    tf: *Transform.TransformComponent,
    rigid_body: ?*Collisions.RigidBodyComponent,
    collision_system: ?*CollisionSystem,
    dx: f32, // pixels
    dy: f32,
    _: f64,
) void {
    const local_forward = tf.local_transform.get_forward();
    const local_right = tf.local_transform.get_right();

    // Remove dt scaling for mouse input - mouse movement is already frame-rate independent
    const roll_angle = @as(f32, @floatCast(dx)) * 0.1; // Reduced sensitivity
    const pitch_angle = -@as(f32, @floatCast(-dy)) * 0.1; // Reduced sensitivity

    if (rigid_body != null and collision_system != null and rigid_body.?.mass > 0.0) {

        // Use physics torque for dynamic bodies
        const roll_torque_vec = local_forward.scale(-Math.radians(roll_angle) * rigid_body.?.mass);
        const pitch_torque_vec = local_right.scale(-Math.radians(pitch_angle) * rigid_body.?.mass);

        // Combine the torques
        const combined_torque = Vec3.init(roll_torque_vec.x() + pitch_torque_vec.x(), roll_torque_vec.y() + pitch_torque_vec.y(), roll_torque_vec.z() + pitch_torque_vec.z());
        collision_system.?.applyTorque(eid, combined_torque.data);
    } else {
        // Fallback to direct transform manipulation
        const roll_quat = Math.Quaternion.from_axis_angle(local_forward, roll_angle);
        const pitch_quat = Math.Quaternion.from_axis_angle(local_right, pitch_angle);

        var relative_rotation = Math.Quaternion.identity();
        relative_rotation = relative_rotation.multiply(pitch_quat);
        relative_rotation = relative_rotation.multiply(roll_quat);
        relative_rotation = relative_rotation.normalize();

        tf.rotate(relative_rotation);
    }
}

/// Factory that returns a ready‑to‑attach ControllerComponent -------------
pub fn makeBoxController(a: std.mem.Allocator) !Controller.ControllerComponent {
    var c = Controller.ControllerComponent.init(a);

    // mouse
    c.mouse_move = mouseLook;

    // keyboard
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_W, .onPressed = up });
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_S, .onPressed = down });
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_A, .onPressed = yawLeft });
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_D, .onPressed = yawRight });

    return c;
}

/// Collision shape type for the box
pub const BoxCollisionType = enum {
    Primitive, // Simple box shape
    ConvexHull, // Compound shape with convex hull
};

/// Spawn a simple box with collision and drone controls
pub fn spawn(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
    collision_type: BoxCollisionType,
    position: [3]f32,
    size: [3]f32,
    mass: f32,
    scene_width: u32,
    scene_height: u32,
) !Core.EntityID {
    const root_tf = Transform.TransformComponent.init(alloc);
    const root_ctrl = try makeBoxController(alloc);

    const box_mesh = try createBoxMesh(alloc, size);
    const mesh_name = try std.fmt.allocPrint(alloc, "box_mesh_{d}_{d}_{d}", .{
        @as(u32, @intFromFloat(size[0] * 100)),
        @as(u32, @intFromFloat(size[1] * 100)),
        @as(u32, @intFromFloat(size[2] * 100)),
    });
    defer alloc.free(mesh_name);

    // Add mesh to resource manager
    try ecs.world.resource_manager.meshes.put(try alloc.dupe(u8, mesh_name), .{ .mesh = box_mesh, .instance_count = 1 });

    // Create renderer component
    var box_renderer = try Renderer.Renderable.init(alloc, try alloc.dupe(u8, mesh_name));

    // Create a simple material for the box
    const ResourceManager = @import("../ResourceManager.zig");
    const box_material = ResourceManager.MaterialVariant{
        .PBR = ResourceManager.Material(.PBR){
            .data = .{
                .baseColorFactor = .{ 1.0, 0.0, 0.0, 1.0 }, // Red color
            },
        },
    };

    // Create unique material name
    const material_name = try std.fmt.allocPrint(alloc, "box_material_{d}_{d}_{d}", .{
        @as(u32, @intFromFloat(size[0] * 100)),
        @as(u32, @intFromFloat(size[1] * 100)),
        @as(u32, @intFromFloat(size[2] * 100)),
    });
    defer alloc.free(material_name);

    const material_name_owned = try alloc.dupeZ(u8, material_name);

    // Load the material into the resource manager
    try ecs.world.resource_manager.loadMaterial(material_name_owned, box_material, null);
    try box_renderer.setMaterial(alloc, material_name_owned);

    // // Create collider based on type
    const collider = switch (collision_type) {
        .Primitive => blk: {
            const half_extents = [3]f32{ size[0] / 2.0, size[1] / 2.0, size[2] / 2.0 };
            break :blk try Collisions.ColliderComponent.init(
                alloc,
                .{ .Box = .{ .half_extents = half_extents } },
                box_mesh,
            );
        },
        .ConvexHull => blk: {
            // Generate convex hull from box mesh
            const generated_hulls = try ecs.world.resource_manager.getOrGenerateCollisionMesh(box_mesh);
            break :blk try Collisions.ColliderComponent.init(
                alloc,
                .{ .ConvexHull = generated_hulls },
                box_mesh,
            );
        },
    };

    // Create physics body
    var rigid_body = Collisions.RigidBodyComponent.init(mass, collider.bullet_shape.?);
    rigid_body.translate(position);
    std.debug.print("Created box with collision type: {s}, position: [{d:.2}, {d:.2}, {d:.2}], mass: {d:.2}\n", .{
        @tagName(collision_type),
        position[0],
        position[1],
        position[2],
        mass,
    });

    // Create camera using DroneCamera
    const box_cam = try DroneCamera.generate(alloc, .{}, scene_width, scene_height);

    // Create camera frustum for visualization
    const box_cam_frustum = try Frustum.generate(
        alloc,
        ecs,
        "box_cam_frustum",
        box_cam.cam.fov,
        box_cam.cam.aspect,
        1.0, // far
        0.1, // near
    );

    // Spawn entities
    const box_eid = try ecs.spawn(.{ root_tf, root_ctrl, collider, rigid_body, box_renderer });
    const box_cam_eid = try ecs.spawn(box_cam);
    const box_cam_frustum_eid = try ecs.spawn(box_cam_frustum);

    // Set up parent-child relationships: box -> camera -> frustum
    try ecs.transform_system.addChild(box_eid, box_cam_eid);
    try ecs.transform_system.addChild(box_cam_eid, box_cam_frustum_eid);

    return box_eid;
}

// Create a simple box mesh with given dimensions
fn createBoxMesh(alloc: std.mem.Allocator, size: [3]f32) !*Mesh {
    const half_x = size[0] / 2.0;
    const half_y = size[1] / 2.0;
    const half_z = size[2] / 2.0;

    // Define 8 vertices of a box
    const vertices = [_]Mesh.Vertex{
        // Front face
        .{ .position = .{ -half_x, -half_y, half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, 0, 1 }, .texture = .{ 0, 0 } },
        .{ .position = .{ half_x, -half_y, half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, 0, 1 }, .texture = .{ 1, 0 } },
        .{ .position = .{ half_x, half_y, half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, 0, 1 }, .texture = .{ 1, 1 } },
        .{ .position = .{ -half_x, half_y, half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, 0, 1 }, .texture = .{ 0, 1 } },

        // Back face
        .{ .position = .{ -half_x, -half_y, -half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, 0, -1 }, .texture = .{ 1, 0 } },
        .{ .position = .{ -half_x, half_y, -half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, 0, -1 }, .texture = .{ 1, 1 } },
        .{ .position = .{ half_x, half_y, -half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, 0, -1 }, .texture = .{ 0, 1 } },
        .{ .position = .{ half_x, -half_y, -half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, 0, -1 }, .texture = .{ 0, 0 } },

        // Left face
        .{ .position = .{ -half_x, -half_y, -half_z }, .color = .{ 1, 0, 0 }, .normal = .{ -1, 0, 0 }, .texture = .{ 0, 0 } },
        .{ .position = .{ -half_x, -half_y, half_z }, .color = .{ 1, 0, 0 }, .normal = .{ -1, 0, 0 }, .texture = .{ 1, 0 } },
        .{ .position = .{ -half_x, half_y, half_z }, .color = .{ 1, 0, 0 }, .normal = .{ -1, 0, 0 }, .texture = .{ 1, 1 } },
        .{ .position = .{ -half_x, half_y, -half_z }, .color = .{ 1, 0, 0 }, .normal = .{ -1, 0, 0 }, .texture = .{ 0, 1 } },

        // Right face
        .{ .position = .{ half_x, -half_y, -half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 1, 0, 0 }, .texture = .{ 1, 0 } },
        .{ .position = .{ half_x, half_y, -half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 1, 0, 0 }, .texture = .{ 1, 1 } },
        .{ .position = .{ half_x, half_y, half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 1, 0, 0 }, .texture = .{ 0, 1 } },
        .{ .position = .{ half_x, -half_y, half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 1, 0, 0 }, .texture = .{ 0, 0 } },

        // Top face
        .{ .position = .{ -half_x, half_y, -half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, 1, 0 }, .texture = .{ 0, 1 } },
        .{ .position = .{ -half_x, half_y, half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, 1, 0 }, .texture = .{ 0, 0 } },
        .{ .position = .{ half_x, half_y, half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, 1, 0 }, .texture = .{ 1, 0 } },
        .{ .position = .{ half_x, half_y, -half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, 1, 0 }, .texture = .{ 1, 1 } },

        // Bottom face
        .{ .position = .{ -half_x, -half_y, -half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, -1, 0 }, .texture = .{ 0, 0 } },
        .{ .position = .{ half_x, -half_y, -half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, -1, 0 }, .texture = .{ 1, 0 } },
        .{ .position = .{ half_x, -half_y, half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, -1, 0 }, .texture = .{ 1, 1 } },
        .{ .position = .{ -half_x, -half_y, half_z }, .color = .{ 1, 0, 0 }, .normal = .{ 0, -1, 0 }, .texture = .{ 0, 1 } },
    };

    // Define indices for triangles (2 triangles per face, 6 faces)
    const indices = [_]u32{
        // Front face
        0,  1,  2,  2,  3,  0,
        // Back face
        4,  5,  6,  6,  7,  4,
        // Left face
        8,  9,  10, 10, 11, 8,
        // Right face
        12, 13, 14, 14, 15, 12,
        // Top face
        16, 17, 18, 18, 19, 16,
        // Bottom face
        20, 21, 22, 22, 23, 20,
    };

    // Allocate and copy data
    const vertex_data = try alloc.dupe(Mesh.Vertex, &vertices);
    const index_data = try alloc.dupe(u32, &indices);

    return Mesh.init(alloc, vertex_data, index_data, Mesh.gen_draw(.triangles));
}
