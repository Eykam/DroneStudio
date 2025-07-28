const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const gl = @import("../../bindings/gl.zig");
const Opengl = @import("../graphics/OpenGL.zig");
const Camera = @import("../components/Camera.zig");
const ECSManager = @import("../ECSManager.zig");
const Controller = @import("../components/Controller.zig");
const Transform = @import("../components/Transform.zig");
const Viewport = @import("../components/Viewports.zig");
const Collisions = @import("../components/Collisions.zig");
const DroneCamera = @import("DroneCamera.zig");
const SensorCamera = @import("SensorCamera.zig");
const Frustum = @import("Frustum.zig");

const glfw = gl.glfw;
const Vec3 = Math.Vec3;

/// ---------------------------------------------------------------------------
/// Tunable constants
const Defaults = struct {
    const thrust: f32 = 20.0; // m/s
    const yawRate: f32 = 2; // °/s
    const rollRate: f32 = 60;
    const pitchRate: f32 = 60; // ° per pixel
};

/// Helpers ---------------------------------------------------------------
const CollisionSystem = Collisions.CollisionSystem;

// Physics-aware movement: use forces if physics is available, otherwise direct transform
inline fn move(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, dir: Vec3, speed: f32, dt: f32) void {
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
inline fn rotate(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, axis: Vec3, deg: f32, dt: f32) void {
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
        // DEBUG: Let's see what axes we're working with
        std.debug.print("Mouse input: dx={d:.2}, dy={d:.2} -> roll={d:.2}, pitch={d:.2}\n", .{ dx, dy, roll_angle, pitch_angle });
        std.debug.print("Local axes - forward: [{d:.2}, {d:.2}, {d:.2}], right: [{d:.2}, {d:.2}, {d:.2}]\n", .{ local_forward.x(), local_forward.y(), local_forward.z(), local_right.x(), local_right.y(), local_right.z() });

        // Use physics torque for dynamic bodies - try much larger values
        const roll_torque_vec = local_forward.scale(-Math.radians(roll_angle) * rigid_body.?.mass);
        const pitch_torque_vec = local_right.scale(-Math.radians(pitch_angle) * rigid_body.?.mass);

        // Combine the torques
        const combined_torque = Vec3.init(roll_torque_vec.x() + pitch_torque_vec.x(), roll_torque_vec.y() + pitch_torque_vec.y(), roll_torque_vec.z() + pitch_torque_vec.z());

        std.debug.print("Applied torque: [{d:.2}, {d:.2}, {d:.2}] (mass: {d:.2})\n", .{ combined_torque.x(), combined_torque.y(), combined_torque.z(), rigid_body.?.mass });
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
pub fn makeDroneController(a: std.mem.Allocator) !Controller.ControllerComponent {
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

pub fn spawn(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
    scene_width: u32,
    scene_height: u32,
) !Core.EntityID {
    const root_tf = Transform.TransformComponent.init(alloc);
    const root_ctrl = try makeDroneController(alloc);

    const drone_body_resource = try ecs.world.resource_manager.loadGLTFModelCached(
        alloc,
        "assets/drone/scene.gltf",
    );
    defer drone_body_resource.deinit();

    // Create visual model (no physics)
    var entities = try ecs.createEntitiesFromModel(drone_body_resource);
    const drone_body_entity = entities.root_entity;

    // Create convex hull collider from model
    const collider = try Collisions.ColliderComponent.initFromModel(
        alloc,
        drone_body_resource,
        .{ .ConvexHull = &[_]Collisions.ConvexHullShape{} },
        ecs.world.resource_manager,
    );

    // Create physics body with the collider's shape
    var rigid_body = Collisions.RigidBodyComponent.init(1.0, collider.bullet_shape.?);

    // Set initial position offset from collision shape center of mass
    rigid_body.translate(.{ 0, 30, 0 });
    rigid_body.rotate(.{ 0, 0, 0, 1 });

    const drone_cam = try DroneCamera.generate(alloc, .{}, scene_width, scene_height);

    const disparity = 0.075; // 75mm
    const sensor_cam_left = try SensorCamera.generate(alloc, "sensor_cam_left", .{ .pos = .{ -disparity / 2.0, 0.0, 0.15 } });
    const sensor_cam_right = try SensorCamera.generate(alloc, "sensor_cam_right", .{ .pos = .{ disparity / 2.0, 0.0, 0.15 } });

    const drone_cam_frustum = try Frustum.generate(
        alloc,
        ecs,
        "drone_cam_frustum",
        drone_cam.cam.fov,
        drone_cam.cam.aspect,
        1.0,
        0.1,
    );
    const sensor_cam_frustum_left = try Frustum.generate(
        alloc,
        ecs,
        "sensor_cam_frustum",
        sensor_cam_left.cam.fov,
        sensor_cam_left.cam.aspect,
        1.0,
        0.1,
    );

    const sensor_cam_frustum_right = try Frustum.generate(
        alloc,
        ecs,
        "sensor_cam_frustum",
        sensor_cam_right.cam.fov,
        sensor_cam_right.cam.aspect,
        1.0,
        0.1,
    );

    // No need for physics target in new architecture - physics is on the same entity as controller
    const root_eid = try ecs.spawn(.{ root_tf, root_ctrl, collider, rigid_body });

    // Clean up the entity map since we no longer need it
    entities.entity_map.deinit();

    const drone_cam_eid = try ecs.spawn(drone_cam);
    const sensor_cam_left_eid = try ecs.spawn(sensor_cam_left);
    const sensor_cam_right_eid = try ecs.spawn(sensor_cam_right);

    const drone_cam_frustum_eid = try ecs.spawn(drone_cam_frustum);
    const sensor_cam_frustum_left_eid = try ecs.spawn(sensor_cam_frustum_left);
    const sensor_cam_frustum_right_eid = try ecs.spawn(sensor_cam_frustum_right);

    try ecs.transform_system.addChild(root_eid, drone_body_entity);
    try ecs.transform_system.addChild(root_eid, drone_cam_eid);
    try ecs.transform_system.addChild(drone_cam_eid, drone_cam_frustum_eid);

    try ecs.transform_system.addChild(root_eid, sensor_cam_left_eid);
    try ecs.transform_system.addChild(sensor_cam_left_eid, sensor_cam_frustum_left_eid);

    try ecs.transform_system.addChild(root_eid, sensor_cam_right_eid);
    try ecs.transform_system.addChild(sensor_cam_right_eid, sensor_cam_frustum_right_eid);

    return root_eid;
}
