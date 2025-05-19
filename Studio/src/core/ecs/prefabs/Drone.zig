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
const DroneCamera = @import("DroneCamera.zig");
const SensorCamera = @import("SensorCamera.zig");
const Frustum = @import("Frustum.zig");

const glfw = gl.glfw;
const Vec3 = Math.Vec3;

/// ---------------------------------------------------------------------------
/// Tunable constants
const Defaults = struct {
    const thrust: f32 = 4.0; // m/s
    const yawRate: f32 = 120; // °/s
    const rollRate: f32 = 70;
    const pitchRate: f32 = 70; // ° per pixel
};

/// Helpers ---------------------------------------------------------------
inline fn rotate(tf: *Transform.TransformComponent, axis: Vec3, deg: f32, dt: f32) void {
    const q = Math.Quaternion.from_axis_angle(axis, deg * dt);
    tf.rotate(q);
}
inline fn move(tf: *Transform.TransformComponent, dir: Vec3, speed: f32, dt: f32) void {
    tf.translate(dir.scale(speed * dt).data);
}

/// Key‑binding callbacks --------------------------------------------------
fn up(eid: Core.EntityID, tf: *Transform.TransformComponent, dt: f32) void {
    _ = eid;
    move(tf, tf.world_transform.get_up(), Defaults.thrust, dt);
}
fn down(eid: Core.EntityID, tf: *Transform.TransformComponent, dt: f32) void {
    _ = eid;
    move(tf, tf.world_transform.get_up().scale(-1), Defaults.thrust, dt);
}
fn yawLeft(eid: Core.EntityID, tf: *Transform.TransformComponent, dt: f32) void {
    _ = eid;
    rotate(tf, Vec3.init(0, 1, 0), -Defaults.yawRate, dt);
}
fn yawRight(eid: Core.EntityID, tf: *Transform.TransformComponent, dt: f32) void {
    _ = eid;
    rotate(tf, Vec3.init(0, 1, 0), Defaults.yawRate, dt);
}

/// Mouse‑look (pitch / roll) ----------------------------------------------
fn mouseLook(
    eid: Core.EntityID,
    tf: *Transform.TransformComponent,
    dx: f32, // pixels
    dy: f32,
    dt: f64,
) void {
    _ = eid;

    const local_foward = tf.world_transform.get_forward();
    const local_right = tf.world_transform.get_right();

    const roll_angle = -@as(f32, @floatCast(dx)) * Defaults.rollRate * @as(f32, @floatCast(dt));
    const roll_quat = Math.Quaternion.from_axis_angle(local_foward, roll_angle);

    const pitch_angle = @as(f32, @floatCast(-dy)) * Defaults.pitchRate * @as(f32, @floatCast(dt));
    const pitch_quat = Math.Quaternion.from_axis_angle(local_right, pitch_angle);

    var relative_rotation = Math.Quaternion.identity();
    relative_rotation = relative_rotation.multiply(pitch_quat);
    relative_rotation = relative_rotation.multiply(roll_quat);
    relative_rotation = relative_rotation.normalize();

    tf.rotate(relative_rotation);
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
) !void {
    var root_tf = Transform.TransformComponent.init(alloc);
    root_tf.setPosition(0, 2, 0);

    const root_ctrl = try makeDroneController(alloc);

    const drone_body_resource = try ecs.world.resource_manager.loadGLTFModelCached(
        alloc,
        "assets/drone/scene.gltf",
    );

    const drone_body_entity = try ecs.createEntitiesFromModel(drone_body_resource);

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

    const root_eid = try ecs.spawn(.{ root_tf, root_ctrl });

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

    return;
}
