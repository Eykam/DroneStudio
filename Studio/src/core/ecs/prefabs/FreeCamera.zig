const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const gl = @import("../../bindings/gl.zig");
const Opengl = @import("../graphics/OpenGL.zig");
const Camera = @import("../components/Camera.zig");
const Globals = @import("../components/Globals.zig");
const Controller = @import("../components/Controller.zig");
const Transform = @import("../components/Transform.zig");
const Viewport = @import("../components/Viewports.zig");
const Physics = @import("../components/Physics.zig");
const Collisions = @import("../components/Collisions.zig");

const glfw = gl.glfw;
const Vec3 = Math.Vec3;
const ControllerComponent = Controller.ControllerComponent;
const TransformComponent = Transform.TransformComponent;
const ViewportComponent = Viewport.ViewportComponent;

const Defaults = struct {
    speed: f32 = 7.5,
    sensitivity: f32 = 0.25,
};

inline fn move(
    tf: *TransformComponent,
    dir: Vec3,
    speed: f32,
    dt: f32,
) void {
    tf.translate(dir.scale(speed * dt).data);
}

const d = Defaults{};

/// FPS‑style movement callbacks
/// Signature must match Controller.KeyBinding.onPressed
fn forward(_: Core.EntityID, tf: *TransformComponent, _: ?*Collisions.RigidBodyComponent, _: ?*Collisions.CollisionSystem, dt: f32) void {
    const dir = tf.world_transform.get_forward();
    move(tf, dir, d.speed, dt);
}
fn back(_: Core.EntityID, tf: *TransformComponent, _: ?*Collisions.RigidBodyComponent, _: ?*Collisions.CollisionSystem, dt: f32) void {
    const dir = tf.world_transform.get_forward().scale(-1);
    move(tf, dir, d.speed, dt);
}
fn left(_: Core.EntityID, tf: *TransformComponent, _: ?*Collisions.RigidBodyComponent, _: ?*Collisions.CollisionSystem, dt: f32) void {
    const dir = tf.world_transform.get_right();
    move(tf, dir, d.speed, dt);
}
fn right(_: Core.EntityID, tf: *TransformComponent, _: ?*Collisions.RigidBodyComponent, _: ?*Collisions.CollisionSystem, dt: f32) void {
    const dir = tf.world_transform.get_right().scale(-1);
    move(tf, dir, d.speed, dt);
}
fn up(_: Core.EntityID, tf: *TransformComponent, _: ?*Collisions.RigidBodyComponent, _: ?*Collisions.CollisionSystem, dt: f32) void {
    move(tf, Vec3.init(0, 1, 0), d.speed, dt);
}
fn down(_: Core.EntityID, tf: *TransformComponent, _: ?*Collisions.RigidBodyComponent, _: ?*Collisions.CollisionSystem, dt: f32) void {
    move(tf, Vec3.init(0, -1, 0), d.speed, dt);
}

/// Called every frame to apply mouse yaw/pitch
fn mouseLook(
    _: Core.EntityID,
    tf: *TransformComponent,
    _: ?*Collisions.RigidBodyComponent,
    _: ?*Collisions.CollisionSystem,
    yawDelta: f32,
    pitchDelta: f32,
    _: f64,
) void {
    const sy = @as(f32, @floatCast(yawDelta)) * d.sensitivity;
    const sp = @as(f32, @floatCast(pitchDelta)) * d.sensitivity;

    const yawQuat = Math.Quaternion.from_axis_angle(Vec3.init(0, 1, 0), sy);
    const pitchQuat = Math.Quaternion.from_axis_angle(tf.world_transform.get_right(), sp).scale(-1);

    tf.rotate(yawQuat.multiply(pitchQuat));
}

fn makeFreeController(alloc: std.mem.Allocator) !ControllerComponent {
    var c = ControllerComponent.init(alloc);
    c.mouse_move = mouseLook;
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_W, .onPressed = forward });
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_S, .onPressed = back });
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_A, .onPressed = left });
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_D, .onPressed = right });
    return c;
}

pub fn generate(
    alloc: std.mem.Allocator,
    desc: struct { pos: [3]f32 = .{ 0, 1, 5 }, fov: f32 = 90, aspect: f32 = 16.0 / 9.0 },
    width: u32,
    height: u32,
) !struct {
    tf: Transform.TransformComponent,
    cam: Camera.CameraComponent,
    ctrl: Controller.ControllerComponent,
    vp: Viewport.ViewportComponent,
} {
    var tf = Transform.TransformComponent.init(alloc);
    tf.setPosition(desc.pos[0], desc.pos[1], desc.pos[2]);

    const cam = Camera.CameraComponent{
        .entity_id = undefined,
        .fov = desc.fov,
        .aspect = desc.aspect,
        .active = true,
    };

    var vp = try Viewport.ViewportComponent.init(
        alloc,
        "free_cam",
        width,
        height,
    );
    vp.resizable = true;

    const ctrl = try makeFreeController(alloc);
    return .{ .tf = tf, .cam = cam, .ctrl = ctrl, .vp = vp };
}
