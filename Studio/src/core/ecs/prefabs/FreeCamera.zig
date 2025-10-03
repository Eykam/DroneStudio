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
const ECSManager = @import("../ECSManager.zig");

const glfw = gl.glfw;
const Vec3 = Math.Vec3;
const ControllerComponent = Controller.ControllerComponent;
const TransformComponent = Transform.TransformComponent;
const ViewportComponent = Viewport.ViewportComponent;

// Free camera controller with limited movement (yaw and up/down only)
pub const FreeCameraController = struct {
    const Defaults = struct {
        speed: f32 = 10,
        sensitivity: f32 = 0.25,
    };

    const d = Defaults{};

    pub fn createComponent() Controller.ControllerComponent {
        var controller = Controller.ControllerComponent.init(2, "FreeCamera", .Entity);

        // WASD - Directional movement
        controller.addBinding(.{
            .key = .W,
            .handler = handleMoveForward,
            .context = null,
            .binding_type = .Continuous,
        }) catch unreachable;

        controller.addBinding(.{
            .key = .S,
            .handler = handleMoveBackward,
            .context = null,
            .binding_type = .Continuous,
        }) catch unreachable;

        controller.addBinding(.{
            .key = .A,
            .handler = handleMoveLeft,
            .context = null,
            .binding_type = .Continuous,
        }) catch unreachable;

        controller.addBinding(.{
            .key = .D,
            .handler = handleMoveRight,
            .context = null,
            .binding_type = .Continuous,
        }) catch unreachable;

        // Space/Shift - Vertical movement
        controller.addBinding(.{
            .key = .Space,
            .binding_type = .Continuous,
            .handler = handleMoveUp,
            .context = null,
        }) catch unreachable;

        controller.addBinding(.{
            .key = .LeftShift,
            .handler = handleMoveDown,
            .context = null,
            .binding_type = .Continuous,
        }) catch unreachable;

        // Mouse handler for yaw & pitch
        controller.setMouseHandler(handleMouseLook, null);

        return controller;
    }

    inline fn move(
        tf: *TransformComponent,
        dir: Vec3,
        speed: f32,
        dt: f32,
    ) void {
        // Use a minimum dt to ensure consistent movement speed
        const effective_dt = @min(dt, 0.016); // At least 60fps equivalent
        tf.translate(dir.scale(speed * effective_dt).data);
    }

    fn handleMoveForward(event: *Controller.InputEvent, context: ?*anyopaque) void {
        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        if (event.action == .Release) return;

        if (ecs.transform_components.get(selected_entity)) |transform| {
            const dir = transform.world_transform.get_forward();
            move(transform, dir, d.speed, event.dt);
        }
        event.consume();
    }

    fn handleMoveBackward(event: *Controller.InputEvent, context: ?*anyopaque) void {
        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        if (event.action == .Release) return;

        if (ecs.transform_components.get(selected_entity)) |transform| {
            const dir = transform.world_transform.get_forward().scale(-1);
            move(transform, dir, d.speed, event.dt);
        }
        event.consume();
    }

    fn handleMoveLeft(event: *Controller.InputEvent, context: ?*anyopaque) void {
        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        if (event.action == .Release) return;

        if (ecs.transform_components.get(selected_entity)) |transform| {
            const dir = transform.world_transform.get_right();
            move(transform, dir, d.speed, event.dt);
        }
        event.consume();
    }

    fn handleMoveRight(event: *Controller.InputEvent, context: ?*anyopaque) void {
        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        if (event.action == .Release) return;

        if (ecs.transform_components.get(selected_entity)) |transform| {
            const dir = transform.world_transform.get_right().scale(-1);
            move(transform, dir, d.speed, event.dt);
        }
        event.consume();
    }

    fn handleMoveUp(event: *Controller.InputEvent, context: ?*anyopaque) void {
        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        if (event.action == .Release) return;

        if (ecs.transform_components.get(selected_entity)) |transform| {
            move(transform, Vec3.init(0, 1, 0), d.speed, event.dt);
        }
        event.consume();
    }

    fn handleMoveDown(event: *Controller.InputEvent, context: ?*anyopaque) void {
        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        if (event.action == .Release) return;

        if (ecs.transform_components.get(selected_entity)) |transform| {
            move(transform, Vec3.init(0, -1, 0), d.speed, event.dt);
        }
        event.consume();
    }

    fn handleMouseLook(event: *Controller.MouseEvent, context: ?*anyopaque) void {
        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const selected_entity = ecs.control_system.selected_entity orelse return;

        // Only process mouse movement (not clicks)
        if (event.button != null) return;

        if (ecs.transform_components.get(selected_entity)) |transform| {
            // Only apply yaw rotation (no ptransformationitch to restrict movement)
            const yaw_delta = @as(f32, @floatCast(event.dx)) * d.sensitivity;
            const yaw_quat = Math.Quaternion.from_axis_angle(Vec3.init(0, 1, 0), yaw_delta);

            const pitch_delta = @as(f32, @floatCast(event.dy)) * d.sensitivity;
            const pitch_quat = Math.Quaternion.from_axis_angle(transform.world_transform.get_right(), pitch_delta).scale(-1);

            const rotation_quat = yaw_quat.multiply(pitch_quat);
            transform.rotate(rotation_quat);
        }
        event.consume();
    }
};

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

    const ctrl = FreeCameraController.createComponent();
    return .{ .tf = tf, .cam = cam, .ctrl = ctrl, .vp = vp };
}

pub fn spawn(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
    width: u32,
    height: u32,
) Core.EntityID {
    const free_cam = generate(alloc, .{}, width, height);
    return ecs.spawn(free_cam) catch |err| {
        std.debug.print("Error while spawning free camera => {any}\n", .{err});
        @panic("Failed to generate free camera...");
    };
}
