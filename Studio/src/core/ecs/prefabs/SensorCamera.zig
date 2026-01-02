const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const gl = @import("../../bindings/gl.zig");
const Opengl = @import("../graphics/OpenGL.zig");
const Camera = @import("../components/Camera.zig");
const Controller = @import("../components/Controller.zig");
const Transform = @import("../components/Transform.zig");
const Viewport = @import("../components/Viewports.zig");
const Renderer = @import("../components/Renderer.zig");
const Frustum = @import("Frustum.zig");
const ECSManager = @import("../ECSManager.zig");
pub const Sensors = @import("Sensors.zig");

const glfw = gl.glfw;
const ControllerComponent = Controller.ControllerComponent;
const TransformComponent = Transform.TransformComponent;
const ViewportComponent = Viewport.ViewportComponent;

pub const Config = struct {
    pos: [3]f32 = .{ 0.0, 0.0, 0.15 },
    module: Sensors.CameraModule = Sensors.Default,
    resolution_width: u32 = 1280,
    resolution_height: u32 = 720,
};

pub fn spawn(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
    name: []const u8,
    config: Config,
) !Core.EntityID {
    var tf = Transform.TransformComponent.init(alloc);
    tf.setPosition(config.pos[0], config.pos[1], config.pos[2]);

    const vfov = config.module.vfov();
    const aspect = @as(f32, @floatFromInt(config.resolution_width)) / @as(f32, @floatFromInt(config.resolution_height));

    const cam = Camera.CameraComponent{
        .entity_id = undefined,
        .fov = vfov,
        .aspect = aspect,
        .active = true,
    };

    const vp = try Viewport.ViewportComponent.init(
        alloc,
        name,
        config.resolution_width,
        config.resolution_height,
    );
    // vp.enableSharing();

    const frustum = Frustum.generate(
        alloc,
        ecs,
        name,
        vfov,
        aspect,
        1.0,
        0.1,
    );

    // Spawn camera entity
    const cam_eid = try ecs.spawn(.{ tf, cam, vp });

    // Spawn frustum as child
    const frustum_eid = try ecs.spawn(.{ frustum.tf, frustum.renderable });
    try ecs.transform_system.addChild(cam_eid, frustum_eid);

    return cam_eid;
}
