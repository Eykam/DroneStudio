const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const gl = @import("../../bindings/gl.zig");
const Opengl = @import("../graphics/OpenGL.zig");
const Camera = @import("../components/Camera.zig");
const Globals = @import("../components/Globals.zig");
const Transform = @import("../components/Transform.zig");
const Viewport = @import("../components/Viewports.zig");
const Renderer = @import("../components/Renderer.zig");
const Frustum = @import("Frustum.zig");
const ECSManager = @import("../ECSManager.zig");

const glfw = gl.glfw;
const Vec3 = Math.Vec3;
const TransformComponent = Transform.TransformComponent;
const ViewportComponent = Viewport.ViewportComponent;

pub fn spawn(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
    desc: struct { pos: [3]f32 = .{ 0, 1, -3 }, fov: f32 = 90, aspect: f32 = 16.0 / 9.0 },
    width: u32,
    height: u32,
) !Core.EntityID {
    var tf = Transform.TransformComponent.init(alloc);
    tf.setPosition(desc.pos[0], desc.pos[1], desc.pos[2]);

    const cam = Camera.CameraComponent{
        .entity_id = undefined,
        .fov = desc.fov,
        .aspect = desc.aspect,
    };

    var vp = try Viewport.ViewportComponent.init(
        alloc,
        "drone_camera",
        width,
        height,
    );
    vp.visibility_mask = Viewport.VisibilityLayer.DEFAULT; // Don't render debug visualizations

    const frustum = Frustum.generate(
        alloc,
        ecs,
        "drone_cam_frustum",
        desc.fov,
        desc.aspect,
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
