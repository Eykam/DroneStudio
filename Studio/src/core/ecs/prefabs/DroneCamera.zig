const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const gl = @import("../../bindings/gl.zig");
const Opengl = @import("../graphics/OpenGL.zig");
const Camera = @import("../components/Camera.zig");
const Globals = @import("../components/Globals.zig");
const Transform = @import("../components/Transform.zig");
const Viewport = @import("../components/Viewports.zig");
const Frustum = @import("Frustum.zig");

const glfw = gl.glfw;
const Vec3 = Math.Vec3;
const TransformComponent = Transform.TransformComponent;
const ViewportComponent = Viewport.ViewportComponent;

pub fn generate(
    alloc: std.mem.Allocator,
    desc: struct { pos: [3]f32 = .{ 0, 1, -3 }, fov: f32 = 90, aspect: f32 = 16.0 / 9.0 },
    width: u32,
    height: u32,
) !struct {
    tf: Transform.TransformComponent,
    cam: Camera.CameraComponent,
    vp: Viewport.ViewportComponent,
} {
    var tf = Transform.TransformComponent.init(alloc);
    tf.setPosition(desc.pos[0], desc.pos[1], desc.pos[2]);

    const cam = Camera.CameraComponent{
        .entity_id = undefined,
        .fov = desc.fov,
        .aspect = desc.aspect,
    };

    const vp = try Viewport.ViewportComponent.init(
        alloc,
        "drone_camera",
        width,
        height,
    );

    return .{ .tf = tf, .cam = cam, .vp = vp };
}
