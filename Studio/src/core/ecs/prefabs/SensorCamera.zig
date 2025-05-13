const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const gl = @import("../../bindings/gl.zig");
const Opengl = @import("../graphics/OpenGL.zig");
const Camera = @import("../components/Camera.zig");
const Controller = @import("../components/Controller.zig");
const Transform = @import("../components/Transform.zig");
const Viewport = @import("../components/Viewports.zig");

const glfw = gl.glfw;
const ControllerComponent = Controller.ControllerComponent;
const TransformComponent = Transform.TransformComponent;
const ViewportComponent = Viewport.ViewportComponent;

const Defaults = struct {
    pos: [3]f32 = .{ 0.0, 0.0, 0.15 }, // Position can be adjusted as needed
    focal_length_mm: f32 = 3.04, // Focal length in mm (calculated for 102° FOV)
    sensor_width_mm: f32 = 6.287, // Sensor width in mm (1/2.3" sensor)
    sensor_height_mm: f32 = 4.712, // Sensor height in mm
    resolution_width: u32 = 1280, // Resolution width in pixels
    resolution_height: u32 = 720, // Resolution height in pixels
};

const defaults = Defaults{};

pub fn generate(
    alloc: std.mem.Allocator,
    desc: Defaults,
) !struct {
    tf: Transform.TransformComponent,
    cam: Camera.CameraComponent,
    vp: Viewport.ViewportComponent,
} {
    var tf = Transform.TransformComponent.init(alloc);
    tf.setPosition(desc.pos[0], desc.pos[1], desc.pos[2]);

    const cam = Camera.CameraComponent{
        .entity_id = undefined,
        .fov = 2.0 * Math.degrees(std.math.atan(desc.sensor_height_mm / (2.0 * desc.focal_length_mm))),
        .aspect = @as(f32, @floatFromInt(desc.resolution_width)) / @as(f32, @floatFromInt(desc.resolution_height)),
        .active = true,
    };

    var vp = try Viewport.ViewportComponent.init(
        alloc,
        "sensor_cam",
        defaults.resolution_width,
        defaults.resolution_height,
    );
    vp.enableSharing();

    return .{ .tf = tf, .cam = cam, .vp = vp };
}
