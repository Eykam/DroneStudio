const std = @import("std");
const gl = @import("../../bindings/gl.zig");
const Math = @import("../../Math.zig");
const Mesh = @import("../../Mesh.zig");
const ECSManager = @import("../ECSManager.zig");
const Renderer = @import("../components/Renderer.zig");
const Transform = @import("../components/Transform.zig");

const glad = gl.glad;
const Vec3 = Math.Vec3;
const Renderable = Renderer.Renderable;
const TransformComponent = Transform.TransformComponent;

const Self = @This();

fov: f32,
aspect_ratio: f32,
frustum_debug_near: f32,
frustum_debug_far: f32,

fn create_frustum_visualization(self: Self, allocator: std.mem.Allocator) !struct { vertices: []Mesh.Vertex, draw_fn: Mesh.draw } {
    // Create a mesh for the frustum visualization
    const vertices = try self.generate_frustum_vertices(allocator);

    var frustum_mesh = try Mesh.init(allocator, vertices, null, Mesh.gen_draw(glad.GL_LINES));
    frustum_mesh.drawType = glad.GL_LINES;

    // Set frustum color
    for (frustum_mesh.vertices) |*vertex| {
        vertex.color = .{ 0.0, 1.0, 1.0 }; // Cyan color for camera frustum
    }

    return .{ .vertices = vertices, .draw_fn = Mesh.gen_draw(glad.GL_LINES) };
}

fn generate_frustum_vertices(self: Self, allocator: std.mem.Allocator) ![]Mesh.Vertex {
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

pub fn generate(alloc: std.mem.Allocator, ecs: *ECSManager, name: []const u8, fov: f32, aspect_ratio: f32, far: f32, near: f32) !struct { renderable: Renderable, tf: TransformComponent } {
    const self = Self{ .fov = fov, .aspect_ratio = aspect_ratio, .frustum_debug_far = far, .frustum_debug_near = near };
    const resources = try self.create_frustum_visualization(alloc);
    defer alloc.free(resources.vertices);

    _ = try ecs.world.resource_manager.loadMesh(name, resources.vertices, null, resources.draw_fn);
    const renderable = try Renderable.init(alloc, name);

    var tf = Transform.TransformComponent.init(alloc);
    tf.rotateWithEuler(0, 180, 0);

    return .{ .renderable = renderable, .tf = tf };
}
