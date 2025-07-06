const std = @import("std");
const gl = @import("../../bindings/gl.zig");
const c = @import("../../bindings/c.zig");
const Math = @import("../../Math.zig");
const Mesh = @import("../../Mesh.zig");
const ECSManager = @import("../ECSManager.zig");
const Renderer = @import("../components/Renderer.zig");
const Transform = @import("../components/Transform.zig");
const Collisions = @import("../components/Collisions.zig");

const glad = gl.glad;
const bullet = c.bullet;
const Vec3 = Math.Vec3;
const Renderable = Renderer.Renderable;
const TransformComponent = Transform.TransformComponent;

// Structure to collect line data from Bullet's debug drawing
const LineCollector = struct {
    vertices: std.ArrayList(Mesh.Vertex),
    allocator: std.mem.Allocator,
    color: [3]f32,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, color: [3]f32) Self {
        return .{
            .vertices = std.ArrayList(Mesh.Vertex).init(allocator),
            .allocator = allocator,
            .color = color,
        };
    }

    pub fn deinit(self: *Self) void {
        self.vertices.deinit();
    }

    pub fn toOwnedSlice(self: *Self) ![]Mesh.Vertex {
        return try self.vertices.toOwnedSlice();
    }
};

// C callback function that Bullet will call
export fn debugDrawLineCallback(context: ?*anyopaque, p0: [*c]const f32, p1: [*c]const f32, color: [*c]const f32) void {
    _ = color; // Ignore Bullet's color, use our own
    if (context) |ctx| {
        const collector: *LineCollector = @ptrCast(@alignCast(ctx));

        // Add two vertices for the line
        collector.vertices.append(.{
            .position = .{ p0[0], p0[1], p0[2] },
            .color = collector.color,
        }) catch return;

        collector.vertices.append(.{
            .position = .{ p1[0], p1[1], p1[2] },
            .color = collector.color,
        }) catch return;
    }
}

// Function to extract wireframe from any Bullet shape using debug drawing
fn extractShapeWireframe(alloc: std.mem.Allocator, bullet_shape: bullet.CbtShapeHandle, color: [3]f32) ![]Mesh.Vertex {
    // Create a temporary world just for this shape
    const temp_world = bullet.cbtWorldCreate();
    defer bullet.cbtWorldDestroy(temp_world);

    // Create a temporary body just for debug drawing
    const temp_body = bullet.cbtBodyAllocate();
    defer {
        // Must remove from world, then destroy, then deallocate
        bullet.cbtWorldRemoveBody(temp_world, temp_body);
        bullet.cbtBodyDestroy(temp_body);
        bullet.cbtBodyDeallocate(temp_body);
    }

    // Create body with identity transform
    var identity_transform = [4][3]f32{
        [3]f32{ 1.0, 0.0, 0.0 },
        [3]f32{ 0.0, 1.0, 0.0 },
        [3]f32{ 0.0, 0.0, 1.0 },
        [3]f32{ 0.0, 0.0, 0.0 },
    };

    // Create body with the original shape and identity transform
    // The debug drawing will handle compound shapes correctly
    bullet.cbtBodyCreate(temp_body, 0.0, &identity_transform, bullet_shape);

    // Add to temporary world (so only this body gets drawn)
    bullet.cbtWorldAddBody(temp_world, temp_body);

    // Set up line collector
    var collector = LineCollector.init(alloc, color);
    defer collector.deinit();

    // Set up debug drawer
    var debug_draw = bullet.CbtDebugDraw{
        .drawLine1 = debugDrawLineCallback,
        .drawLine2 = null, // Optional
        .drawContactPoint = null, // Optional
        .context = &collector,
    };

    // Configure Bullet to draw wireframes on temporary world
    bullet.cbtWorldDebugSetDrawer(temp_world, &debug_draw);
    bullet.cbtWorldDebugSetMode(temp_world, bullet.CBT_DBGMODE_DRAW_WIREFRAME);

    // Trigger the debug drawing - this will only draw our temp body
    bullet.cbtWorldDebugDrawAll(temp_world);

    return try collector.toOwnedSlice();
}

// Main entry point for generating debug visualization based on collider type
pub fn generateDebugVisualization(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
    name: []const u8,
    collider_shape: Collisions.ColliderShape,
    collider: *Collisions.ColliderComponent,
    is_dynamic: bool,
    mesh_data: ?*Mesh,
) !struct { renderable: Renderable, tf: TransformComponent } {
    const color = if (is_dynamic) [3]f32{ 0.0, 1.0, 0.0 } else [3]f32{ 0.0, 0.0, 1.0 };

    _ = mesh_data;
    _ = collider_shape;

    const vertices = if (collider.bullet_shape) |bullet_shape|
        try extractShapeWireframe(alloc, bullet_shape, color)
    else
        return error.ColliderMissingBulletShape;
    defer alloc.free(vertices);

    _ = try ecs.world.resource_manager.loadMesh(name, vertices, null, Mesh.gen_draw(glad.GL_LINES));
    const renderable = try Renderable.init(alloc, name);

    const tf = Transform.TransformComponent.init(alloc);

    return .{ .renderable = renderable, .tf = tf };
}
