const std = @import("std");
const gl = @import("../../bindings/gl.zig");
const c = @import("../../bindings/c.zig");
const Math = @import("../../Math.zig");
const Mesh = @import("../../Mesh.zig");
const ECSManager = @import("../ECSManager.zig");
const Core = @import("../Core.zig");
const Renderer = @import("../components/Renderer.zig");
const Transform = @import("../components/Transform.zig");
const Collisions = @import("../components/Collisions.zig");

const glad = gl.glad;
const bullet = c.bullet;
const Vec3 = Math.Vec3;
const Renderable = Renderer.Renderable;
const TransformComponent = Transform.TransformComponent;

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

// This function is no longer needed - wireframes are extracted on the physics thread

pub fn generateDebugVisualization(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
    name: []const u8,
    collider_eid: Core.EntityID,
    is_dynamic: bool,
) !Core.EntityID {
    _ = ecs.collider_components.get(collider_eid) orelse return error.NoColliderFound;
    
    // Get the rigid body to access the bullet body handle
    const rigid_body = ecs.rigid_body_components.get(collider_eid) orelse return error.NoRigidBodyFound;
    
    // Get physics world from collision system
    const physics_world = if (ecs.collision_system.physics_thread) |physics_thread| 
        physics_thread.bullet_world 
    else 
        return error.NoPhysicsWorld;
    
    _ = rigid_body.bullet_body orelse return error.NoPhysicsBody;

    const color = if (is_dynamic) [3]f32{ 0.0, 1.0, 0.0 } else [3]f32{ 0.0, 0.0, 1.0 };
    const vertices = try extractShapeWireframe(alloc, physics_world, color);
    defer alloc.free(vertices);

    _ = try ecs.world.resource_manager.loadMesh(name, vertices, null, Mesh.gen_draw(glad.GL_LINES));

    const renderable = try Renderable.init(alloc, name);
    const tf = TransformComponent.init(alloc);

    const debug_eid = try ecs.spawn(.{ renderable, tf });

    try ecs.setParent(debug_eid, collider_eid);

    return debug_eid;
}
