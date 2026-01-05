const std = @import("std");
const Mesh = @import("../../Mesh.zig");
const ECSManager = @import("../ECSManager.zig");
const Core = @import("../Core.zig");
const Renderer = @import("../components/Renderer.zig");
const Transform = @import("../components/Transform.zig");

const Renderable = Renderer.Renderable;
const TransformComponent = Transform.TransformComponent;

/// Create an instanced point cloud mesh for SLAM landmarks
pub fn generate(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
    name: [:0]const u8,
    max_instances: u32,
) struct { renderable: Renderable, tf: TransformComponent } {
    // Create a single point vertex at origin - instances provide actual positions
    var vertices = alloc.alloc(Mesh.Vertex, 1) catch |err| {
        std.debug.print("Failed to allocate landmark vertex => {}\n", .{err});
        @panic("Failed to allocate landmark vertex");
    };
    defer alloc.free(vertices);

    vertices[0] = .{
        .position = .{ 0.0, 0.0, 0.0 },
        .color = .{ 1.0, 1.0, 1.0 }, // Default white, instance colors override
    };

    // Load mesh into resource manager
    _ = ecs.world.resource_manager.loadMesh(name, vertices, null, Mesh.instanced_point_draw) catch |err| {
        std.debug.print("Failed to load landmark mesh => {}\n", .{err});
        @panic("Failed to load landmark mesh");
    };

    // Initialize instance buffers for the mesh
    if (ecs.world.resource_manager.meshes.getPtr(name)) |mesh_resource| {
        mesh_resource.initInstancedPoints(max_instances);
    }

    var renderable = Renderable.init(alloc, name) catch |err| {
        std.debug.print("Failed to initialize landmark renderable => {}\n", .{err});
        @panic("Failed to initialize landmark renderable");
    };
    renderable.visibility_mask = Renderer.VisibilityLayer.DEBUG;

    const tf = TransformComponent.init(alloc);

    return .{ .renderable = renderable, .tf = tf };
}

/// Spawn a landmark cloud entity and return its ID
pub fn spawn(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
    name: [:0]const u8,
    max_instances: u32,
) !Core.EntityID {
    const components = generate(alloc, ecs, name, max_instances);
    return try ecs.spawn(.{ components.tf, components.renderable });
}

/// Update landmark positions and colors
pub fn updateLandmarks(
    ecs: *ECSManager,
    mesh_name: [:0]const u8,
    positions: []const [4]f32, // xyz + point size in w
    colors: []const [4]f32, // rgba
) void {
    if (ecs.world.resource_manager.meshes.getPtr(mesh_name)) |mesh_resource| {
        mesh_resource.updateInstancePositions(positions);
        mesh_resource.updateInstanceColors(colors);
    }
}
