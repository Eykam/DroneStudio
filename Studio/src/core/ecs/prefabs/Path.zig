const std = @import("std");
const gl = @import("../../bindings/gl.zig");
const Math = @import("../../Math.zig");
const Mesh = @import("../../Mesh.zig");
const ECSManager = @import("../ECSManager.zig");
const Renderer = @import("../components/Renderer.zig");
const Transform = @import("../components/Transform.zig");
const Frustum = @import("./Frustum.zig");
const Core = @import("../Core.zig");
const PathSystem = @import("../components/PathSystem.zig");

const glad = gl.glad;
const Vec3 = Math.Vec3;
const Quaternion = Math.Quaternion;
const Renderable = Renderer.Renderable;
const TransformComponent = Transform.TransformComponent;
const Waypoint = PathSystem.Waypoint;
const CubicBezier3 = PathSystem.CubicBezier3;

pub const PathEntities = struct {
    path_line: Core.EntityID,
    waypoint_points: Core.EntityID,
    frustums: Core.EntityID,

    pub fn setVisible(self: *PathEntities, ecs: *ECSManager, visible: bool) void {
        ecs.render_system.setVisibility(self.path_line, visible);
        ecs.render_system.setVisibility(self.waypoint_points, visible);
        ecs.render_system.setVisibility(self.frustums, visible);
    }

    pub fn deinit(self: *PathEntities) void {
        _ = self;
    }
};

pub fn spawn(
    allocator: std.mem.Allocator,
    ecs: *ECSManager,
    waypoints: []const Waypoint,
    samples: []const Vec3,
    velocities: []const f32,
    path_id: usize,
    waypoint_color: [3]f32,
) !PathEntities {
    // Find min/max velocities for color mapping
    var v_min: f32 = std.math.floatMax(f32);
    var v_max: f32 = -std.math.floatMax(f32);
    for (velocities) |v| {
        v_min = @min(v_min, v);
        v_max = @max(v_max, v);
    }
    const v_range = v_max - v_min;

    // Create path line mesh from samples
    var path_vertices = std.ArrayList(Mesh.Vertex).init(allocator);
    defer path_vertices.deinit();

    for (0..samples.len - 1) |i| {
        const sample = samples[i];
        const next_sample = samples[i + 1];

        const v = if (i < velocities.len) velocities[i] else velocities[velocities.len - 1];
        const t = if (v_range > 1e-6) (v - v_min) / v_range else 0.0;

        const color = [3]f32{
            t,
            1.0 - t,
            0.0,
        };

        try path_vertices.append(.{
            .position = [3]f32{ sample.x(), sample.y(), sample.z() },
            .color = color,
        });
        try path_vertices.append(.{
            .position = [3]f32{ next_sample.x(), next_sample.y(), next_sample.z() },
            .color = color,
        });
    }

    const path_vertices_owned = try allocator.dupe(Mesh.Vertex, path_vertices.items);
    const path_mesh = try Mesh.init(allocator, path_vertices_owned, null, drawPathLines);
    path_mesh.drawType = Mesh.DrawType.lines.toGL();

    const path_mesh_name = try std.fmt.allocPrint(allocator, "path_lines_{d}", .{path_id});
    defer allocator.free(path_mesh_name);
    const path_mesh_name_owned = try allocator.dupe(u8, path_mesh_name);

    const ResourceManager = @import("../ResourceManager.zig");
    const path_mesh_resource = ResourceManager.MeshResource.init(allocator, path_mesh);
    try ecs.world.resource_manager.meshes.put(path_mesh_name_owned, path_mesh_resource);

    const path_transform = TransformComponent.init(allocator);
    var path_renderer = try Renderable.init(allocator, path_mesh_name_owned);
    path_renderer.visibility_mask = Renderer.VisibilityLayer.DEBUG;

    const path_entity = try ecs.spawn(.{
        path_transform,
        path_renderer,
    });

    // Create frustums parent entity
    const frustums_parent_transform = TransformComponent.init(allocator);
    const frustums_parent = try ecs.spawn(.{frustums_parent_transform});

    // Create waypoint points and frustums in a single loop
    var waypoint_vertices = std.ArrayList(Mesh.Vertex).init(allocator);
    defer waypoint_vertices.deinit();

    for (waypoints, 0..) |wp, i| {
        // Add waypoint vertex
        try waypoint_vertices.append(.{
            .position = wp.p.data,
            .color = waypoint_color,
        });

        // Create frustum for this waypoint
        const frustum_name = try std.fmt.allocPrint(allocator, "path_{d}_frustum_{d}", .{ path_id, i });
        defer allocator.free(frustum_name);

        var frustum_resources = Frustum.generate(
            allocator,
            ecs,
            frustum_name,
            60.0, // FOV
            16.0 / 9.0, // aspect ratio
            0.5, // far
            0.01, // near
        );

        var frustum_transform = &frustum_resources.tf;
        frustum_transform.setPosition(wp.p.x(), wp.p.y(), wp.p.z());

        // Apply yaw rotation around Y axis
        const yaw_quat = Quaternion.from_axis_angle(Vec3.init(0, 1, 0), Math.degrees(wp.yaw));
        frustum_transform.setRotation(yaw_quat);

        const frustum_entity = try ecs.spawn(frustum_resources);

        // Add frustum as child of parent
        try ecs.transform_system.addChild(frustums_parent, frustum_entity);
    }

    // Create waypoint mesh
    const waypoint_vertices_owned = try allocator.dupe(Mesh.Vertex, waypoint_vertices.items);
    const waypoint_mesh = try Mesh.init(allocator, waypoint_vertices_owned, null, drawPathPoints);
    waypoint_mesh.drawType = Mesh.DrawType.points.toGL();

    const waypoint_mesh_name = try std.fmt.allocPrint(allocator, "path_waypoints_{d}", .{path_id});
    defer allocator.free(waypoint_mesh_name);
    const waypoint_mesh_name_owned = try allocator.dupe(u8, waypoint_mesh_name);

    const waypoint_mesh_resource = ResourceManager.MeshResource.init(allocator, waypoint_mesh);
    try ecs.world.resource_manager.meshes.put(waypoint_mesh_name_owned, waypoint_mesh_resource);

    const waypoint_transform = TransformComponent.init(allocator);
    var waypoint_renderer = try Renderable.init(allocator, waypoint_mesh_name_owned);
    waypoint_renderer.visibility_mask = Renderer.VisibilityLayer.DEBUG;

    const waypoint_entity = try ecs.spawn(.{
        waypoint_transform,
        waypoint_renderer,
    });

    return PathEntities{
        .path_line = path_entity,
        .waypoint_points = waypoint_entity,
        .frustums = frustums_parent,
    };
}

fn drawPathLines(mesh: *Mesh) void {
    glad.glLineWidth(3.0);
    glad.glBindVertexArray(mesh.meta.VAO);
    glad.glDrawArrays(mesh.drawType, 0, @intCast(mesh.vertices.len));
    glad.glLineWidth(1.0);
}

fn drawPathPoints(mesh: *Mesh) void {
    glad.glPointSize(8.0);
    glad.glBindVertexArray(mesh.meta.VAO);
    glad.glDrawArrays(mesh.drawType, 0, @intCast(mesh.vertices.len));
    glad.glPointSize(1.0);
}
