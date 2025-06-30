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

// Main entry point for generating debug visualization based on collider type
pub fn generateDebugVisualization(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
    name: []const u8,
    collider_shape: Collisions.ColliderShape,
    _: *Collisions.ColliderComponent,
    is_dynamic: bool,
    mesh_data: ?*Mesh,
) !struct { renderable: Renderable, tf: TransformComponent } {
    const color = if (is_dynamic) [3]f32{ 0.0, 1.0, 0.0 } else [3]f32{ 0.0, 0.0, 1.0 };

    const vertices = switch (collider_shape) {
        .Box => |data| try createBoxWireframe(alloc, data.half_extents, color),
        .Sphere => |data| try createSphereWireframe(alloc, data.radius, color),
        .Capsule => |data| try createCapsuleWireframe(alloc, data.radius, data.height, color),
        .Cylinder => |data| try createCylinderWireframe(alloc, data.half_extents, color),
        .Cone => |data| try createConeWireframe(alloc, data.radius, data.height, color),
        .TriangleMesh => blk: {
            if (mesh_data) |mesh| {
                break :blk try createTriangleMeshWireframe(alloc, mesh, color);
            } else {
                return error.TriangleMeshRequiresMeshData;
            }
        },
        .CompoundShape => blk: {
            // For compound shapes, show a simple bounding box for now
            const half_extents = [3]f32{ 1.0, 1.0, 1.0 }; // Fallback size
            break :blk try createBoxWireframe(alloc, half_extents, color);
        },
        .ConvexHull => |hulls| blk: {
            // For convex hulls, create wireframes for all hulls
            var all_vertices = std.ArrayList(Mesh.Vertex).init(alloc);
            defer all_vertices.deinit();

            for (hulls) |hull| {
                // Create wireframe from convex hull triangles
                const hull_vertices = try createConvexHullWireframe(alloc, hull, color);
                defer alloc.free(hull_vertices);
                try all_vertices.appendSlice(hull_vertices);
            }

            break :blk try all_vertices.toOwnedSlice();
        },
    };
    defer alloc.free(vertices);

    _ = try ecs.world.resource_manager.loadMesh(name, vertices, null, Mesh.gen_draw(glad.GL_LINES));
    const renderable = try Renderable.init(alloc, name);

    const tf = Transform.TransformComponent.init(alloc);

    return .{ .renderable = renderable, .tf = tf };
}

// Create box wireframe
fn createBoxWireframe(alloc: std.mem.Allocator, half_extents: [3]f32, color: [3]f32) ![]Mesh.Vertex {
    const hx = half_extents[0];
    const hy = half_extents[1];
    const hz = half_extents[2];

    // Define 8 corners of the box
    const corners = [8][3]f32{
        .{ -hx, -hy, -hz }, // 0: bottom-left-back
        .{ hx, -hy, -hz }, // 1: bottom-right-back
        .{ hx, hy, -hz }, // 2: top-right-back
        .{ -hx, hy, -hz }, // 3: top-left-back
        .{ -hx, -hy, hz }, // 4: bottom-left-front
        .{ hx, -hy, hz }, // 5: bottom-right-front
        .{ hx, hy, hz }, // 6: top-right-front
        .{ -hx, hy, hz }, // 7: top-left-front
    };

    // Create lines for wireframe (12 edges of a cube = 24 vertices)
    var vertices = try alloc.alloc(Mesh.Vertex, 24);

    // Bottom face edges
    vertices[0] = .{ .position = corners[0], .color = color }; // 0 -> 1
    vertices[1] = .{ .position = corners[1], .color = color };
    vertices[2] = .{ .position = corners[1], .color = color }; // 1 -> 5
    vertices[3] = .{ .position = corners[5], .color = color };
    vertices[4] = .{ .position = corners[5], .color = color }; // 5 -> 4
    vertices[5] = .{ .position = corners[4], .color = color };
    vertices[6] = .{ .position = corners[4], .color = color }; // 4 -> 0
    vertices[7] = .{ .position = corners[0], .color = color };

    // Top face edges
    vertices[8] = .{ .position = corners[3], .color = color }; // 3 -> 2
    vertices[9] = .{ .position = corners[2], .color = color };
    vertices[10] = .{ .position = corners[2], .color = color }; // 2 -> 6
    vertices[11] = .{ .position = corners[6], .color = color };
    vertices[12] = .{ .position = corners[6], .color = color }; // 6 -> 7
    vertices[13] = .{ .position = corners[7], .color = color };
    vertices[14] = .{ .position = corners[7], .color = color }; // 7 -> 3
    vertices[15] = .{ .position = corners[3], .color = color };

    // Vertical edges
    vertices[16] = .{ .position = corners[0], .color = color }; // 0 -> 3
    vertices[17] = .{ .position = corners[3], .color = color };
    vertices[18] = .{ .position = corners[1], .color = color }; // 1 -> 2
    vertices[19] = .{ .position = corners[2], .color = color };
    vertices[20] = .{ .position = corners[5], .color = color }; // 5 -> 6
    vertices[21] = .{ .position = corners[6], .color = color };
    vertices[22] = .{ .position = corners[4], .color = color }; // 4 -> 7
    vertices[23] = .{ .position = corners[7], .color = color };

    return vertices;
}

// Create sphere wireframe
fn createSphereWireframe(alloc: std.mem.Allocator, radius: f32, color: [3]f32) ![]Mesh.Vertex {
    const segments = 16;
    const rings = 8;
    var vertices = std.ArrayList(Mesh.Vertex).init(alloc);
    defer vertices.deinit();

    // Create horizontal rings
    for (0..rings) |ring| {
        const theta = @as(f32, @floatFromInt(ring)) * std.math.pi / @as(f32, @floatFromInt(rings - 1));
        const sin_theta = @sin(theta);
        const cos_theta = @cos(theta);
        const ring_radius = radius * sin_theta;
        const y = radius * cos_theta;

        for (0..segments) |i| {
            const phi1 = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / @as(f32, @floatFromInt(segments));
            const phi2 = @as(f32, @floatFromInt((i + 1) % segments)) * 2.0 * std.math.pi / @as(f32, @floatFromInt(segments));

            const x1 = ring_radius * @cos(phi1);
            const z1 = ring_radius * @sin(phi1);
            const x2 = ring_radius * @cos(phi2);
            const z2 = ring_radius * @sin(phi2);

            try vertices.append(.{ .position = .{ x1, y, z1 }, .color = color });
            try vertices.append(.{ .position = .{ x2, y, z2 }, .color = color });
        }
    }

    // Create vertical meridians
    for (0..segments / 2) |i| {
        const phi = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / @as(f32, @floatFromInt(segments));
        const cos_phi = @cos(phi);
        const sin_phi = @sin(phi);

        for (0..rings - 1) |ring| {
            const theta1 = @as(f32, @floatFromInt(ring)) * std.math.pi / @as(f32, @floatFromInt(rings - 1));
            const theta2 = @as(f32, @floatFromInt(ring + 1)) * std.math.pi / @as(f32, @floatFromInt(rings - 1));

            const y1 = radius * @cos(theta1);
            const r1 = radius * @sin(theta1);
            const y2 = radius * @cos(theta2);
            const r2 = radius * @sin(theta2);

            try vertices.append(.{ .position = .{ r1 * cos_phi, y1, r1 * sin_phi }, .color = color });
            try vertices.append(.{ .position = .{ r2 * cos_phi, y2, r2 * sin_phi }, .color = color });
        }
    }

    return try vertices.toOwnedSlice();
}

// Create capsule wireframe
fn createCapsuleWireframe(alloc: std.mem.Allocator, radius: f32, height: f32, color: [3]f32) ![]Mesh.Vertex {
    var vertices = std.ArrayList(Mesh.Vertex).init(alloc);
    defer vertices.deinit();

    const half_height = height / 2.0;
    const segments = 16;

    // Cylindrical section
    for (0..segments) |i| {
        const angle1 = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / @as(f32, @floatFromInt(segments));
        const angle2 = @as(f32, @floatFromInt((i + 1) % segments)) * 2.0 * std.math.pi / @as(f32, @floatFromInt(segments));

        const x1 = radius * @cos(angle1);
        const z1 = radius * @sin(angle1);
        const x2 = radius * @cos(angle2);
        const z2 = radius * @sin(angle2);

        // Bottom circle
        try vertices.append(.{ .position = .{ x1, -half_height, z1 }, .color = color });
        try vertices.append(.{ .position = .{ x2, -half_height, z2 }, .color = color });

        // Top circle
        try vertices.append(.{ .position = .{ x1, half_height, z1 }, .color = color });
        try vertices.append(.{ .position = .{ x2, half_height, z2 }, .color = color });

        // Vertical lines
        try vertices.append(.{ .position = .{ x1, -half_height, z1 }, .color = color });
        try vertices.append(.{ .position = .{ x1, half_height, z1 }, .color = color });
    }

    // Hemisphere caps (simplified)
    for (0..segments / 2) |i| {
        const angle = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / @as(f32, @floatFromInt(segments));
        const x = radius * @cos(angle);
        const z = radius * @sin(angle);

        // Top hemisphere
        try vertices.append(.{ .position = .{ x, half_height, z }, .color = color });
        try vertices.append(.{ .position = .{ 0, half_height + radius, 0 }, .color = color });

        // Bottom hemisphere
        try vertices.append(.{ .position = .{ x, -half_height, z }, .color = color });
        try vertices.append(.{ .position = .{ 0, -half_height - radius, 0 }, .color = color });
    }

    return try vertices.toOwnedSlice();
}

// Create cylinder wireframe
fn createCylinderWireframe(alloc: std.mem.Allocator, half_extents: [3]f32, color: [3]f32) ![]Mesh.Vertex {
    var vertices = std.ArrayList(Mesh.Vertex).init(alloc);
    defer vertices.deinit();

    const radius = @max(half_extents[0], half_extents[2]);
    const half_height = half_extents[1];
    const segments = 16;

    // Top and bottom circles
    for (0..segments) |i| {
        const angle1 = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / @as(f32, @floatFromInt(segments));
        const angle2 = @as(f32, @floatFromInt((i + 1) % segments)) * 2.0 * std.math.pi / @as(f32, @floatFromInt(segments));

        const x1 = radius * @cos(angle1);
        const z1 = radius * @sin(angle1);
        const x2 = radius * @cos(angle2);
        const z2 = radius * @sin(angle2);

        // Bottom circle
        try vertices.append(.{ .position = .{ x1, -half_height, z1 }, .color = color });
        try vertices.append(.{ .position = .{ x2, -half_height, z2 }, .color = color });

        // Top circle
        try vertices.append(.{ .position = .{ x1, half_height, z1 }, .color = color });
        try vertices.append(.{ .position = .{ x2, half_height, z2 }, .color = color });

        // Vertical lines (only every 4th segment to avoid clutter)
        if (i % 4 == 0) {
            try vertices.append(.{ .position = .{ x1, -half_height, z1 }, .color = color });
            try vertices.append(.{ .position = .{ x1, half_height, z1 }, .color = color });
        }
    }

    return try vertices.toOwnedSlice();
}

// Create cone wireframe
fn createConeWireframe(alloc: std.mem.Allocator, radius: f32, height: f32, color: [3]f32) ![]Mesh.Vertex {
    var vertices = std.ArrayList(Mesh.Vertex).init(alloc);
    defer vertices.deinit();

    const half_height = height / 2.0;
    const segments = 16;

    // Base circle
    for (0..segments) |i| {
        const angle1 = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / @as(f32, @floatFromInt(segments));
        const angle2 = @as(f32, @floatFromInt((i + 1) % segments)) * 2.0 * std.math.pi / @as(f32, @floatFromInt(segments));

        const x1 = radius * @cos(angle1);
        const z1 = radius * @sin(angle1);
        const x2 = radius * @cos(angle2);
        const z2 = radius * @sin(angle2);

        // Base circle
        try vertices.append(.{ .position = .{ x1, -half_height, z1 }, .color = color });
        try vertices.append(.{ .position = .{ x2, -half_height, z2 }, .color = color });

        // Lines to apex (only every 4th segment to avoid clutter)
        if (i % 4 == 0) {
            try vertices.append(.{ .position = .{ x1, -half_height, z1 }, .color = color });
            try vertices.append(.{ .position = .{ 0, half_height, 0 }, .color = color });
        }
    }

    return try vertices.toOwnedSlice();
}

// Create triangle mesh wireframe from original mesh data
fn createTriangleMeshWireframe(alloc: std.mem.Allocator, mesh: *Mesh, color: [3]f32) ![]Mesh.Vertex {
    var vertices = std.ArrayList(Mesh.Vertex).init(alloc);
    defer vertices.deinit();

    // Create wireframe by drawing triangle edges
    if (mesh.indices) |indices| {
        // Use indices if available
        var i: usize = 0;
        while (i < indices.len) : (i += 3) {
            if (i + 2 < indices.len) {
                const v0 = mesh.vertices[indices[i]];
                const v1 = mesh.vertices[indices[i + 1]];
                const v2 = mesh.vertices[indices[i + 2]];

                // Add triangle edges
                try vertices.append(.{ .position = v0.position, .color = color });
                try vertices.append(.{ .position = v1.position, .color = color });

                try vertices.append(.{ .position = v1.position, .color = color });
                try vertices.append(.{ .position = v2.position, .color = color });

                try vertices.append(.{ .position = v2.position, .color = color });
                try vertices.append(.{ .position = v0.position, .color = color });
            }
        }
    } else {
        // No indices, assume sequential triangles
        var i: usize = 0;
        while (i < mesh.vertices.len) : (i += 3) {
            if (i + 2 < mesh.vertices.len) {
                const v0 = mesh.vertices[i];
                const v1 = mesh.vertices[i + 1];
                const v2 = mesh.vertices[i + 2];

                // Add triangle edges
                try vertices.append(.{ .position = v0.position, .color = color });
                try vertices.append(.{ .position = v1.position, .color = color });

                try vertices.append(.{ .position = v1.position, .color = color });
                try vertices.append(.{ .position = v2.position, .color = color });

                try vertices.append(.{ .position = v2.position, .color = color });
                try vertices.append(.{ .position = v0.position, .color = color });
            }
        }
    }

    return try vertices.toOwnedSlice();
}

// Create convex hull wireframe
fn createConvexHullWireframe(alloc: std.mem.Allocator, hull: Collisions.ConvexHullShape, color: [3]f32) ![]Mesh.Vertex {
    var vertices = std.ArrayList(Mesh.Vertex).init(alloc);
    defer vertices.deinit();

    // Create wireframe from triangle edges
    for (0..hull.n_triangles) |i| {
        const base_idx = i * 3;
        if (base_idx + 2 >= hull.triangles.len) continue;

        const idx0 = hull.triangles[base_idx + 0];
        const idx1 = hull.triangles[base_idx + 1];
        const idx2 = hull.triangles[base_idx + 2];

        if (idx0 * 3 + 2 >= hull.points.len or
            idx1 * 3 + 2 >= hull.points.len or
            idx2 * 3 + 2 >= hull.points.len) continue;

        const v0 = [3]f32{
            @floatCast(hull.points[idx0 * 3 + 0]),
            @floatCast(hull.points[idx0 * 3 + 1]),
            @floatCast(hull.points[idx0 * 3 + 2]),
        };
        const v1 = [3]f32{
            @floatCast(hull.points[idx1 * 3 + 0]),
            @floatCast(hull.points[idx1 * 3 + 1]),
            @floatCast(hull.points[idx1 * 3 + 2]),
        };
        const v2 = [3]f32{
            @floatCast(hull.points[idx2 * 3 + 0]),
            @floatCast(hull.points[idx2 * 3 + 1]),
            @floatCast(hull.points[idx2 * 3 + 2]),
        };

        // Add triangle edges
        try vertices.append(.{ .position = v0, .color = color });
        try vertices.append(.{ .position = v1, .color = color });

        try vertices.append(.{ .position = v1, .color = color });
        try vertices.append(.{ .position = v2, .color = color });

        try vertices.append(.{ .position = v2, .color = color });
        try vertices.append(.{ .position = v0, .color = color });
    }

    return try vertices.toOwnedSlice();
}
