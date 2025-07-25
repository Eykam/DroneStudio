// src/core/ecs/components/Collider.zig
const std = @import("std");
const c = @import("../../bindings/c.zig");
const Core = @import("../Core.zig");
const ECSManager = @import("../ECSManager.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const GLTF = @import("../../GLTF.zig");
const Transform = @import("../components/Transform.zig");
const Physics = @import("../components/Physics.zig");
const Mesh = @import("../../Mesh.zig");
const ResourceManager = @import("../ResourceManager.zig");
const Math = @import("../../Math.zig");
const PhysicsThread = @import("PhysicsThread.zig");
const Renderer = @import("Renderer.zig");

const TransformComponent = Transform.TransformComponent;
const PhysicsComponent = Physics.PhysicsComponent;

const bullet = c.bullet;
const vhacd = c.vhacd;
const Vec3 = Math.Vec3;
const Mat4 = Math.Mat4;
const Quaternion = Math.Quaternion;

// Shape-specific data structures
pub const BoxShape = struct {
    half_extents: [3]f32,

    pub fn createBulletShape(self: BoxShape, bullet_shape: bullet.CbtShapeHandle) void {
        bullet.cbtShapeBoxCreate(bullet_shape, &self.half_extents);
    }
};

pub const SphereShape = struct {
    radius: f32,

    pub fn createBulletShape(self: SphereShape, bullet_shape: bullet.CbtShapeHandle) void {
        bullet.cbtShapeSphereCreate(bullet_shape, self.radius);
    }
};

pub const CapsuleShape = struct {
    radius: f32,
    height: f32,

    pub fn createBulletShape(self: CapsuleShape, bullet_shape: bullet.CbtShapeHandle) void {
        bullet.cbtShapeCapsuleCreate(bullet_shape, self.radius, self.height, bullet.CBT_LINEAR_AXIS_Y);
    }
};

pub const CylinderShape = struct {
    half_extents: [3]f32,

    pub fn createBulletShape(self: CylinderShape, bullet_shape: bullet.CbtShapeHandle) void {
        bullet.cbtShapeCylinderCreate(bullet_shape, &self.half_extents, bullet.CBT_LINEAR_AXIS_Y);
    }
};

pub const ConeShape = struct {
    radius: f32,
    height: f32,

    pub fn createBulletShape(self: ConeShape, bullet_shape: bullet.CbtShapeHandle) void {
        bullet.cbtShapeConeCreate(bullet_shape, self.radius, self.height, bullet.CBT_LINEAR_AXIS_Y);
    }
};

pub const ConvexHullShape = struct {
    points: []f64, // Flattened array: x1,y1,z1,x2,y2,z2,...
    triangles: []u32, // Triangle indices: i1,i2,i3,i4,i5,i6,...
    n_points: u32,
    n_triangles: u32,

    // Generate convex hulls from mesh using V-HACD
    pub fn generateFromMesh(allocator: std.mem.Allocator, mesh: *Mesh) ![]ConvexHullShape {
        // Create V-HACD instance
        const vhacd_handle = vhacd.vhacd_create();
        if (vhacd_handle == null) {
            return error.VHACDCreationFailed;
        }
        defer vhacd.vhacd_release(vhacd_handle);

        // Set V-HACD parameters optimized for speed/accuracy balance
        vhacd.vhacd_set_resolution(vhacd_handle, 100000); // Balanced resolution (100k)
        vhacd.vhacd_set_max_convex_hulls(vhacd_handle, 32); // Reasonable hull count for performance
        vhacd.vhacd_set_max_num_vertices_per_ch(vhacd_handle, 64); // Balanced vertex count per hull
        vhacd.vhacd_set_minimum_volume_percent_error_allowed(vhacd_handle, 1.0); // 1% error tolerance (reasonable)
        vhacd.vhacd_set_max_recursion_depth(vhacd_handle, 10); // Moderate recursion depth
        vhacd.vhacd_set_shrink_wrap(vhacd_handle, 1); // Keep shrink wrap for better fit
        vhacd.vhacd_set_fill_mode(vhacd_handle, vhacd.VHACD_FILL_FLOOD_FILL);

        // Convert mesh vertices to float array for V-HACD
        const vertex_count = mesh.vertices.len;
        const triangle_count = if (mesh.indices) |indices| indices.len / 3 else mesh.vertices.len / 3;

        var points = try allocator.alloc(f32, vertex_count * 3);
        defer allocator.free(points);

        for (mesh.vertices, 0..) |vertex, i| {
            points[i * 3 + 0] = vertex.position[0];
            points[i * 3 + 1] = vertex.position[1];
            points[i * 3 + 2] = vertex.position[2];
        }

        // Process with V-HACD
        const result = if (mesh.indices) |indices|
            vhacd.vhacd_compute_float(
                vhacd_handle,
                points.ptr,
                @intCast(vertex_count),
                indices.ptr,
                @intCast(triangle_count),
            )
        else blk: {
            // Create sequential indices if none exist
            var sequential_indices = try allocator.alloc(u32, triangle_count * 3);
            defer allocator.free(sequential_indices);
            for (0..triangle_count * 3) |i| {
                sequential_indices[i] = @intCast(i);
            }
            break :blk vhacd.vhacd_compute_float(
                vhacd_handle,
                points.ptr,
                @intCast(vertex_count),
                sequential_indices.ptr,
                @intCast(triangle_count),
            );
        };
        if (result != 1) {
            return error.VHACDComputeFailed;
        }

        // Get results
        const n_hulls = vhacd.vhacd_get_n_convex_hulls(vhacd_handle);
        if (n_hulls == 0) {
            return error.NoConvexHullsGenerated;
        }

        var hulls = try allocator.alloc(ConvexHullShape, n_hulls);

        std.debug.print("Num Hulls => {d}\n", .{n_hulls});

        for (0..n_hulls) |i| {
            var hull_points: [*c]f64 = undefined;
            var hull_triangles: [*c]u32 = undefined;
            var n_points: u32 = undefined;
            var n_triangles: u32 = undefined;

            const get_result = vhacd.vhacd_get_convex_hull(
                vhacd_handle,
                @intCast(i),
                &hull_points,
                &n_points,
                &hull_triangles,
                &n_triangles,
            );
            if (get_result != 1) {
                // Clean up partial results
                for (0..i) |j| {
                    allocator.free(hulls[j].points);
                    allocator.free(hulls[j].triangles);
                }
                allocator.free(hulls);
                return error.VHACDGetHullFailed;
            }

            // Copy hull data
            hulls[i].n_points = n_points;
            hulls[i].n_triangles = n_triangles;

            // Handle empty hulls safely
            if (n_points > 0 and n_points <= 10000) { // Reasonable upper bound
                const points_len = n_points * 3;
                hulls[i].points = try allocator.alloc(f64, points_len);
                for (0..points_len) |j| {
                    hulls[i].points[j] = hull_points[j];
                }
            } else {
                hulls[i].points = try allocator.alloc(f64, 0);
            }

            if (n_triangles > 0 and n_triangles <= 10000) { // Reasonable upper bound
                const triangles_len = n_triangles * 3;
                hulls[i].triangles = try allocator.alloc(u32, triangles_len);
                for (0..triangles_len) |j| {
                    hulls[i].triangles[j] = hull_triangles[j];
                }
            } else {
                hulls[i].triangles = try allocator.alloc(u32, 0);
            }

            std.debug.print("Hull: {d}\n", .{i});
            std.debug.print("Points: {d} Triangles: {d}\n", .{ n_points, n_triangles });
            std.debug.print("Points: {d:.2}\n", .{hulls[i].points});
            std.debug.print("Triangles {d:.2}\n\n", .{hulls[i].triangles});
        }

        return hulls;
    }

    pub fn createBulletShape(hull: ConvexHullShape, allocator: std.mem.Allocator) ?bullet.CbtShapeHandle {
        const child_shape = bullet.cbtShapeAllocate(bullet.CBT_SHAPE_TYPE_CONVEX_HULL);
        if (child_shape == null) return null;

        // Convert points from f64 to f32 for Bullet
        var points_f32 = allocator.alloc(f32, hull.n_points * 3) catch return null;
        defer allocator.free(points_f32);

        for (0..hull.n_points * 3) |i| {
            points_f32[i] = @floatCast(hull.points[i]);
        }

        // Create convex hull shape using Bullet's convex hull API
        bullet.cbtShapeConvexHullCreate(child_shape, points_f32.ptr, @intCast(hull.n_points), @sizeOf(f32) * 3);

        return child_shape;
    }
};

pub const TriangleMeshShape = struct {
    // Triangle mesh data is stored in the Mesh itself

    pub fn createBulletShape(self: TriangleMeshShape, allocator: std.mem.Allocator, bullet_shape: bullet.CbtShapeHandle, mesh: ?*Mesh) !void {
        _ = self; // unused

        bullet.cbtShapeTriMeshCreateBegin(bullet_shape);

        if (mesh) |m| {
            if (m.indices) |indices| {
                const num_triangles = @as(i32, @intCast(indices.len / 3));
                const triangle_stride = @sizeOf(u32) * 3;
                const num_vertices = @as(i32, @intCast(m.vertices.len));
                const vertex_stride = @sizeOf(Mesh.Vertex);

                bullet.cbtShapeTriMeshAddIndexVertexArray(
                    bullet_shape,
                    num_triangles,
                    indices.ptr,
                    triangle_stride,
                    num_vertices,
                    m.vertices.ptr,
                    vertex_stride,
                );
            } else {
                const num_triangles = @as(i32, @intCast(m.vertices.len / 3));
                if (num_triangles > 0) {
                    var triangle_indices = std.ArrayList(u32).init(allocator);
                    defer triangle_indices.deinit();

                    var i: u32 = 0;
                    while (i < m.vertices.len) : (i += 3) {
                        if (i + 2 < m.vertices.len) {
                            try triangle_indices.append(i);
                            try triangle_indices.append(i + 1);
                            try triangle_indices.append(i + 2);
                        }
                    }

                    const triangle_stride = @sizeOf(u32) * 3;
                    const num_vertices = @as(i32, @intCast(m.vertices.len));
                    const vertex_stride = @sizeOf(Mesh.Vertex);

                    bullet.cbtShapeTriMeshAddIndexVertexArray(
                        bullet_shape,
                        num_triangles,
                        triangle_indices.items.ptr,
                        triangle_stride,
                        num_vertices,
                        m.vertices.ptr,
                        vertex_stride,
                    );
                }
            }
        }

        bullet.cbtShapeTriMeshCreateEnd(bullet_shape);
    }
};

pub const CompoundShape = struct {
    // For future compound shape implementation

    pub fn createBulletShape(self: CompoundShape, bullet_shape: bullet.CbtShapeHandle) void {
        _ = self; // unused
        bullet.cbtShapeCompoundCreate(bullet_shape, true, 0);
    }
};

// Tagged union for collider shape data
pub const ColliderShape = union(enum) {
    Box: BoxShape,
    Sphere: SphereShape,
    Capsule: CapsuleShape,
    Cylinder: CylinderShape,
    Cone: ConeShape,
    CompoundShape: CompoundShape,
    TriangleMesh: TriangleMeshShape,
    ConvexHull: []ConvexHullShape, // Array of convex hulls from V-HACD

    // Generic function to create Bullet shape
    pub fn createBulletShape(self: ColliderShape, allocator: std.mem.Allocator, bullet_shape: bullet.CbtShapeHandle, mesh: ?*Mesh) !void {
        switch (self) {
            .Box => |data| data.createBulletShape(bullet_shape),
            .Sphere => |data| data.createBulletShape(bullet_shape),
            .Capsule => |data| data.createBulletShape(bullet_shape),
            .Cylinder => |data| data.createBulletShape(bullet_shape),
            .Cone => |data| data.createBulletShape(bullet_shape),
            .CompoundShape => |data| data.createBulletShape(bullet_shape),
            .TriangleMesh => |data| try data.createBulletShape(allocator, bullet_shape, mesh),
            .ConvexHull => |data| {
                bullet.cbtShapeCompoundCreate(bullet_shape, true, @intCast(data.len));

                for (data) |hull| {
                    const child_shape = hull.createBulletShape(allocator);

                    if (child_shape == null) return error.FailedtoCreateConvexHull;

                    // Add child to compound shape with identity transform
                    var identity_transform = [4][3]f32{
                        [3]f32{ 1.0, 0.0, 0.0 },
                        [3]f32{ 0.0, 1.0, 0.0 },
                        [3]f32{ 0.0, 0.0, 1.0 },
                        [3]f32{ 0.0, 0.0, 0.0 },
                    };
                    bullet.cbtShapeCompoundAddChild(bullet_shape, &identity_transform, child_shape.?);
                }
            },
        }
    }

    // Get the Bullet shape type for this ColliderShape
    pub fn getBulletShapeType(self: ColliderShape) i32 {
        return switch (self) {
            .Box => bullet.CBT_SHAPE_TYPE_BOX,
            .Sphere => bullet.CBT_SHAPE_TYPE_SPHERE,
            .Capsule => bullet.CBT_SHAPE_TYPE_CAPSULE,
            .Cylinder => bullet.CBT_SHAPE_TYPE_CYLINDER,
            .Cone => bullet.CBT_SHAPE_TYPE_CONE,
            .CompoundShape => bullet.CBT_SHAPE_TYPE_COMPOUND,
            .TriangleMesh => bullet.CBT_SHAPE_TYPE_TRIANGLE_MESH,
            .ConvexHull => bullet.CBT_SHAPE_TYPE_COMPOUND, // Convex hulls use compound shape
        };
    }
};

// Helper function to calculate mesh bounds
fn calculateMeshBounds(mesh: *Mesh) struct { min_bounds: [3]f32, max_bounds: [3]f32, half_extents: [3]f32, dimensions: [3]f32 } {
    var min_bounds = [3]f32{ std.math.floatMax(f32), std.math.floatMax(f32), std.math.floatMax(f32) };
    var max_bounds = [3]f32{ std.math.floatMin(f32), std.math.floatMin(f32), std.math.floatMin(f32) };

    for (mesh.vertices) |vertex| {
        min_bounds[0] = @min(min_bounds[0], vertex.position[0]);
        min_bounds[1] = @min(min_bounds[1], vertex.position[1]);
        min_bounds[2] = @min(min_bounds[2], vertex.position[2]);

        max_bounds[0] = @max(max_bounds[0], vertex.position[0]);
        max_bounds[1] = @max(max_bounds[1], vertex.position[1]);
        max_bounds[2] = @max(max_bounds[2], vertex.position[2]);
    }

    const dimensions = [3]f32{
        max_bounds[0] - min_bounds[0],
        max_bounds[1] - min_bounds[1],
        max_bounds[2] - min_bounds[2],
    };

    const half_extents = [3]f32{
        dimensions[0] / 2.0,
        dimensions[1] / 2.0,
        dimensions[2] / 2.0,
    };

    return .{
        .min_bounds = min_bounds,
        .max_bounds = max_bounds,
        .half_extents = half_extents,
        .dimensions = dimensions,
    };
}

// Create a collider based on mesh data with specified shape
pub fn createColliderFromMesh(allocator: std.mem.Allocator, resource_manager: *ResourceManager, mesh: *Mesh, shape: ColliderShape) !ColliderComponent {
    switch (shape) {
        .TriangleMesh => return ColliderComponent.init(allocator, .{ .TriangleMesh = TriangleMeshShape{} }, mesh),
        .ConvexHull => |hulls| {
            // If hulls are empty, generate them from the mesh
            if (hulls.len == 0) {
                const generated_hulls = try resource_manager.getOrGenerateCollisionMesh(mesh);
                return ColliderComponent.init(allocator, .{ .ConvexHull = generated_hulls }, mesh);
            } else {
                // Use provided hulls
                return ColliderComponent.init(allocator, .{ .ConvexHull = hulls }, mesh);
            }
        },
        .Box => {
            const bounds = calculateMeshBounds(mesh);
            return ColliderComponent.init(allocator, .{ .Box = BoxShape{ .half_extents = bounds.half_extents } }, mesh);
        },
        .Sphere => {
            const bounds = calculateMeshBounds(mesh);
            const radius = (bounds.half_extents[0] + bounds.half_extents[1] + bounds.half_extents[2]) / 3.0;
            return ColliderComponent.init(allocator, .{ .Sphere = SphereShape{ .radius = radius } }, mesh);
        },
        .Capsule => {
            const bounds = calculateMeshBounds(mesh);
            const radius = (bounds.dimensions[0] + bounds.dimensions[2]) / 4.0;
            const height = bounds.dimensions[1];
            return ColliderComponent.init(allocator, .{ .Capsule = CapsuleShape{ .radius = radius, .height = height } }, mesh);
        },
        .Cylinder => {
            const bounds = calculateMeshBounds(mesh);
            return ColliderComponent.init(allocator, .{ .Cylinder = CylinderShape{ .half_extents = bounds.half_extents } }, mesh);
        },
        .Cone => {
            const bounds = calculateMeshBounds(mesh);
            const radius = @max(bounds.dimensions[0], bounds.dimensions[2]) / 2.0;
            const height = bounds.dimensions[1];
            return ColliderComponent.init(allocator, .{ .Cone = ConeShape{ .radius = radius, .height = height } }, mesh);
        },
        .CompoundShape => {
            const bounds = calculateMeshBounds(mesh);
            return ColliderComponent.init(allocator, .{ .Box = BoxShape{ .half_extents = bounds.half_extents } }, mesh);
        },
    }
}

// Separate RigidBody component for physics dynamics
pub const RigidBodyComponent = struct {
    const Self = @This();

    entity_id: Core.EntityID = undefined,
    bullet_body: ?bullet.CbtBodyHandle = null, //TODO: Remove this field since the physics thread handles this
    bullet_shape: bullet.CbtShapeHandle,
    mass: f32 = 1.0,
    initial_position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    initial_rotation: [4]f32 = .{ 0.0, 0.0, 0.0, 1.0 }, // quaternion (x, y, z, w)

    pub fn init(mass: f32, shape: bullet.CbtShapeHandle) Self {
        return .{ .mass = mass, .bullet_shape = shape };
    }

    pub fn translate(self: *Self, offset: [3]f32) void {
        self.initial_position = [3]f32{
            self.initial_position[0] + offset[0],
            self.initial_position[1] + offset[1],
            self.initial_position[2] + offset[2],
        };
    }

    pub fn rotate(self: *Self, rotation: [4]f32) void {
        self.initial_rotation = rotation;
    }

    pub fn attach(self: *Self, ecs: *ECSManager, eid: Core.EntityID) !void {
        self.entity_id = eid;

        // Send command to physics thread to create body instead of creating in main thread
        if (ecs.collision_system.physics_thread) |physics| {
            const initial_pos = self.initial_position;
            const initial_rot = self.initial_rotation;

            // Get collision properties from collider component (if it exists)
            var restitution: f32 = 0.5;
            var friction: f32 = 0.5;
            var rolling_friction: f32 = 0.1;

            if (ecs.collision_system.collider_components.get(eid)) |collider| {
                restitution = collider.restitution;
                friction = collider.friction;
                rolling_friction = collider.rolling_friction;
            }

            const command = PhysicsThread.PhysicsCommand{ .CreateRigidBody = .{
                .entity_id = eid,
                .mass = self.mass,
                .shape_handle = self.bullet_shape,
                .initial_pos = initial_pos,
                .initial_rot = initial_rot,
                .restitution = restitution,
                .friction = friction,
                .rolling_friction = rolling_friction,
            } };

            const success = physics.sendCommand(command);
            if (success) {
                std.debug.print("Sent CreateRigidBody command for entity {d} to physics thread\n", .{eid.id});
                // Note: bullet_body remains null in main thread - physics thread manages the actual body
                self.bullet_body = null;
            } else {
                std.debug.print("Failed to send CreateRigidBody command for entity {d} - queue full\n", .{eid.id});
                return error.FailedToCreatePhysicsBody;
            }
        } else {
            std.debug.print("No physics thread available for entity {d}\n", .{eid.id});
            return error.NoPhysicsThread;
        }

        try ecs.rigid_body_components.add(eid, self.*);
    }

    pub fn deinit(self: *Self) void {
        if (self.bullet_body) |body| {
            bullet.cbtBodyDeallocate(body);
            self.bullet_body = null;
        }
    }
};

// Collider component now focuses only on shape and collision properties
pub const ColliderComponent = struct {
    const Self = @This();

    // Shape and collision properties only (no physics)
    entity_id: Core.EntityID = undefined,
    shape: ColliderShape,
    bullet_shape: ?bullet.CbtShapeHandle = null,

    // Collision properties
    friction: f32 = 0.5,
    rolling_friction: f32 = 0.1,
    restitution: f32 = 0.0,
    collision_group: u16 = 1,
    collision_mask: u16 = 0xFFFF, // Collide with everything by default

    // For simple shapes (box, sphere, triangle mesh, etc.)
    pub fn init(allocator: std.mem.Allocator, shape: ColliderShape, mesh: ?*Mesh) !Self {
        var collider = Self{
            .shape = shape,
        };

        collider.bullet_shape = bullet.cbtShapeAllocate(shape.getBulletShapeType());
        if (collider.bullet_shape == null) return error.ShapeAllocationFailed;

        try shape.createBulletShape(allocator, collider.bullet_shape.?, mesh);

        return collider;
    }

    // Depth-first traversal helper for processing GLTF nodes with accumulated transforms
    fn processNodeDFS(
        allocator: std.mem.Allocator,
        resource_manager: *ResourceManager,
        model_resource: *GLTF.ModelResource,
        compound_shape: bullet.CbtShapeHandle,
        base_shape: ColliderShape,
        node_index: usize,
        accumulated_transform: Mat4,
    ) !void {
        if (node_index >= model_resource.entities.len) return;

        const node = model_resource.entities[node_index];

        // Calculate this node's transform by combining accumulated + local
        var current_transform = accumulated_transform;

        std.debug.print("Node {d}: Processing (accumulated transform pos=[{d:.3}, {d:.3}, {d:.3}])\n", .{ node_index, accumulated_transform.get_position().x(), accumulated_transform.get_position().y(), accumulated_transform.get_position().z() });

        if (node.local_transformation) |local_transform| {
            std.debug.print("Node {d}: has local_transformation\n", .{node_index});
            current_transform = local_transform.multiply(current_transform);
        } else {
            var local_matrix = Mat4.identity();

            // Apply translation
            if (node.translation) |t| {
                local_matrix = local_matrix.translate(t[0], t[1], t[2]);
            }

            // Apply rotation
            if (node.rotation) |r| {
                const quat = Quaternion.init(r[0], r[1], r[2], r[3]);
                local_matrix = local_matrix.multiply(Mat4.from_quaternion(quat));
            }

            // Apply scale
            if (node.scale) |s| {
                local_matrix = local_matrix.scale(s[0], s[1], s[2]);
            }

            current_transform = local_matrix.multiply(current_transform);
        }

        // If this node has a mesh, create collision shape with accumulated transform
        var transform_for_children = current_transform;
        if (node.mesh_name) |mesh_name| {
            if (resource_manager.meshes.get(mesh_name)) |*mesh_res| {
                // Extract the 3x3 rotation matrix and translation directly from current_transform
                const m = current_transform.base.data;

                // Create child shape
                const shape_type = base_shape.getBulletShapeType();
                const mesh_shape = bullet.cbtShapeAllocate(shape_type);

                if (mesh_shape != null) {
                    var indices_opt: ?[]u32 = null;
                    defer if (indices_opt) |indices| allocator.free(indices);

                    if (mesh_res.mesh.indices) |indices| {
                        indices_opt = try allocator.dupe(u32, indices);
                    }

                    var temp_mesh = try Mesh.init(allocator, mesh_res.mesh.vertices, indices_opt, mesh_res.mesh._draw);
                    defer temp_mesh.deinit();

                    var mesh_collider = try createColliderFromMesh(allocator, resource_manager, temp_mesh, base_shape);
                    try mesh_collider.shape.createBulletShape(allocator, mesh_shape.?, temp_mesh);

                    var child_transform = [4][3]f32{
                        [3]f32{ m[0], m[1], m[2] }, // First column of rotation
                        [3]f32{ m[4], m[5], m[6] }, // Second column of rotation
                        [3]f32{ m[8], m[9], m[10] }, // Third column of rotation
                        [3]f32{ m[12], m[13], m[14] }, // Translation
                    };

                    std.debug.print("  Adding mesh '{s}' to compound with transform:\n", .{mesh_name});
                    std.debug.print("    Row 0: [{d:.3}, {d:.3}, {d:.3}] (X axis)\n", .{ child_transform[0][0], child_transform[0][1], child_transform[0][2] });
                    std.debug.print("    Row 1: [{d:.3}, {d:.3}, {d:.3}] (Y axis)\n", .{ child_transform[1][0], child_transform[1][1], child_transform[1][2] });
                    std.debug.print("    Row 2: [{d:.3}, {d:.3}, {d:.3}] (Z axis)\n", .{ child_transform[2][0], child_transform[2][1], child_transform[2][2] });
                    std.debug.print("    Row 3: [{d:.3}, {d:.3}, {d:.3}] (Position)\n", .{ child_transform[3][0], child_transform[3][1], child_transform[3][2] });

                    bullet.cbtShapeCompoundAddChild(compound_shape, &child_transform, mesh_shape.?);

                    // Reset transform accumulation for children since this node has a mesh
                    transform_for_children = Mat4.identity();
                    std.debug.print("Node {d}: Found mesh, resetting transform accumulation for children\n", .{node_index});
                }
            }
        }

        // TODO: Fix the GLTF parser to properly populate children arrays with entity indices
        // instead of GLTF node indices. For now, use O(n²) solution.

        // Process all nodes that have this node as their parent
        for (model_resource.entities, 0..) |child_node, child_idx| {
            if (child_node.parent_idx) |parent| {
                if (parent == node_index) {
                    std.debug.print("  Node {d} has child {d}\n", .{ node_index, child_idx });
                    try processNodeDFS(allocator, resource_manager, model_resource, compound_shape, base_shape, child_idx, transform_for_children);
                }
            }
        }
    }

    // For compound shapes from GLTF models
    pub fn initFromModel(
        allocator: std.mem.Allocator,
        model_resource: *GLTF.ModelResource,
        base_shape: ColliderShape,
        resource_manager: *ResourceManager,
    ) !Self {
        var mesh_count: u32 = 0;
        for (model_resource.entities) |node| {
            if (node.mesh_name != null) mesh_count += 1;
        }

        if (mesh_count == 0) return error.NoMeshesToCreateColliderFrom;
        var compound_collider = Self{
            .shape = .{ .CompoundShape = CompoundShape{} },
        };

        // Allocate compound shape
        compound_collider.bullet_shape = bullet.cbtShapeAllocate(bullet.CBT_SHAPE_TYPE_COMPOUND);
        if (compound_collider.bullet_shape == null) return error.ShapeAllocationFailed;

        bullet.cbtShapeCompoundCreate(
            compound_collider.bullet_shape.?,
            true,
            @intCast(mesh_count),
        );

        // The entities array is flattened, so we need to process all nodes
        // and accumulate transforms based on parent_idx relationships
        // Process nodes starting from those with no parent (parent_idx == null)
        for (model_resource.entities, 0..) |node, i| {
            if (node.parent_idx == null) {
                std.debug.print("Processing root node {d} with no parent\n", .{i});
                try processNodeDFS(allocator, resource_manager, model_resource, compound_collider.bullet_shape.?, base_shape, i, Mat4.identity());
            }
        }

        return compound_collider;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        if (self.bullet_shape) |shape| {
            bullet.cbtShapeDeallocate(shape);
            self.bullet_shape = null;
        }

        // Clean up shape-specific data
        switch (self.shape) {
            .ConvexHull => |hulls| {
                for (hulls) |hull| {
                    allocator.free(hull.points);
                    allocator.free(hull.triangles);
                }
                allocator.free(hulls);
            },
            else => {},
        }
    }

    pub fn attach(self: *Self, ecs: *ECSManager, eid: Core.EntityID) !void {
        self.entity_id = eid;
        try ecs.collider_components.add(eid, self.*);
    }
};

pub const CollisionSystem = struct {
    const Self = @This();

    world: *Core.World,
    allocator: std.mem.Allocator,
    transform_components: *SparseSet(TransformComponent),
    rigid_body_components: *SparseSet(RigidBodyComponent),
    collider_components: *SparseSet(ColliderComponent),
    renderer_components: *SparseSet(Renderer.Renderable),

    // Threaded physics system
    physics_thread: ?*PhysicsThread.ThreadedPhysicsSystem = null,
    debug_wireframe_system: ?DebugWireframeSystem = null,

    pub fn init(
        allocator: std.mem.Allocator,
        world: *Core.World,
        transform_components: *SparseSet(TransformComponent),
        rigid_body_components: *SparseSet(RigidBodyComponent),
        collider_components: *SparseSet(ColliderComponent),
        renderer_components: *SparseSet(Renderer.Renderable),
    ) !Self {
        var system = Self{
            .world = world,
            .allocator = allocator,
            .transform_components = transform_components,
            .rigid_body_components = rigid_body_components,
            .collider_components = collider_components,
            .renderer_components = renderer_components,
        };

        try system.initThreadedPhysics();

        return system;
    }

    pub fn deinit(self: *Self) void {
        // Clean up debug wireframe system
        if (self.debug_wireframe_system) |*debug_system| {
            debug_system.deinit();
            self.debug_wireframe_system = null;
        }

        // Clean up threaded physics system
        if (self.physics_thread) |physics| {
            physics.deinit();
            self.physics_thread = null;
        }
    }

    fn initThreadedPhysics(self: *Self) !void {
        // Initialize the threaded physics system with thread-safe allocator
        self.physics_thread = try PhysicsThread.ThreadedPhysicsSystem.init(self.allocator);

        // Initialize debug wireframe system
        if (self.physics_thread) |physics_thread| {
            self.debug_wireframe_system = try DebugWireframeSystem.init(self.allocator, physics_thread);
            std.debug.print("Debug wireframe system initialized\n", .{});
        }
    }

    pub fn update(self: *Self) !void {
        if (self.physics_thread) |physics| {
            const physics_states = physics.getPhysicsStates();

            // Update transform components with latest physics data
            for (physics_states) |state| {
                if (self.transform_components.get(state.entity_id)) |transform| {
                    // Debug output for physics state updates
                    if (state.entity_id.id < 5) { // Only show first few entities to avoid spam
                        std.debug.print("CollisionSystem: Updating entity {d} transform from physics: pos=[{d:.3}, {d:.3}, {d:.3}]\n", .{
                            state.entity_id.id,
                            state.position[0],
                            state.position[1],
                            state.position[2],
                        });
                    }

                    // Update position and rotation from physics thread
                    transform.position = state.position;

                    const quat = Math.Quaternion{ .data = state.rotation };
                    transform.rotation = quat.normalize();
                    transform.updateLocalTransform();
                }
            }
        }

        // Update debug wireframe system if enabled
        if (self.debug_wireframe_system) |*debug_system| {
            debug_system.update(self.world, self.world.resource_manager, self.renderer_components, self.transform_components) catch |err| {
                std.debug.print("Error updating debug wireframe system: {any}\n", .{err});
            };
        }
    }

    // Reset all dynamic bodies to their initial state
    pub fn resetAllDynamicBodies(self: *Self) void {
        std.debug.print("Starting resetAllDynamicBodies...\n", .{});

        // In threaded physics, send reset command to physics thread
        if (self.physics_thread) |physics| {
            const command = PhysicsThread.PhysicsCommand{ .ResetDynamicBodies = .{} };
            const success = physics.sendCommand(command);

            if (success) {
                std.debug.print("Sent ResetDynamicBodies command to physics thread\n", .{});

                // Also send SetTransform commands for each dynamic body to reset positions
                var it = self.rigid_body_components.iterator();
                var reset_count: u32 = 0;

                while (it.next()) |entry| {
                    const entity_id = entry.entity_id;
                    const rigid_body = entry.component;

                    // Only reset dynamic bodies (mass > 0)
                    if (rigid_body.mass > 0.0) {
                        // Send SetTransform command to physics thread
                        const transform_command = PhysicsThread.PhysicsCommand{ .SetTransform = .{
                            .entity_id = entity_id,
                            .position = rigid_body.initial_position,
                            .rotation = rigid_body.initial_rotation,
                        } };

                        if (physics.sendCommand(transform_command)) {
                            // Update the transform component in main thread to match
                            if (self.transform_components.get(entity_id)) |transform| {
                                transform.setPosition(rigid_body.initial_position[0], rigid_body.initial_position[1], rigid_body.initial_position[2]);
                                // Reset rotation to identity quaternion (should match initial_rotation but using identity for simplicity)
                                const identity_quat = Math.Quaternion.identity();
                                transform.setRotation(identity_quat);
                            }

                            reset_count += 1;
                            std.debug.print(
                                "Reset entity {d} to initial position: [{d:.2}, {d:.2}, {d:.2}]\n",
                                .{
                                    entity_id.id,
                                    rigid_body.initial_position[0],
                                    rigid_body.initial_position[1],
                                    rigid_body.initial_position[2],
                                },
                            );
                        } else {
                            std.debug.print(
                                "Failed to send SetTransform command for entity {d} - queue full\n",
                                .{entity_id.id},
                            );
                        }
                    }
                }

                std.debug.print(
                    "Reset complete! {d} dynamic bodies reset via physics thread commands.\n",
                    .{reset_count},
                );
            } else {
                std.debug.print(
                    "Failed to send ResetDynamicBodies command - physics command queue full\n",
                    .{},
                );
            }
        } else {
            std.debug.print("No physics thread available for reset operation\n", .{});
        }
    }

    // Apply a central force to an entity via physics thread
    pub fn applyCentralForce(self: *Self, entity_id: Core.EntityID, force: [3]f32) void {
        if (self.physics_thread) |physics| {
            const command = PhysicsThread.PhysicsCommand{ .ApplyForce = .{ .entity_id = entity_id, .force = force } };
            const success = physics.sendCommand(command);

            if (success) {
                std.debug.print(
                    "Sent force command [{d:.2}, {d:.2}, {d:.2}] for eid {d}\n",
                    .{
                        force[0],
                        force[1],
                        force[2],
                        entity_id.id,
                    },
                );
            } else {
                std.debug.print(
                    "Failed to send force command for eid {d} - command queue full\n",
                    .{entity_id.id},
                );
            }
        } else {
            std.debug.print("No physics thread available for eid {d}\n", .{entity_id.id});
        }
    }

    // Apply a central impulse to an entity via physics thread
    pub fn applyCentralImpulse(self: *Self, entity_id: Core.EntityID, impulse: [3]f32) void {
        if (self.physics_thread) |physics| {
            const command = PhysicsThread.PhysicsCommand{ .ApplyImpulse = .{ .entity_id = entity_id, .impulse = impulse } };
            const success = physics.sendCommand(command);

            if (!success) {
                std.debug.print("Failed to send impulse command for eid {d} - command queue full\n", .{entity_id.id});
            }
        } else {
            std.debug.print("No physics thread available for impulse on eid {d}\n", .{entity_id.id});
        }
    }

    // Apply a torque to an entity via physics thread
    pub fn applyTorque(self: *Self, entity_id: Core.EntityID, torque: [3]f32) void {
        if (self.physics_thread) |physics| {
            const command = PhysicsThread.PhysicsCommand{ .ApplyTorque = .{ .entity_id = entity_id, .torque = torque } };
            const success = physics.sendCommand(command);

            if (!success) {
                std.debug.print("Failed to send torque command for eid {d} - command queue full\n", .{entity_id.id});
            }
        } else {
            std.debug.print("No physics thread available for torque on eid {d}\n", .{entity_id.id});
        }
    }

    // Perform a ray test in the world
    pub fn rayTest(self: *Self, ray_from: [3]f32, ray_to: [3]f32, collision_filter_group: i32, collision_filter_mask: i32) ?struct {
        hit_entity: ?Core.EntityID,
        hit_point: [3]f32,
        hit_normal: [3]f32,
        hit_fraction: f32,
    } {
        if (self.bullet_world) |world| {
            var result: bullet.CbtRayCastResult = undefined;

            const hit = bullet.cbtWorldRayTestClosest(
                world,
                &ray_from,
                &ray_to,
                collision_filter_group,
                collision_filter_mask,
                bullet.CBT_RAYCAST_FLAG_NONE,
                &result,
            );

            if (hit) {
                // Look up the entity ID from the body
                var hit_entity: ?Core.EntityID = null;

                var it = self.collider_components.iterator();
                while (it.next()) |entry| {
                    if (entry.component.bullet_body == result.body) {
                        hit_entity = entry.entity_id;
                        break;
                    }
                }

                return .{
                    .hit_entity = hit_entity,
                    .hit_point = result.hit_point_world,
                    .hit_normal = result.hit_normal_world,
                    .hit_fraction = result.hit_fraction,
                };
            }
        }

        return null;
    }

    /// Enable or disable debug wireframes
    pub fn setDebugWireframes(self: *Self, enabled: bool) !void {
        if (self.debug_wireframe_system) |*debug_system| {
            try debug_system.setEnabled(enabled);
        } else {
            return error.DebugWireframeSystemNotInitialized;
        }
    }

    /// Pause or resume physics simulation
    pub fn setPhysicsPaused(self: *Self, paused: bool) void {
        if (self.physics_thread) |physics| {
            const success = physics.setPhysicsPaused(paused);
            if (!success) {
                std.debug.print("Failed to send pause command to physics thread - queue full\n", .{});
            }
        } else {
            std.debug.print("No physics thread available for pause operation\n", .{});
        }
    }

    /// Check if physics simulation is currently paused
    pub fn isPhysicsPaused(self: *Self) bool {
        if (self.physics_thread) |physics| {
            return physics.isPhysicsPaused();
        }
        return false;
    }
};

pub const DebugWireframeSystem = struct {
    const DebugSelf = @This();

    const DebugInfo = struct {
        mesh_name: []u8,
        debug_entity: Core.EntityID,
    };

    allocator: std.mem.Allocator,
    physics_thread: *PhysicsThread.ThreadedPhysicsSystem,
    debug_entities: std.AutoHashMap(Core.EntityID, DebugInfo), // physics_entity => debug info
    enabled: bool = false,
    last_wireframe_version: u32 = 0, // Track last processed wireframe version

    pub fn init(allocator: std.mem.Allocator, physics_thread: *PhysicsThread.ThreadedPhysicsSystem) !DebugSelf {
        return DebugSelf{
            .allocator = allocator,
            .physics_thread = physics_thread,
            .debug_entities = std.AutoHashMap(Core.EntityID, DebugInfo).init(allocator),
            .enabled = false,
        };
    }

    pub fn deinit(self: *DebugSelf) void {
        // Free all cached mesh names
        var it = self.debug_entities.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.mesh_name);
        }
        self.debug_entities.deinit();
    }

    /// Enable or disable debug wireframes
    pub fn setEnabled(self: *DebugSelf, enabled: bool) !void {
        if (self.enabled == enabled) return; // No change

        self.enabled = enabled;

        if (enabled) {
            // Enable debug wireframes in physics thread
            const dynamic_color = [3]f32{ 0.0, 1.0, 0.0 }; // Green for dynamic
            const static_color = [3]f32{ 0.0, 0.0, 1.0 }; // Blue for static

            const success = self.physics_thread.setDebugWireframes(true, dynamic_color, static_color);
            if (!success) {
                return error.FailedToEnableDebugWireframes;
            }

            std.debug.print("DebugWireframeSystem: Enabled debug wireframes (renderables will be created when wireframes are ready)\n", .{});
        } else {
            // Disable debug wireframes in physics thread
            const success = self.physics_thread.setDebugWireframes(false, .{ 0, 0, 0 }, .{ 0, 0, 0 });
            if (!success) {
                return error.FailedToDisableDebugWireframes;
            }

            std.debug.print("DebugWireframeSystem: Disabled debug wireframes (cleanup will happen in update)\n", .{});
        }
    }

    pub fn update(
        self: *DebugSelf, 
        world: *Core.World,
        resource_manager: *ResourceManager,
        renderer_components: *SparseSet(Renderer.Renderable),
        transform_components: *SparseSet(Transform.TransformComponent)
    ) !void {
        if (!self.enabled) {
            // Clean up when disabled
            if (self.debug_entities.count() > 0) {
                self.cleanupDebugEntities(renderer_components, transform_components);
            }
            return;
        }

        // Check if wireframes have been updated
        const current_version = self.physics_thread.getWireframeVersion();
        if (current_version > self.last_wireframe_version) {
            std.debug.print("DebugWireframeSystem: New wireframes available (version {} -> {})\n", .{self.last_wireframe_version, current_version});
            
            // Clean up old debug entities first
            self.cleanupDebugEntities(renderer_components, transform_components);

            // Create new debug entities with fresh wireframes
            try self.createDebugEntities(world, resource_manager, renderer_components, transform_components);

            self.last_wireframe_version = current_version;
            std.debug.print("DebugWireframeSystem: Created {} debug entities for version {}\n", .{self.debug_entities.count(), current_version});
        }

        // Update transforms for existing debug entities
        const physics_states = self.physics_thread.getPhysicsStates();
        for (physics_states) |state| {
            if (self.debug_entities.get(state.entity_id)) |debug_info| {
                if (transform_components.get(debug_info.debug_entity)) |transform| {
                    transform.setPosition(state.position[0], state.position[1], state.position[2]);
                    const quat = Math.Quaternion{ .data = state.rotation };
                    transform.setRotation(quat);
                }
            }
        }
    }

    /// Clean up all debug entities and their components
    fn cleanupDebugEntities(
        self: *DebugSelf,
        renderer_components: *SparseSet(Renderer.Renderable),
        transform_components: *SparseSet(Transform.TransformComponent)
    ) void {
        var it = self.debug_entities.iterator();
        while (it.next()) |entry| {
            const debug_info = entry.value_ptr.*;
            
            // Remove components from debug entity
            _ = renderer_components.remove(debug_info.debug_entity) catch false;
            _ = transform_components.remove(debug_info.debug_entity) catch false;
            
            // Free mesh name
            self.allocator.free(debug_info.mesh_name);
        }
        self.debug_entities.clearAndFree();
        std.debug.print("DebugWireframeSystem: Cleaned up debug entities\n", .{});
    }

    /// Create debug entities from current wireframe data
    fn createDebugEntities(
        self: *DebugSelf,
        world: *Core.World,
        resource_manager: *ResourceManager,
        renderer_components: *SparseSet(Renderer.Renderable),
        transform_components: *SparseSet(Transform.TransformComponent)
    ) !void {
        const wireframes = self.physics_thread.getDebugWireframes();
        for (wireframes) |wireframe_data| {
            const mesh_name = try self.getMeshName(wireframe_data.entity_id);
            
            // Create a debug entity for this physics entity
            const debug_entity = try world.createEntity();
            
            // Add transform component to debug entity (will be updated each frame)
            const initial_transform = Transform.TransformComponent.init(self.allocator);
            try transform_components.add(debug_entity, initial_transform);
            
            // Create mesh from wireframe vertices
            const mesh_vertices = try self.allocator.alloc(Mesh.Vertex, wireframe_data.vertices.len);
            for (wireframe_data.vertices, 0..) |vertex, i| {
                mesh_vertices[i] = Mesh.Vertex{
                    .position = vertex.position,
                    .color = vertex.color,
                    .normal = .{ 0.0, 0.0, 1.0 },
                    .texture = .{ 0.0, 0.0 },
                };
            }
            
            // Create mesh in resource manager
            try resource_manager.updateMesh(mesh_name, mesh_vertices, null, Mesh.gen_draw(.lines));
            
            // Create renderable for debug entity
            const renderable = try Renderer.Renderable.init(self.allocator, mesh_name);
            try renderer_components.add(debug_entity, renderable);
            
            // Store mapping from physics entity to debug entity
            const debug_info = DebugInfo{
                .mesh_name = mesh_name,
                .debug_entity = debug_entity,
            };
            try self.debug_entities.put(wireframe_data.entity_id, debug_info);
            
            std.debug.print("Created debug entity {d} for physics entity {d}\n", .{debug_entity.id, wireframe_data.entity_id.id});
        }
    }

    /// Generate a unique mesh name for a physics entity
    fn getMeshName(self: *DebugSelf, physics_eid: Core.EntityID) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "debug_wireframe_{d}", .{physics_eid.id});
    }
};
