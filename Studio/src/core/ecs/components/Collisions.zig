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

        // Set V-HACD parameters for maximum accuracy
        vhacd.vhacd_set_resolution(vhacd_handle, 1000000); // Increased from 100k to 1M for much higher detail
        vhacd.vhacd_set_max_convex_hulls(vhacd_handle, 512); // Allow more hulls for complex geometry
        vhacd.vhacd_set_max_num_vertices_per_ch(vhacd_handle, 128); // More vertices per hull for better approximation
        vhacd.vhacd_set_minimum_volume_percent_error_allowed(vhacd_handle, 0.01); // Very low error tolerance (1%)
        vhacd.vhacd_set_max_recursion_depth(vhacd_handle, 20); // Deeper recursion for finer detail
        vhacd.vhacd_set_shrink_wrap(vhacd_handle, 1); // Enable shrink wrap for tighter fit
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
            hulls[i].points = try allocator.dupe(f64, hull_points[0 .. n_points * 3]);
            hulls[i].triangles = try allocator.dupe(u32, hull_triangles[0 .. n_triangles * 3]);

            std.debug.print("Hull: {d}\n", .{i});
            std.debug.print("Points: {d} Triangles: {d}\n", .{ n_points, n_triangles });
            std.debug.print("Points: {d:.2}\n", .{hulls[i].points});
            std.debug.print("Triangles {d:.2}\n\n", .{hulls[i].triangles});
        }

        return hulls;
    }

    pub fn createBulletShape(hull: ConvexHullShape, allocator: std.mem.Allocator) ?bullet.CbtShapeHandle {
        const child_shape = bullet.cbtShapeAllocate(bullet.CBT_SHAPE_TYPE_TRIANGLE_MESH);
        if (child_shape == null) return null;

        // Create a mesh structure from hull data
        var hull_vertices = allocator.alloc(Mesh.Vertex, hull.n_points) catch return null;
        defer allocator.free(hull_vertices);

        for (0..hull.n_points) |i| {
            hull_vertices[i] = Mesh.Vertex{
                .position = [3]f32{
                    @floatCast(hull.points[i * 3 + 0]),
                    @floatCast(hull.points[i * 3 + 1]),
                    @floatCast(hull.points[i * 3 + 2]),
                },
                .color = [3]f32{ 1.0, 1.0, 1.0 },
                .texture = null,
                .normal = null,
                .tangent = null,
                .bitangent = null,
            };
        }

        const triangle_stride = @sizeOf(u32) * 3;
        const vertex_stride = @sizeOf(Mesh.Vertex);
        bullet.cbtShapeTriMeshCreateBegin(child_shape);
        bullet.cbtShapeTriMeshAddIndexVertexArray(
            child_shape,
            @intCast(hull.n_triangles),
            hull.triangles.ptr,
            triangle_stride,
            @intCast(hull.n_points),
            hull_vertices.ptr,
            vertex_stride,
        );
        bullet.cbtShapeTriMeshCreateEnd(child_shape);

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
pub fn createColliderFromMesh(allocator: std.mem.Allocator, mesh: *Mesh, shape: ColliderShape) !ColliderComponent {
    switch (shape) {
        .TriangleMesh => return ColliderComponent.init(allocator, .{ .TriangleMesh = TriangleMeshShape{} }, mesh),
        .ConvexHull => |hulls| {
            // If hulls are empty, generate them from the mesh
            if (hulls.len == 0) {
                const generated_hulls = try ConvexHullShape.generateFromMesh(allocator, mesh);
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
    bullet_body: ?bullet.CbtBodyHandle = null,
    bullet_shape: bullet.CbtShapeHandle,
    mass: f32 = 1.0,
    initial_position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    initial_rotation: [4]f32 = .{ 0.0, 0.0, 0.0, 1.0 }, // quaternion (x, y, z, w)

    pub fn init(mass: f32, shape: bullet.CbtShapeHandle) Self {
        return .{ .mass = mass, .bullet_shape = shape };
    }

    pub fn setInitialTransform(self: *Self, position: [3]f32, rotation: [4]f32) void {
        self.initial_position = position;
        self.initial_rotation = rotation;
    }

    pub fn attach(self: *Self, ecs: *ECSManager, eid: Core.EntityID) !void {
        self.entity_id = eid;

        // Create Bullet rigid body
        self.bullet_body = bullet.cbtBodyAllocate();
        if (self.bullet_body == null) return error.BodyAllocationFailed;

        // Get transform for initial position/rotation
        if (ecs.transform_components.get(eid)) |transform| {
            const mat4 = transform.local_transform;
            const right = mat4.get_right();
            const up = mat4.get_up();
            const forward = mat4.get_forward();
            const position = mat4.get_position();

            // Convert to Bullet format (4x3 matrix: 3 basis vectors + position)
            var transform_matrix = [4][3]f32{
                [3]f32{ right.x(), right.y(), right.z() }, // Right vector
                [3]f32{ up.x(), up.y(), up.z() }, // Up vector
                [3]f32{ forward.x(), forward.y(), forward.z() }, // Forward vector
                [3]f32{ position.x(), position.y(), position.z() }, // Position
            };

            // Create body with the provided shape
            bullet.cbtBodyCreate(self.bullet_body.?, self.mass, &transform_matrix, self.bullet_shape);

            // Set physics properties
            bullet.cbtBodySetDamping(self.bullet_body.?, 0.05, 0.05);

            // For dynamic bodies (mass > 0), disable automatic deactivation
            if (self.mass > 0.0) {
                bullet.cbtBodySetActivationState(self.bullet_body.?, bullet.CBT_DISABLE_DEACTIVATION);
                std.debug.print("Disabled deactivation for dynamic body eid {d}\n", .{eid.id});
            }

            // Add to physics world
            if (ecs.collision_system.bullet_world) |world| {
                bullet.cbtWorldAddBody(world, self.bullet_body.?);
            }
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

    // Debug visualization
    debug_entity_id: ?Core.EntityID = null,

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

    // For compound shapes from GLTF models
    pub fn initFromModel(
        allocator: std.mem.Allocator,
        model_resource: *GLTF.ModelResource,
        base_shape: ColliderShape,
        resource_manager: *ResourceManager,
    ) !Self {
        // Count meshes to allocate compound shape properly
        var mesh_count: u32 = 0;
        for (model_resource.entities) |node| {
            if (node.mesh_name != null) mesh_count += 1;
        }

        if (mesh_count == 0) return error.NoMeshesToCreateColliderFrom;

        // Create a compound shape collider
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

        // Add each mesh as a child shape with its relative transform
        for (model_resource.entities) |node| {
            if (node.mesh_name) |mesh_name| {
                if (resource_manager.meshes.get(mesh_name)) |*mesh_res| {
                    // Create child shape based on the base collider shape type
                    const child_shape = bullet.cbtShapeAllocate(switch (base_shape) {
                        .Box => bullet.CBT_SHAPE_TYPE_BOX,
                        .Sphere => bullet.CBT_SHAPE_TYPE_SPHERE,
                        .Capsule => bullet.CBT_SHAPE_TYPE_CAPSULE,
                        .Cylinder => bullet.CBT_SHAPE_TYPE_CYLINDER,
                        .Cone => bullet.CBT_SHAPE_TYPE_CONE,
                        .TriangleMesh => bullet.CBT_SHAPE_TYPE_TRIANGLE_MESH,
                        .ConvexHull => bullet.CBT_SHAPE_TYPE_COMPOUND,
                        .CompoundShape => bullet.CBT_SHAPE_TYPE_COMPOUND,
                    });

                    if (child_shape == null) continue;

                    // Create shape from mesh using existing createColliderFromMesh logic
                    const mesh_collider = try createColliderFromMesh(allocator, mesh_res.mesh, base_shape);

                    // Use the unified createBulletShape method that all shapes support
                    try mesh_collider.shape.createBulletShape(allocator, child_shape, mesh_res.mesh);

                    // Extract transform from GLTF node data
                    var child_transform: [4][3]f32 = undefined;

                    if (node.local_transformation) |local_transformation| {
                        // Use the full transformation matrix if available
                        const mat4 = local_transformation;
                        const right = mat4.get_right();
                        const up = mat4.get_up();
                        const forward = mat4.get_forward();
                        const position = mat4.get_position();

                        child_transform = [4][3]f32{
                            [3]f32{ right.x(), right.y(), right.z() }, // Right vector
                            [3]f32{ up.x(), up.y(), up.z() }, // Up vector
                            [3]f32{ forward.x(), forward.y(), forward.z() }, // Forward vector
                            [3]f32{ position.x(), position.y(), position.z() }, // Position
                        };
                    } else {
                        // Build transform from individual TRS components
                        var transform_matrix = Mat4.identity();

                        // Apply translation
                        if (node.translation) |t| {
                            transform_matrix = transform_matrix.translate(t[0], t[1], t[2]);
                        }

                        // Apply rotation
                        if (node.rotation) |r| {
                            const quat = Quaternion.init(r[0], r[1], r[2], r[3]);
                            transform_matrix = transform_matrix.multiply(Mat4.from_quaternion(quat));
                        }

                        // Apply scale
                        if (node.scale) |s| {
                            transform_matrix = transform_matrix.scale(s[0], s[1], s[2]);
                        }

                        // Extract vectors from the built matrix
                        const right = transform_matrix.get_right();
                        const up = transform_matrix.get_up();
                        const forward = transform_matrix.get_forward();
                        const position = transform_matrix.get_position();

                        child_transform = [4][3]f32{
                            [3]f32{ right.x(), right.y(), right.z() }, // Right vector
                            [3]f32{ up.x(), up.y(), up.z() }, // Up vector
                            [3]f32{ forward.x(), forward.y(), forward.z() }, // Forward vector
                            [3]f32{ position.x(), position.y(), position.z() }, // Position
                        };
                    }

                    // Add child shape to compound with its local transform
                    bullet.cbtShapeCompoundAddChild(
                        compound_collider.bullet_shape.?,
                        &child_transform,
                        child_shape,
                    );
                }
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

    // Bullet physics world
    bullet_world: ?bullet.CbtWorldHandle = null,

    pub fn init(
        allocator: std.mem.Allocator,
        world: *Core.World,
        transform_components: *SparseSet(TransformComponent),
        rigid_body_components: *SparseSet(RigidBodyComponent),
        collider_components: *SparseSet(ColliderComponent),
    ) !Self {
        var system = Self{
            .world = world,
            .allocator = allocator,
            .transform_components = transform_components,
            .rigid_body_components = rigid_body_components,
            .collider_components = collider_components,
        };

        try system.initBulletPhysics();

        return system;
    }

    pub fn deinit(self: *Self) void {
        // Clean up all bullet physics objects
        if (self.bullet_world) |world| {
            // First remove all bodies
            while (bullet.cbtWorldGetNumBodies(world) > 0) {
                const body = bullet.cbtWorldGetBody(world, 0);
                bullet.cbtWorldRemoveBody(world, body);
            }

            // Then destroy the world
            bullet.cbtWorldDestroy(world);
            self.bullet_world = null;
        }
    }

    fn initBulletPhysics(self: *Self) !void {
        // Initialize task scheduler for multi-threading (optional)
        bullet.cbtTaskSchedInit();

        self.bullet_world = bullet.cbtWorldCreate();
        if (self.bullet_world == null) return error.BulletInitFailed;

        var gravity = [3]f32{ 0.0, -9.81, 0.0 };
        bullet.cbtWorldSetGravity(self.bullet_world.?, &gravity);
    }

    pub fn update(self: *Self, dt: f32) !void {
        // Step the physics simulation TODO: Separate this system into own thread with a fixed time step
        if (self.bullet_world) |world| {
            _ = bullet.cbtWorldStepSimulation(world, dt, 10, 1.0 / 60.0);
        }

        var it = self.rigid_body_components.iterator();
        while (it.next()) |entry| {
            const entity_id = entry.entity_id;
            const rigid_body = entry.component;

            if (rigid_body.bullet_body) |body| {
                if (bullet.cbtBodyIsStaticOrKinematic(body)) continue;

                if (self.transform_components.get(entity_id)) |transform| {
                    // Get the world transform from Bullet
                    var transform_matrix: [4][3]f32 = undefined;
                    bullet.cbtBodyGetCenterOfMassTransform(body, &transform_matrix);

                    transform.position = .{
                        transform_matrix[3][0],
                        transform_matrix[3][1],
                        transform_matrix[3][2],
                    };

                    const basis = Mat4.from_array(
                        [16]f32{
                            transform_matrix[0][0], transform_matrix[0][1], transform_matrix[0][2], 0.0,
                            transform_matrix[1][0], transform_matrix[1][1], transform_matrix[1][2], 0.0,
                            transform_matrix[2][0], transform_matrix[2][1], transform_matrix[2][2], 0.0,
                            0.0,                    0.0,                    0.0,                    1.0,
                        },
                    );

                    const trs = basis.decomposeTRS();
                    transform.rotation = trs.rotation;

                    // Simple ground collision - prevent falling below Y = 0
                    // if (transform.position[1] < 0.0) {
                    //     transform.position[1] = 0.0;
                    //     // Reset vertical velocity in Bullet to stop bouncing
                    //     var velocity: [3]f32 = undefined;
                    //     bullet.cbtBodyGetLinearVelocity(body, &velocity);
                    //     velocity[1] = 0.0;
                    //     bullet.cbtBodySetLinearVelocity(body, &velocity);
                    // }

                    transform.updateLocalTransform();
                }
            }
        }
    }

    // Reset all dynamic bodies to their initial state
    pub fn resetAllDynamicBodies(self: *Self) void {
        std.debug.print("Starting resetAllDynamicBodies...\n", .{});
        var it = self.rigid_body_components.iterator();
        var reset_count: u32 = 0;
        
        while (it.next()) |entry| {
            const entity_id = entry.entity_id;
            const rigid_body = entry.component;
            
            std.debug.print("Checking entity {d}: mass={d:.2}, has_body={any}\n", .{ entity_id.id, rigid_body.mass, rigid_body.bullet_body != null });
            
            // Only reset dynamic bodies (mass > 0)
            if (rigid_body.mass > 0.0 and rigid_body.bullet_body != null) {
                const body = rigid_body.bullet_body.?;
                
                std.debug.print("Resetting entity {d} to initial position: [{d:.2}, {d:.2}, {d:.2}]\n", .{ entity_id.id, rigid_body.initial_position[0], rigid_body.initial_position[1], rigid_body.initial_position[2] });
                
                // Create identity rotation matrix with initial position
                var transform_matrix = [4][3]f32{
                    [3]f32{ 1.0, 0.0, 0.0 }, // Right vector (identity)
                    [3]f32{ 0.0, 1.0, 0.0 }, // Up vector (identity)
                    [3]f32{ 0.0, 0.0, 1.0 }, // Forward vector (identity)
                    [3]f32{ rigid_body.initial_position[0], rigid_body.initial_position[1], rigid_body.initial_position[2] }, // Position
                };

                // Reset physics body transform
                bullet.cbtBodySetCenterOfMassTransform(body, &transform_matrix);
                
                // Reset velocities
                const zero_vel = [3]f32{ 0.0, 0.0, 0.0 };
                bullet.cbtBodySetLinearVelocity(body, &zero_vel);
                bullet.cbtBodySetAngularVelocity(body, &zero_vel);
                
                // Force activation to ensure physics updates
                bullet.cbtBodySetActivationState(body, bullet.CBT_ACTIVE_TAG);
                
                // Also update the transform component to match
                if (self.transform_components.get(entity_id)) |transform| {
                    transform.setPosition(rigid_body.initial_position[0], rigid_body.initial_position[1], rigid_body.initial_position[2]);
                    // Reset rotation to identity quaternion
                    const identity_quat = Math.Quaternion.identity();
                    transform.setRotation(identity_quat);
                }
                
                reset_count += 1;
                std.debug.print("Successfully reset entity {d} to initial state\n", .{entity_id.id});
            }
        }
        
        std.debug.print("Reset complete! {d} dynamic bodies reset.\n", .{reset_count});
    }

    // Apply a central force to an entity
    pub fn applyCentralForce(self: *Self, entity_id: Core.EntityID, force: [3]f32) void {
        if (self.rigid_body_components.get(entity_id)) |rigid_body| {
            if (rigid_body.bullet_body) |body| {
                std.debug.print("Applying force [{d:.2}, {d:.2}, {d:.2}] to bullet body for eid {d}\n", .{ force[0], force[1], force[2], entity_id.id });

                bullet.cbtBodyApplyCentralForce(body, &force);

                // Debug: Check if body is static/kinematic
                if (bullet.cbtBodyIsStaticOrKinematic(body)) {
                    std.debug.print("WARNING: Body for eid {d} is static/kinematic!\n", .{entity_id.id});
                } else {
                    // Check if body is active and has reasonable mass
                    const mass = bullet.cbtBodyGetMass(body);
                    const is_active = bullet.cbtBodyIsActive(body);
                    std.debug.print("Body for eid {d}: mass={d:.2}, active={any}\n", .{ entity_id.id, mass, is_active });
                    
                    // Force activation if not active
                    if (!is_active) {
                        bullet.cbtBodySetActivationState(body, bullet.CBT_ACTIVE_TAG);
                        std.debug.print("Forced activation of body for eid {d}\n", .{entity_id.id});
                    }
                    
                    // Check current velocity after applying force
                    var velocity: [3]f32 = undefined;
                    bullet.cbtBodyGetLinearVelocity(body, &velocity);
                    std.debug.print("Body velocity after force: [{d:.2}, {d:.2}, {d:.2}]\n", .{ velocity[0], velocity[1], velocity[2] });
                }
            } else {
                std.debug.print("No bullet body found for eid {d}\n", .{entity_id.id});
            }
        } else {
            std.debug.print("No rigid body found for eid {d}\n", .{entity_id.id});
        }
    }

    // Apply a central impulse to an entity
    pub fn applyCentralImpulse(self: *Self, entity_id: Core.EntityID, impulse: [3]f32) void {
        if (self.rigid_body_components.get(entity_id)) |rigid_body| {
            if (rigid_body.bullet_body) |body| {
                bullet.cbtBodyApplyCentralImpulse(body, &impulse);
            }
        }
    }

    // Apply a torque to an entity
    pub fn applyTorque(self: *Self, entity_id: Core.EntityID, torque: [3]f32) void {
        if (self.rigid_body_components.get(entity_id)) |rigid_body| {
            if (rigid_body.bullet_body) |body| {
                bullet.cbtBodyApplyTorque(body, &torque);
            }
        }
    }

    // Apply collider properties to a rigid body (collision properties, inertia, etc.)
    pub fn linkColliderToRigidBody(self: *Self, entity_id: Core.EntityID) !void {
        const collider = self.collider_components.get(entity_id) orelse return error.NoCollider;
        const rigid_body = self.rigid_body_components.get(entity_id) orelse return error.NoRigidBody;

        if (rigid_body.bullet_body == null) return error.NoRigidBody;

        // Set collision properties from collider
        bullet.cbtBodySetRestitution(rigid_body.bullet_body.?, collider.restitution);
        bullet.cbtBodySetFriction(rigid_body.bullet_body.?, collider.friction);
        bullet.cbtBodySetRollingFriction(rigid_body.bullet_body.?, collider.rolling_friction);

        // Calculate and set inertia for dynamic bodies
        if (rigid_body.mass > 0.0) {
            var local_inertia = [3]f32{ 0.0, 0.0, 0.0 };
            bullet.cbtShapeCalculateLocalInertia(rigid_body.bullet_shape, rigid_body.mass, &local_inertia);
            bullet.cbtBodySetMassProps(rigid_body.bullet_body.?, rigid_body.mass, &local_inertia);
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

    // Create debug visualization for a collider
    pub fn createDebugVisualization(self: *Self, ecs_manager: *ECSManager, collider_eid: Core.EntityID, is_dynamic: bool) !void {
        if (self.collider_components.get(collider_eid)) |collider| {
            // Import ColliderDebug here to avoid circular dependency
            const ColliderDebug = @import("../prefabs/ColliderDebug.zig");

            // Generate deterministic name for debug mesh (will be cached by ResourceManager)
            var name_buf: [64]u8 = undefined;
            const mesh_name = try std.fmt.bufPrint(&name_buf, "collider_debug_{s}_{d}", .{
                @tagName(collider.shape),
                collider_eid.id,
            });

            // Get mesh data for triangle mesh debug visualization
            var mesh_data: ?*Mesh = null;
            if (std.meta.activeTag(collider.shape) == .TriangleMesh) {
                // For triangle meshes, get the original mesh data from the renderable component
                if (ecs_manager.renderer_components.get(collider_eid)) |renderable| {
                    if (self.world.resource_manager.meshes.get(renderable.mesh_name)) |mesh_resource| {
                        mesh_data = mesh_resource.mesh;
                    }
                }
            }

            // Create debug visualization using the new unified API
            const debug_components = try ColliderDebug.generateDebugVisualization(
                self.allocator,
                ecs_manager,
                mesh_name,
                collider.shape,
                collider,
                is_dynamic,
                mesh_data,
            );

            // Spawn debug entity
            const debug_eid = try ecs_manager.spawn(.{
                debug_components.tf,
                debug_components.renderable,
            });

            // Link to parent entity
            if (self.transform_components.get(collider_eid)) |_| {
                try ecs_manager.transform_system.addChild(collider_eid, debug_eid);
            }

            // Store reference
            collider.debug_entity_id = debug_eid;
        }
    }

    // Remove debug visualization for a collider
    pub fn removeDebugVisualization(self: *Self, ecs_manager: *ECSManager, collider_eid: Core.EntityID) !void {
        if (self.collider_components.get(collider_eid)) |collider| {
            if (collider.debug_entity_id) |debug_eid| {
                // Remove entity from ECS
                ecs_manager.destroyEntity(debug_eid);

                // Clear the reference
                collider.debug_entity_id = null;
            }
        }
    }

    // Toggle debug visualization for all colliders
    pub fn toggleDebugVisualization(self: *Self, ecs_manager: *ECSManager, show_debug: bool) !void {
        var it = self.collider_components.iterator();
        while (it.next()) |entry| {
            const collider_eid = entry.entity_id;
            const collider = entry.component;

            if (show_debug and collider.debug_entity_id == null) {
                // Determine if it's dynamic based on rigid body component
                var is_dynamic = false;
                if (self.rigid_body_components.get(collider_eid)) |rigid_body| {
                    is_dynamic = rigid_body.mass > 0.0; // Dynamic bodies have mass > 0
                }
                try self.createDebugVisualization(ecs_manager, collider_eid, is_dynamic);
            } else if (!show_debug and collider.debug_entity_id != null) {
                try self.removeDebugVisualization(ecs_manager, collider_eid);
            }
        }
    }
};
