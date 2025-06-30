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

pub const PhysicsType = enum {
    Static, // Collision only, no physics simulation
    Dynamic, // Full physics simulation
    None, // No collision or physics
};

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
        .TriangleMesh => return ColliderComponent.init(.{ .TriangleMesh = TriangleMeshShape{} }),
        .ConvexHull => |hulls| {
            // If hulls are empty, generate them from the mesh
            if (hulls.len == 0) {
                const generated_hulls = try ConvexHullShape.generateFromMesh(allocator, mesh);
                return ColliderComponent.init(.{ .ConvexHull = generated_hulls });
            } else {
                // Use provided hulls
                return ColliderComponent.init(.{ .ConvexHull = hulls });
            }
        },
        .Box => {
            const bounds = calculateMeshBounds(mesh);
            return ColliderComponent.init(.{ .Box = BoxShape{ .half_extents = bounds.half_extents } });
        },
        .Sphere => {
            const bounds = calculateMeshBounds(mesh);
            const radius = (bounds.half_extents[0] + bounds.half_extents[1] + bounds.half_extents[2]) / 3.0;
            return ColliderComponent.init(.{ .Sphere = SphereShape{ .radius = radius } });
        },
        .Capsule => {
            const bounds = calculateMeshBounds(mesh);
            const radius = (bounds.dimensions[0] + bounds.dimensions[2]) / 4.0;
            const height = bounds.dimensions[1];
            return ColliderComponent.init(.{ .Capsule = CapsuleShape{ .radius = radius, .height = height } });
        },
        .Cylinder => {
            const bounds = calculateMeshBounds(mesh);
            return ColliderComponent.init(.{ .Cylinder = CylinderShape{ .half_extents = bounds.half_extents } });
        },
        .Cone => {
            const bounds = calculateMeshBounds(mesh);
            const radius = @max(bounds.dimensions[0], bounds.dimensions[2]) / 2.0;
            const height = bounds.dimensions[1];
            return ColliderComponent.init(.{ .Cone = ConeShape{ .radius = radius, .height = height } });
        },
        .CompoundShape => {
            const bounds = calculateMeshBounds(mesh);
            return ColliderComponent.init(.{ .Box = BoxShape{ .half_extents = bounds.half_extents } });
        },
    }
}

pub const ColliderComponent = struct {
    const Self = @This();

    // Common properties
    entity_id: Core.EntityID = undefined,
    shape: ColliderShape,
    bullet_shape: ?bullet.CbtShapeHandle = null,
    bullet_body: ?bullet.CbtBodyHandle = null,

    // Collision properties
    friction: f32 = 0.5,
    rolling_friction: f32 = 0.1,
    restitution: f32 = 0.0,
    collision_group: u16 = 1,
    collision_mask: u16 = 0xFFFF, // Collide with everything by default

    // Debug visualization
    debug_entity_id: ?Core.EntityID = null,

    pub fn init(shape: ColliderShape) Self {
        return .{
            .shape = shape,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        if (self.bullet_body) |body| {
            bullet.cbtBodyDeallocate(body);
            self.bullet_body = null;
        }

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
    physics_components: *SparseSet(PhysicsComponent),
    collider_components: *SparseSet(ColliderComponent),

    // Bullet physics world
    bullet_world: ?bullet.CbtWorldHandle = null,

    pub fn init(
        allocator: std.mem.Allocator,
        world: *Core.World,
        transform_components: *SparseSet(TransformComponent),
        physics_components: *SparseSet(PhysicsComponent),
        collider_components: *SparseSet(ColliderComponent),
    ) !Self {
        var system = Self{
            .world = world,
            .allocator = allocator,
            .transform_components = transform_components,
            .physics_components = physics_components,
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

        // Create the dynamic world
        self.bullet_world = bullet.cbtWorldCreate();
        if (self.bullet_world == null) return error.BulletInitFailed;

        // Set default gravity
        var gravity = [3]f32{ 0.0, -9.81, 0.0 };
        bullet.cbtWorldSetGravity(self.bullet_world.?, &gravity);
    }

    pub fn update(self: *Self, dt: f32) !void {
        // Step the physics simulation
        if (self.bullet_world) |world| {
            _ = bullet.cbtWorldStepSimulation(world, dt, 10, 1.0 / 60.0);
        }

        // Update transforms of all entities based on the physics simulation
        var it = self.collider_components.iterator();
        while (it.next()) |entry| {
            const entity_id = entry.entity_id;
            const collider = entry.component;

            if (collider.bullet_body) |body| {
                // Skip if not a dynamic body

                // std.debug.print("Before Collider for {d}\n", .{entity_id.id});
                if (bullet.cbtBodyIsStaticOrKinematic(body)) continue;

                if (self.transform_components.get(entity_id)) |transform| {
                    // std.debug.print("After Collider for {d}\n", .{entity_id.id});
                    // Get the world transform from Bullet
                    var transform_matrix: [4][3]f32 = undefined;
                    bullet.cbtBodyGetCenterOfMassTransform(body, &transform_matrix);

                    // Update position from the first column of the transform matrix
                    transform.position = .{
                        transform_matrix[3][0],
                        transform_matrix[3][1],
                        transform_matrix[3][2],
                    };

                    // Extract rotation from the transform matrix
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
                    if (transform.position[1] < 0.0) {
                        transform.position[1] = 0.0;
                        // Reset vertical velocity in Bullet to stop bouncing
                        var velocity: [3]f32 = undefined;
                        bullet.cbtBodyGetLinearVelocity(body, &velocity);
                        velocity[1] = 0.0;
                        bullet.cbtBodySetLinearVelocity(body, &velocity);
                    }

                    transform.updateLocalTransform();
                }
            }
        }
    }

    pub fn createBulletShape(self: *Self, collider: *ColliderComponent, mesh: ?*Mesh) !void {
        if (collider.bullet_shape != null) return; // Already created

        // Allocate the shape using the unified shape type getter
        collider.bullet_shape = bullet.cbtShapeAllocate(collider.shape.getBulletShapeType());
        if (collider.bullet_shape == null) return error.ShapeAllocationFailed;

        // Create the shape using the unified shape creation method
        try collider.shape.createBulletShape(self.allocator, collider.bullet_shape.?, mesh);
    }

    // TODO: Refactor this function to use Math helpers to get basis vectors / positions and make sure behavior is expected
    pub fn createRigidBody(self: *Self, collider: *ColliderComponent, transform: *TransformComponent, physics: *PhysicsComponent) !void {
        if (collider.bullet_shape == null) {
            try self.createBulletShape(collider, null);
        }

        if (collider.bullet_body != null) return; // Already created

        // Allocate a rigid body
        collider.bullet_body = bullet.cbtBodyAllocate();
        if (collider.bullet_body == null) return error.BodyAllocationFailed;

        // Calculate local inertia
        var local_inertia = [3]f32{ 0.0, 0.0, 0.0 };
        if (physics.body_type == .Dynamic) {
            bullet.cbtShapeCalculateLocalInertia(collider.bullet_shape.?, physics.mass, &local_inertia);
        }

        // Prepare transform matrix from our TransformComponent
        var transform_matrix: [4][3]f32 = undefined;
        const mat4 = transform.world_transform;

        // Convert our Mat4 to the format expected by Bullet (column-major 4x3)
        transform_matrix[0][0] = mat4.base.data[0 * 4 + 0];
        transform_matrix[0][1] = mat4.base.data[0 * 4 + 1];
        transform_matrix[0][2] = mat4.base.data[0 * 4 + 2];

        transform_matrix[1][0] = mat4.base.data[1 * 4 + 0];
        transform_matrix[1][1] = mat4.base.data[1 * 4 + 1];
        transform_matrix[1][2] = mat4.base.data[1 * 4 + 2];

        transform_matrix[2][0] = mat4.base.data[2 * 4 + 0];
        transform_matrix[2][1] = mat4.base.data[2 * 4 + 1];
        transform_matrix[2][2] = mat4.base.data[2 * 4 + 2];

        transform_matrix[3][0] = mat4.base.data[3 * 4 + 0]; // X position
        transform_matrix[3][1] = mat4.base.data[3 * 4 + 1]; // Y position
        transform_matrix[3][2] = mat4.base.data[3 * 4 + 2]; // Z position

        // Create the rigid body with the shape and transform
        var mass = physics.mass;
        if (physics.body_type == .Static) mass = 0.0;

        bullet.cbtBodyCreate(collider.bullet_body.?, mass, &transform_matrix, collider.bullet_shape.?);

        // Set collision properties
        bullet.cbtBodySetRestitution(collider.bullet_body.?, collider.restitution);
        bullet.cbtBodySetFriction(collider.bullet_body.?, collider.friction);
        bullet.cbtBodySetRollingFriction(collider.bullet_body.?, collider.rolling_friction);

        // Set physics properties
        bullet.cbtBodySetDamping(collider.bullet_body.?, 0.05, 0.05); // Linear and angular damping

        // Handle kinematic bodies
        if (physics.body_type == .Kinematic) {
            bullet.cbtBodySetActivationState(collider.bullet_body.?, bullet.CBT_DISABLE_DEACTIVATION);
        }

        // Add the body to the world
        if (self.bullet_world) |world| {
            bullet.cbtWorldAddBody(world, collider.bullet_body.?);
        }
    }

    // Apply a central force to an entity
    pub fn applyCentralForce(self: *Self, entity_id: Core.EntityID, force: [3]f32) void {
        if (self.collider_components.get(entity_id)) |collider| {
            if (collider.bullet_body) |body| {
                bullet.cbtBodyApplyCentralForce(body, &force);
            }
        }
    }

    // Apply a central impulse to an entity
    pub fn applyCentralImpulse(self: *Self, entity_id: Core.EntityID, impulse: [3]f32) void {
        if (self.collider_components.get(entity_id)) |collider| {
            if (collider.bullet_body) |body| {
                bullet.cbtBodyApplyCentralImpulse(body, &impulse);
            }
        }
    }

    // Apply a torque to an entity
    pub fn applyTorque(self: *Self, entity_id: Core.EntityID, torque: [3]f32) void {
        if (self.collider_components.get(entity_id)) |collider| {
            if (collider.bullet_body) |body| {
                bullet.cbtBodyApplyTorque(body, &torque);
            }
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

    // Create collider for an entity (unified function for both mesh-based and non-mesh-based colliders)
    pub fn createCollider(self: *Self, eid: Core.EntityID, shape: ColliderShape, mesh_name: ?[]const u8) !void {
        var collider_component: ColliderComponent = undefined;
        var mesh_data: ?*Mesh = null;

        if (mesh_name) |name| {
            // Mesh-based collider - get the mesh resource and create collider from it
            if (self.world.resource_manager.meshes.get(name)) |*mesh_resource| {
                mesh_data = mesh_resource.mesh;
                collider_component = try createColliderFromMesh(self.allocator, mesh_resource.mesh, shape);
            } else {
                return error.MeshNotFound;
            }
        } else {
            // Non-mesh-based collider - use the shape directly
            collider_component = ColliderComponent.init(shape);
        }

        // Create the bullet shape
        try self.createBulletShape(&collider_component, mesh_data);

        // Configure physics based on the existing physics component
        if (self.transform_components.get(eid)) |transform| {
            if (self.physics_components.get(eid)) |physics| {
                // Physics type and mass are already set by the caller - just create the rigid body
                try self.createRigidBody(&collider_component, transform, physics);
            }
        }

        // Attach the collider to the entity
        collider_component.entity_id = eid;
        try self.collider_components.add(eid, collider_component);
    }

    // Create compound collider from hierarchical model (for GLTF models with multiple meshes)
    pub fn createCompoundColliderFromModel(
        self: *Self,
        root_eid: Core.EntityID,
        model_resource: *GLTF.ModelResource,
        entity_map: std.AutoHashMap(usize, Core.EntityID),
        base_collider_shape: ColliderShape,
    ) !void {
        // Count meshes to allocate compound shape properly
        var mesh_count: u32 = 0;
        for (model_resource.entities) |node| {
            if (node.mesh_name != null) mesh_count += 1;
        }

        if (mesh_count == 0) return; // No meshes to create colliders from

        // Create a compound shape collider
        var compound_collider = ColliderComponent.init(.{ .CompoundShape = CompoundShape{} });

        // Allocate compound shape
        compound_collider.bullet_shape = bullet.cbtShapeAllocate(bullet.CBT_SHAPE_TYPE_COMPOUND);
        if (compound_collider.bullet_shape == null) return error.ShapeAllocationFailed;

        bullet.cbtShapeCompoundCreate(
            compound_collider.bullet_shape.?,
            true,
            @intCast(mesh_count),
        );

        // Add each mesh as a child shape with its relative transform
        for (model_resource.entities, 0..) |node, idx| {
            if (node.mesh_name) |mesh_name| {
                if (self.world.resource_manager.meshes.get(mesh_name)) |*mesh_res| {
                    // Create child shape based on the base collider shape type
                    const child_shape = bullet.cbtShapeAllocate(switch (base_collider_shape) {
                        .TriangleMesh => bullet.CBT_SHAPE_TYPE_TRIANGLE_MESH,
                        .ConvexHull => bullet.CBT_SHAPE_TYPE_COMPOUND, // ConvexHull is also compound
                        else => bullet.CBT_SHAPE_TYPE_TRIANGLE_MESH, // Default to triangle mesh for complex shapes
                    });

                    if (child_shape == null) continue;

                    // Create the child shape using the base collider shape logic
                    switch (base_collider_shape) {
                        .TriangleMesh => {
                            const triangle_mesh_shape = TriangleMeshShape{};
                            try triangle_mesh_shape.createBulletShape(self.allocator, child_shape, mesh_res.mesh);
                        },
                        .ConvexHull => {
                            // Generate convex hulls for this specific mesh
                            const hulls = try ConvexHullShape.generateFromMesh(self.allocator, mesh_res.mesh);
                            defer {
                                for (hulls) |hull| {
                                    self.allocator.free(hull.points);
                                    self.allocator.free(hull.triangles);
                                }
                                self.allocator.free(hulls);
                            }

                            // Create compound shape from convex hulls
                            bullet.cbtShapeCompoundCreate(child_shape, true, @intCast(hulls.len));
                            for (hulls) |hull| {
                                const hull_child_shape = hull.createBulletShape(self.allocator);
                                if (hull_child_shape != null) {
                                    // Add with identity transform
                                    var identity_transform = [4][3]f32{
                                        [3]f32{ 1.0, 0.0, 0.0 },
                                        [3]f32{ 0.0, 1.0, 0.0 },
                                        [3]f32{ 0.0, 0.0, 1.0 },
                                        [3]f32{ 0.0, 0.0, 0.0 },
                                    };
                                    bullet.cbtShapeCompoundAddChild(
                                        child_shape,
                                        &identity_transform,
                                        hull_child_shape.?,
                                    );
                                }
                            }
                        },
                        else => {
                            // For other shapes, default to triangle mesh
                            const triangle_mesh_shape = TriangleMeshShape{};
                            try triangle_mesh_shape.createBulletShape(self.allocator, child_shape, mesh_res.mesh);
                        },
                    }

                    // Get the entity's transform relative to root using Math library helpers
                    if (entity_map.get(idx)) |entity_id| {
                        if (self.transform_components.get(entity_id)) |transform| {
                            // Use Math library helper functions to extract basis vectors
                            const mat4 = transform.local_transform;
                            const right = mat4.get_right();
                            const up = mat4.get_up();
                            const forward = mat4.get_forward();
                            const position = mat4.get_position();

                            // Convert to Bullet format (4x3 matrix: 3 basis vectors + position)
                            var child_transform = [4][3]f32{
                                [3]f32{ right.x(), right.y(), right.z() }, // Right vector
                                [3]f32{ up.x(), up.y(), up.z() }, // Up vector
                                [3]f32{ forward.x(), forward.y(), forward.z() }, // Forward vector
                                [3]f32{ position.x(), position.y(), position.z() }, // Position
                            };

                            // Add child shape to compound with its local transform
                            bullet.cbtShapeCompoundAddChild(
                                compound_collider.bullet_shape.?,
                                &child_transform,
                                child_shape,
                            );
                        }
                    }
                }
            }
        }

        // Create physics body
        if (self.transform_components.get(root_eid)) |transform| {
            if (self.physics_components.get(root_eid)) |physics| {
                try self.createRigidBody(&compound_collider, transform, physics);
            }
        }

        // Attach the compound collider to the root entity
        compound_collider.entity_id = root_eid;
        try self.collider_components.add(root_eid, compound_collider);
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
                // Determine if it's dynamic based on physics component
                var is_dynamic = false;
                if (self.physics_components.get(collider_eid)) |physics| {
                    is_dynamic = physics.body_type == .Dynamic;
                }
                try self.createDebugVisualization(ecs_manager, collider_eid, is_dynamic);
            } else if (!show_debug and collider.debug_entity_id != null) {
                try self.removeDebugVisualization(ecs_manager, collider_eid);
            }
        }
    }
};
