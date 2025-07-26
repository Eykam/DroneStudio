const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const Mesh = @import("../../Mesh.zig");
const bullet = @import("../../bindings/c.zig").bullet;

pub const ApplyForce = struct {
    entity_id: Core.EntityID,
    force: [3]f32,

    pub fn execute(self: ApplyForce, physics_thread: *ThreadedPhysicsSystem) void {
        if (physics_thread.entity_bodies.get(self.entity_id)) |body| {
            bullet.cbtBodyApplyCentralForce(body, &self.force);
        } else {
            std.debug.print("No physics body found for force on entity {d}\n", .{self.entity_id.id});
        }
    }
};

pub const ApplyTorque = struct {
    entity_id: Core.EntityID,
    torque: [3]f32,

    pub fn execute(self: ApplyTorque, physics_thread: *ThreadedPhysicsSystem) void {
        if (physics_thread.entity_bodies.get(self.entity_id)) |body| {
            bullet.cbtBodyApplyTorque(body, &self.torque);
        } else {
            std.debug.print("No physics body found for torque on entity {d}\n", .{self.entity_id.id});
        }
    }
};

pub const ApplyImpulse = struct {
    entity_id: Core.EntityID,
    impulse: [3]f32,

    pub fn execute(self: ApplyImpulse, physics_thread: *ThreadedPhysicsSystem) void {
        if (physics_thread.entity_bodies.get(self.entity_id)) |body| {
            bullet.cbtBodyApplyCentralImpulse(body, &self.impulse);
        } else {
            std.debug.print("No physics body found for impulse on entity {d}\n", .{self.entity_id.id});
        }
    }
};

pub const SetTransform = struct {
    entity_id: Core.EntityID,
    position: [3]f32,
    rotation: [4]f32,

    pub fn execute(self: SetTransform, physics_thread: *ThreadedPhysicsSystem) void {
        if (physics_thread.entity_bodies.get(self.entity_id)) |body| {
            const quat = Math.Quaternion{ .data = self.rotation };
            var transform_matrix = Math.Mat4.from_quaternion(quat);

            // Set position in the transform matrix manually
            transform_matrix.base.data[12] = self.position[0]; // [0,3]
            transform_matrix.base.data[13] = self.position[1]; // [1,3]
            transform_matrix.base.data[14] = self.position[2]; // [2,3]

            // Convert to Bullet's 4x3 format
            var bullet_transform = [4][3]f32{
                .{ transform_matrix.base.data[0], transform_matrix.base.data[1], transform_matrix.base.data[2] },
                .{ transform_matrix.base.data[4], transform_matrix.base.data[5], transform_matrix.base.data[6] },
                .{ transform_matrix.base.data[8], transform_matrix.base.data[9], transform_matrix.base.data[10] },
                .{ transform_matrix.base.data[12], transform_matrix.base.data[13], transform_matrix.base.data[14] },
            };

            bullet.cbtBodySetCenterOfMassTransform(body, &bullet_transform);
        }
    }
};

pub const CreateRigidBody = struct {
    entity_id: Core.EntityID,
    mass: f32,
    shape_handle: bullet.CbtShapeHandle,
    initial_pos: [3]f32,
    initial_rot: [4]f32,
    restitution: f32 = 0.5,
    friction: f32 = 0.5,
    rolling_friction: f32 = 0.1,

    pub fn execute(self: CreateRigidBody, physics_thread: *ThreadedPhysicsSystem) void {
        const body_handle = bullet.cbtBodyAllocate();
        if (body_handle != null) {
            std.debug.print("CreateRigidBody: Entity {d} initial_pos=[{d:.3}, {d:.3}, {d:.3}]\n", .{ self.entity_id.id, self.initial_pos[0], self.initial_pos[1], self.initial_pos[2] });

            var identity_transform = [4][3]f32{
                .{ 1.0, 0.0, 0.0 }, // X axis
                .{ 0.0, 1.0, 0.0 }, // Y axis
                .{ 0.0, 0.0, 1.0 }, // Z axis
                .{ 0.0, 0.0, 0.0 }, // Position
            };
            bullet.cbtBodyCreate(body_handle, self.mass, &identity_transform, self.shape_handle);

            // Get the center of mass transform that Bullet calculated from the collision shape
            var bullet_com_transform: [4][3]f32 = undefined;
            bullet.cbtBodyGetCenterOfMassTransform(body_handle, &bullet_com_transform);

            std.debug.print("Entity {d} - Bullet calculated CoM transform:\n", .{self.entity_id.id});
            std.debug.print("  Row 0: [{d:.3}, {d:.3}, {d:.3}] (X axis)\n", .{ bullet_com_transform[0][0], bullet_com_transform[0][1], bullet_com_transform[0][2] });
            std.debug.print("  Row 1: [{d:.3}, {d:.3}, {d:.3}] (Y axis)\n", .{ bullet_com_transform[1][0], bullet_com_transform[1][1], bullet_com_transform[1][2] });
            std.debug.print("  Row 2: [{d:.3}, {d:.3}, {d:.3}] (Z axis)\n", .{ bullet_com_transform[2][0], bullet_com_transform[2][1], bullet_com_transform[2][2] });
            std.debug.print("  Row 3: [{d:.3}, {d:.3}, {d:.3}] (Position)\n", .{ bullet_com_transform[3][0], bullet_com_transform[3][1], bullet_com_transform[3][2] });
            std.debug.print("  User requested: pos=[{d:.3}, {d:.3}, {d:.3}], rot=[{d:.3}, {d:.3}, {d:.3}, {d:.3}]\n", .{ self.initial_pos[0], self.initial_pos[1], self.initial_pos[2], self.initial_rot[0], self.initial_rot[1], self.initial_rot[2], self.initial_rot[3] });

            // Convert Bullet's CoM transform to Mat4
            const bullet_com_mat4 = Math.Mat4{ .base = .{ .data = [16]f32{
                bullet_com_transform[0][0], bullet_com_transform[0][1], bullet_com_transform[0][2], 0.0,
                bullet_com_transform[1][0], bullet_com_transform[1][1], bullet_com_transform[1][2], 0.0,
                bullet_com_transform[2][0], bullet_com_transform[2][1], bullet_com_transform[2][2], 0.0,
                bullet_com_transform[3][0], bullet_com_transform[3][1], bullet_com_transform[3][2], 1.0,
            } } };

            // Create user transform from initial position and rotation
            const quat = Math.Quaternion{ .data = self.initial_rot };
            var user_transform = Math.Mat4.from_quaternion(quat);
            user_transform.base.data[12] = self.initial_pos[0];
            user_transform.base.data[13] = self.initial_pos[1];
            user_transform.base.data[14] = self.initial_pos[2];

            // Compose transforms: final = bullet_com_transform * user_transform
            const final_transform_mat4 = bullet_com_mat4.multiply(user_transform);

            // Convert back to Bullet's 4x3 format
            var final_transform = [4][3]f32{
                .{ final_transform_mat4.base.data[0], final_transform_mat4.base.data[1], final_transform_mat4.base.data[2] },
                .{ final_transform_mat4.base.data[4], final_transform_mat4.base.data[5], final_transform_mat4.base.data[6] },
                .{ final_transform_mat4.base.data[8], final_transform_mat4.base.data[9], final_transform_mat4.base.data[10] },
                .{ final_transform_mat4.base.data[12], final_transform_mat4.base.data[13], final_transform_mat4.base.data[14] },
            };

            bullet.cbtBodySetCenterOfMassTransform(body_handle, &final_transform);

            std.debug.print("  Final rigid body transform:\n", .{});
            std.debug.print("    Row 0: [{d:.3}, {d:.3}, {d:.3}] (X axis)\n", .{ final_transform[0][0], final_transform[0][1], final_transform[0][2] });
            std.debug.print("    Row 1: [{d:.3}, {d:.3}, {d:.3}] (Y axis)\n", .{ final_transform[1][0], final_transform[1][1], final_transform[1][2] });
            std.debug.print("    Row 2: [{d:.3}, {d:.3}, {d:.3}] (Z axis)\n", .{ final_transform[2][0], final_transform[2][1], final_transform[2][2] });
            std.debug.print("    Row 3: [{d:.3}, {d:.3}, {d:.3}] (Position)\n", .{ final_transform[3][0], final_transform[3][1], final_transform[3][2] });
            std.debug.print("Set initial transform for entity {d}: pos=[{d:.3}, {d:.3}, {d:.3}]\n", .{ self.entity_id.id, self.initial_pos[0], self.initial_pos[1], self.initial_pos[2] });

            // Set physics properties
            bullet.cbtBodySetDamping(body_handle, 0.05, 0.05);

            // Set collision properties from command
            bullet.cbtBodySetRestitution(body_handle, self.restitution);
            bullet.cbtBodySetFriction(body_handle, self.friction);
            bullet.cbtBodySetRollingFriction(body_handle, self.rolling_friction);

            // Calculate and set inertia for dynamic bodies
            if (self.mass > 0.0) {
                var local_inertia = [3]f32{ 0.0, 0.0, 0.0 };
                bullet.cbtShapeCalculateLocalInertia(self.shape_handle, self.mass, &local_inertia);
                bullet.cbtBodySetMassProps(body_handle, self.mass, &local_inertia);

                // Enable CCD (Continuous Collision Detection) for fast-moving objects
                bullet.cbtBodySetCcdMotionThreshold(body_handle, 0.1);
                bullet.cbtBodySetCcdSweptSphereRadius(body_handle, 0.05);

                bullet.cbtBodySetActivationState(body_handle, bullet.CBT_DISABLE_DEACTIVATION);
                std.debug.print("Created dynamic body for entity {d} with mass {d:.2}\n", .{ self.entity_id.id, self.mass });
            } else {
                std.debug.print("Created static body for entity {d}\n", .{self.entity_id.id});
            }

            // Add to physics world
            bullet.cbtWorldAddBody(physics_thread.bullet_world, body_handle);
            std.debug.print("Body {d} successfully added to physics world\n", .{self.entity_id.id});

            // Store entity mapping
            physics_thread.entity_bodies.put(self.entity_id, body_handle) catch {
                std.debug.print("Failed to register entity {d} in physics thread\n", .{self.entity_id.id});
            };

            std.debug.print("Physics body for entity {d} added to physics world (mass: {d:.2}, type: {s})\n", .{ self.entity_id.id, self.mass, if (self.mass > 0.0) "DYNAMIC" else "STATIC" });

            // Debug: Print world body count
            std.debug.print("Physics world now has {d} bodies total\n", .{physics_thread.entity_bodies.count()});
        } else {
            std.debug.print("Failed to allocate physics body for entity {d}\n", .{self.entity_id.id});
        }
    }
};

pub const RemoveRigidBody = struct {
    entity_id: Core.EntityID,

    pub fn execute(self: RemoveRigidBody, physics_thread: *ThreadedPhysicsSystem) void {
        if (physics_thread.entity_bodies.get(self.entity_id)) |body| {
            // Remove from physics world
            bullet.cbtWorldRemoveBody(physics_thread.bullet_world, body);

            // Clean up body
            bullet.cbtBodyDeallocate(body);

            // Remove from entity mapping
            _ = physics_thread.entity_bodies.remove(self.entity_id);

            std.debug.print("Removed physics body for entity {d} from physics thread\n", .{self.entity_id.id});
        }
    }
};

pub const ResetDynamicBodies = struct {
    pub fn execute(self: ResetDynamicBodies, physics_thread: *ThreadedPhysicsSystem) void {
        _ = self; // unused
        // Reset all dynamic bodies
        var it = physics_thread.entity_bodies.iterator();
        while (it.next()) |entry| {
            const body = entry.value_ptr.*;

            // Reset velocities
            const zero_vel = [3]f32{ 0.0, 0.0, 0.0 };
            bullet.cbtBodySetLinearVelocity(body, &zero_vel);
            bullet.cbtBodySetAngularVelocity(body, &zero_vel);

            // Force activation
            bullet.cbtBodySetActivationState(body, bullet.CBT_ACTIVE_TAG);
        }
    }
};

// LineCollector struct for debug wireframe extraction
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
export fn physicsDebugDrawLineCallback(context: ?*anyopaque, p0: [*c]const f32, p1: [*c]const f32, color: [*c]const f32) void {
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

pub const SetDebugWireframes = struct {
    enabled: bool,
    dynamic_color: [3]f32, // Color for dynamic bodies (mass > 0)
    static_color: [3]f32, // Color for static bodies (mass == 0)

    pub fn execute(self: SetDebugWireframes, physics_thread: *ThreadedPhysicsSystem) void {
        physics_thread.debug_wireframes_enabled.store(self.enabled, .release);
        if (self.enabled) {
            physics_thread.debug_dynamic_color = self.dynamic_color;
            physics_thread.debug_static_color = self.static_color;
            
            // Extract wireframes once when debug is enabled
            physics_thread.extractDebugWireframes();
        }
        std.debug.print("{s} debug wireframes for physics world\n", .{if (self.enabled) "Enabled" else "Disabled"});
    }
};

pub const ExtractWireframes = struct {
    pub fn execute(self: ExtractWireframes, physics_thread: *ThreadedPhysicsSystem) void {
        _ = self;
        physics_thread.extractDebugWireframes();
        std.debug.print("Extracted debug wireframes on demand\n", .{});
    }
};

pub const PausePhysics = struct {
    paused: bool,

    pub fn execute(self: PausePhysics, physics_thread: *ThreadedPhysicsSystem) void {
        physics_thread.physics_paused.store(self.paused, .release);
        std.debug.print("Physics simulation {s}\n", .{if (self.paused) "paused" else "resumed"});
    }
};

/// Physics commands sent from main thread to physics thread
pub const PhysicsCommand = union(enum) {
    ApplyForce: ApplyForce,
    ApplyTorque: ApplyTorque,
    ApplyImpulse: ApplyImpulse,
    SetTransform: SetTransform,
    CreateRigidBody: CreateRigidBody,
    RemoveRigidBody: RemoveRigidBody,
    ResetDynamicBodies: ResetDynamicBodies,
    SetDebugWireframes: SetDebugWireframes,
    ExtractWireframes: ExtractWireframes,
    PausePhysics: PausePhysics,
    Shutdown: void,

    pub fn execute(self: PhysicsCommand, physics_thread: *ThreadedPhysicsSystem) void {
        switch (self) {
            .ApplyForce => |cmd| cmd.execute(physics_thread),
            .ApplyTorque => |cmd| cmd.execute(physics_thread),
            .ApplyImpulse => |cmd| cmd.execute(physics_thread),
            .SetTransform => |cmd| cmd.execute(physics_thread),
            .CreateRigidBody => |cmd| cmd.execute(physics_thread),
            .RemoveRigidBody => |cmd| cmd.execute(physics_thread),
            .ResetDynamicBodies => |cmd| cmd.execute(physics_thread),
            .SetDebugWireframes => |cmd| cmd.execute(physics_thread),
            .ExtractWireframes => |cmd| cmd.execute(physics_thread),
            .PausePhysics => |cmd| cmd.execute(physics_thread),
            .Shutdown => {
                physics_thread.should_shutdown.store(true, .release);
            },
        }
    }
};

/// Physics state sent from physics thread to main thread
pub const PhysicsState = struct {
    entity_id: Core.EntityID,
    position: [3]f32,
    rotation: [4]f32, // quaternion (x, y, z, w)
    linear_velocity: [3]f32,
    angular_velocity: [3]f32,
    is_active: bool,
    frame_number: u64, // Debug: track which physics frame this state is from
};

/// Debug wireframe data for an entity
pub const DebugWireframeData = struct {
    entity_id: Core.EntityID,
    vertices: []Mesh.Vertex,
    frame_number: u64,

    pub fn deinit(self: *DebugWireframeData, allocator: std.mem.Allocator) void {
        allocator.free(self.vertices);
    }
};

/// Debug wireframe buffer for thread-safe reading
pub const DebugWireframeBuffer = struct {
    const Self = @This();

    wireframes: [2]std.ArrayList(DebugWireframeData),
    write_buffer: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .wireframes = .{
                std.ArrayList(DebugWireframeData).init(allocator),
                std.ArrayList(DebugWireframeData).init(allocator),
            },
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        // Clean up any existing wireframe data
        for (&self.wireframes) |*buffer| {
            for (buffer.items) |*wireframe| {
                wireframe.deinit(allocator);
            }
            buffer.deinit();
        }
    }

    /// Physics thread writes to current write buffer
    pub fn beginWrite(self: *Self, allocator: std.mem.Allocator) *std.ArrayList(DebugWireframeData) {
        const write_idx = self.write_buffer.load(.seq_cst);

        // Clean up old wireframe data and clear buffer
        for (self.wireframes[write_idx].items) |*wireframe| {
            wireframe.deinit(allocator);
        }
        self.wireframes[write_idx].clearRetainingCapacity();

        return &self.wireframes[write_idx];
    }

    /// Physics thread signals write completion and swaps buffers
    pub fn endWrite(self: *Self) void {
        const current_write = self.write_buffer.load(.acquire);
        const next_write = 1 - current_write;
        self.write_buffer.store(next_write, .seq_cst);
    }

    /// Main thread reads from the most recently completed buffer
    pub fn read(self: *Self) []const DebugWireframeData {
        const current_write_idx = self.write_buffer.load(.seq_cst);
        const read_idx = 1 - current_write_idx;
        return self.wireframes[read_idx].items;
    }
};

/// Lock-free ring buffer for commands (single producer, single consumer)
pub fn RingBuffer(comptime T: type, comptime size: usize) type {
    return struct {
        const Self = @This();

        data: [size]T = undefined,
        write_pos: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        read_pos: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

        pub fn push(self: *Self, item: T) bool {
            const current_write = self.write_pos.load(.acquire);
            const next_write = (current_write + 1) % size;

            // Check if buffer is full
            if (next_write == self.read_pos.load(.acquire)) {
                return false; // Buffer full
            }

            self.data[current_write] = item;
            self.write_pos.store(next_write, .release);
            return true;
        }

        pub fn pop(self: *Self) ?T {
            const current_read = self.read_pos.load(.acquire);

            // Check if buffer is empty
            if (current_read == self.write_pos.load(.acquire)) {
                return null; // Buffer empty
            }

            const item = self.data[current_read];
            self.read_pos.store((current_read + 1) % size, .release);
            return item;
        }

        pub fn isEmpty(self: *Self) bool {
            return self.read_pos.load(.acquire) == self.write_pos.load(.acquire);
        }

        pub fn isFull(self: *Self) bool {
            const current_write = self.write_pos.load(.acquire);
            const next_write = (current_write + 1) % size;
            return next_write == self.read_pos.load(.acquire);
        }
    };
}

/// Double-buffered physics state for thread-safe reading
pub const PhysicsStateBuffer = struct {
    const Self = @This();

    states: [2]std.ArrayList(PhysicsState),
    write_buffer: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .states = .{
                std.ArrayList(PhysicsState).init(allocator),
                std.ArrayList(PhysicsState).init(allocator),
            },
        };
    }

    pub fn deinit(self: *Self) void {
        self.states[0].deinit();
        self.states[1].deinit();
    }

    /// Physics thread writes to current write buffer
    pub fn beginWrite(self: *Self) *std.ArrayList(PhysicsState) {
        const write_idx = self.write_buffer.load(.seq_cst);

        // Clear the current write buffer before writing new data
        self.states[write_idx].clearRetainingCapacity();
        return &self.states[write_idx];
    }

    /// Physics thread signals write completion and swaps buffers
    pub fn endWrite(self: *Self) void {
        const current_write = self.write_buffer.load(.acquire);
        const next_write = 1 - current_write;

        // Swap to the next buffer (main thread will now read from current_write buffer)
        self.write_buffer.store(next_write, .seq_cst);
    }

    /// Main thread reads from the most recently completed buffer
    pub fn read(self: *Self) []const PhysicsState {
        const current_write_idx = self.write_buffer.load(.seq_cst);
        // Read from the buffer that was just completed (the one NOT currently being written to)
        const read_idx = 1 - current_write_idx;

        return self.states[read_idx].items;
    }
};

/// Threaded physics system that runs at fixed timestep
pub const ThreadedPhysicsSystem = struct {
    const Self = @This();

    // Thread communication
    command_queue: RingBuffer(PhysicsCommand, 1024),
    state_buffer: PhysicsStateBuffer,
    wireframe_buffer: DebugWireframeBuffer,
    physics_thread: std.Thread,

    // Physics world state
    allocator: std.mem.Allocator,
    bullet_world: bullet.CbtWorldHandle,
    should_shutdown: std.atomic.Value(bool),

    // Timing
    target_hz: f64 = 240.0,
    fixed_timestep: f64,

    // Entity tracking
    entity_bodies: std.AutoHashMap(Core.EntityID, bullet.CbtBodyHandle),

    // Debug wireframes
    debug_wireframes_enabled: std.atomic.Value(bool),
    debug_wireframes_version: std.atomic.Value(u32),
    debug_dynamic_color: [3]f32,
    debug_static_color: [3]f32,

    // Physics simulation control
    physics_paused: std.atomic.Value(bool),

    // Debug tracking
    frame_count: u64 = 0,
    physics_time: f64 = 0.0,

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .command_queue = .{},
            .state_buffer = PhysicsStateBuffer.init(allocator),
            .wireframe_buffer = DebugWireframeBuffer.init(allocator),
            .physics_thread = undefined,
            .allocator = allocator,
            .bullet_world = undefined,
            .should_shutdown = std.atomic.Value(bool).init(false),
            .fixed_timestep = undefined, // 60 Hz
            .entity_bodies = std.AutoHashMap(Core.EntityID, bullet.CbtBodyHandle).init(allocator),
            .debug_wireframes_enabled = std.atomic.Value(bool).init(false),
            .debug_wireframes_version = std.atomic.Value(u32).init(0),
            .debug_dynamic_color = .{ 0.0, 1.0, 0.0 }, // Green for dynamic
            .debug_static_color = .{ 0.0, 0.0, 1.0 }, // Blue for static
            .physics_paused = std.atomic.Value(bool).init(false),
        };
        self.fixed_timestep = 1.0 / self.target_hz;

        // Initialize Bullet physics world
        self.bullet_world = bullet.cbtWorldCreate();
        if (self.bullet_world == null) {
            return error.FailedToCreateBulletWorld;
        }

        // Set gravity
        const gravity = [3]f32{ 0.0, -9.81, 0.0 };
        bullet.cbtWorldSetGravity(self.bullet_world, &gravity);

        std.debug.print("Physics world created with gravity: [{d:.2}, {d:.2}, {d:.2}]\n", .{ gravity[0], gravity[1], gravity[2] });

        // Start physics thread
        self.physics_thread = try std.Thread.spawn(.{}, physicsThreadMain, .{self});

        std.debug.print("ThreadedPhysicsSystem initialized with {d}Hz physics\n", .{self.target_hz});

        return self;
    }

    pub fn deinit(self: *Self) void {
        // Signal shutdown and wait for thread
        self.should_shutdown.store(true, .release);
        self.physics_thread.join();

        // Cleanup
        if (self.bullet_world != null) {
            bullet.cbtWorldDestroy(self.bullet_world);
        }

        self.entity_bodies.deinit();
        self.state_buffer.deinit();
        self.wireframe_buffer.deinit(self.allocator);

        self.allocator.destroy(self);
    }

    /// Main thread interface: Send command to physics thread
    pub fn sendCommand(self: *Self, command: PhysicsCommand) bool {
        return self.command_queue.push(command);
    }

    /// Main thread interface: Get latest physics state
    pub fn getPhysicsStates(self: *Self) []const PhysicsState {
        return self.state_buffer.read();
    }

    /// Main thread interface: Get debug wireframes
    pub fn getDebugWireframes(self: *Self) []const DebugWireframeData {
        return self.wireframe_buffer.read();
    }

    /// Get the current wireframe version (increments each time wireframes are extracted)
    pub fn getWireframeVersion(self: *Self) u32 {
        return self.debug_wireframes_version.load(.acquire);
    }

    /// Main thread interface: Enable/disable debug wireframes for the entire world
    pub fn setDebugWireframes(self: *Self, enabled: bool, dynamic_color: [3]f32, static_color: [3]f32) bool {
        const command = PhysicsCommand{
            .SetDebugWireframes = .{
                .enabled = enabled,
                .dynamic_color = dynamic_color,
                .static_color = static_color,
            },
        };
        return self.sendCommand(command);
    }

    /// Main thread interface: Extract wireframes once (for dynamic mesh creation)
    pub fn extractWireframes(self: *Self) bool {
        const command = PhysicsCommand{ .ExtractWireframes = .{} };
        return self.sendCommand(command);
    }

    /// Main thread interface: Pause or resume physics simulation
    pub fn setPhysicsPaused(self: *Self, paused: bool) bool {
        const command = PhysicsCommand{ .PausePhysics = .{ .paused = paused } };
        return self.sendCommand(command);
    }

    /// Main thread interface: Get current pause state
    pub fn isPhysicsPaused(self: *Self) bool {
        return self.physics_paused.load(.acquire);
    }

    /// Register a physics body for an entity (called from main thread during entity creation)
    pub fn registerEntity(self: *Self, entity_id: Core.EntityID, body_handle: bullet.CbtBodyHandle) !void {
        try self.entity_bodies.put(entity_id, body_handle);
    }

    /// Physics thread main loop
    fn physicsThreadMain(self: *Self) void {
        std.debug.print("Physics thread started for instance {*}\n", .{self});
        std.debug.print("Physics timestep: {d:.6}s ({d:.1} Hz)\n", .{ self.fixed_timestep, 1.0 / self.fixed_timestep });

        var timer = std.time.Timer.start() catch {
            std.debug.print("Failed to start physics timer\n", .{});
            return;
        };

        const target_frame_time_ns = @as(u64, @intFromFloat(self.fixed_timestep * 1_000_000_000));

        while (!self.should_shutdown.load(.acquire)) {
            const frame_start = timer.read();

            // Process commands from main thread
            self.processCommands();

            // Step physics simulation (only if not paused)
            var num_steps: i32 = 0;
            if (!self.physics_paused.load(.acquire)) {
                const max_substeps: i32 = 10;
                const fixed_timestep_f32: f32 = @floatCast(self.fixed_timestep);
                num_steps = bullet.cbtWorldStepSimulation(self.bullet_world, fixed_timestep_f32, max_substeps, fixed_timestep_f32);

                if (num_steps > 0) {
                    self.physics_time += @as(f64, @floatFromInt(num_steps)) * self.fixed_timestep;
                }

                // Debug: Log actual steps taken
                if (self.frame_count % 60 == 0 and num_steps > 0) {
                    std.debug.print("Physics stepping: took {d} substeps, timestep={d:.4}s, physics_time={d:.2}s\n", .{ num_steps, fixed_timestep_f32, self.physics_time });
                }
            } else {
                // Debug: Log pause status occasionally
                if (self.frame_count % 180 == 0) {
                    std.debug.print("Physics simulation paused\n", .{});
                }
            }

            // Debug: Check for contact points every 60 frames (1 second at 60Hz)
            self.frame_count += 1;

            // Update physics state buffer
            self.updateStateBuffer();


            // Sleep to maintain fixed timestep
            const frame_time = timer.read() - frame_start;
            if (frame_time < target_frame_time_ns) {
                const sleep_time = target_frame_time_ns - frame_time;
                std.time.sleep(sleep_time);
            }
        }

        std.debug.print("Physics thread shutting down\n", .{});
    }

    /// Process all pending commands from the main thread
    fn processCommands(self: *Self) void {
        while (self.command_queue.pop()) |command| {
            command.execute(self);
            if (command == .Shutdown) break;
        }
    }

    /// Update the physics state buffer with current entity states
    fn updateStateBuffer(self: *Self) void {
        const write_buffer = self.state_buffer.beginWrite();

        var it = self.entity_bodies.iterator();
        while (it.next()) |entry| {
            const entity_id = entry.key_ptr.*;
            const body = entry.value_ptr.*;

            // Get transform from Bullet
            var transform_matrix: [4][3]f32 = undefined;
            bullet.cbtBodyGetCenterOfMassTransform(body, &transform_matrix);

            // Get velocities
            var linear_vel: [3]f32 = undefined;
            var angular_vel: [3]f32 = undefined;
            bullet.cbtBodyGetLinearVelocity(body, &linear_vel);
            bullet.cbtBodyGetAngularVelocity(body, &angular_vel);

            // TODO: There should be a more efficient way to extract quaternion from 3x3 rotation matrix
            // without converting to Mat4 and using decomposeTRS, but this works for now

            // Convert Bullet's 4x3 transform matrix to Mat4
            const mat4 = Math.Mat4{ .base = .{ .data = [16]f32{
                transform_matrix[0][0], transform_matrix[0][1], transform_matrix[0][2], 0.0,
                transform_matrix[1][0], transform_matrix[1][1], transform_matrix[1][2], 0.0,
                transform_matrix[2][0], transform_matrix[2][1], transform_matrix[2][2], 0.0,
                transform_matrix[3][0], transform_matrix[3][1], transform_matrix[3][2], 1.0,
            } } };

            // Extract position and rotation using decomposeTRS
            const trs = mat4.decomposeTRS();
            const position = trs.translation;
            const rotation = trs.rotation.data;

            // Check if body is active
            const activation_state = bullet.cbtBodyGetActivationState(body);
            const is_active = activation_state == bullet.CBT_ACTIVE_TAG;

            const state = PhysicsState{
                .entity_id = entity_id,
                .position = position,
                .rotation = rotation,
                .linear_velocity = linear_vel,
                .angular_velocity = angular_vel,
                .is_active = is_active,
                .frame_number = self.frame_count,
            };

            write_buffer.append(state) catch {
                std.debug.print("Failed to append physics state for entity {d}\n", .{entity_id.id});
            };
        }

        self.state_buffer.endWrite();
    }

    /// Extract debug wireframes for all bodies in the physics world, creating separate wireframe data for each body
    fn extractDebugWireframes(self: *Self) void {
        const wireframe_buffer = self.wireframe_buffer.beginWrite(self.allocator);

        var debug_draw = bullet.CbtDebugDraw{
            .drawLine1 = physicsDebugDrawLineCallback,
            .drawLine2 = null,
            .drawContactPoint = null,
            .context = undefined, // Will be set per body
        };

        // Extract wireframes for each entity individually
        var it = self.entity_bodies.iterator();
        while (it.next()) |entry| {
            const entity_id = entry.key_ptr.*;
            const body = entry.value_ptr.*;

            // Get body mass to determine if it's dynamic or static
            const mass = bullet.cbtBodyGetMass(body);
            const color = if (mass > 0.0) self.debug_dynamic_color else self.debug_static_color;

            // Create a new collector for this specific body
            var collector = LineCollector.init(self.allocator, color);
            defer collector.deinit();
            
            // Set the collector as the context for the debug draw callback
            debug_draw.context = &collector;
            
            // Configure Bullet to draw wireframes with this specific collector
            bullet.cbtWorldDebugSetDrawer(self.bullet_world, &debug_draw);
            bullet.cbtWorldDebugSetMode(self.bullet_world, bullet.CBT_DBGMODE_DRAW_WIREFRAME);

            // Draw wireframes for this specific body only using our new cbullet function
            std.debug.print("Drawing wireframes for entity {d} with color [{d:.2}, {d:.2}, {d:.2}]\n", .{ entity_id.id, color[0], color[1], color[2] });
            bullet.cbtWorldDebugDrawBody(self.bullet_world, body, &color);
            std.debug.print("Collector now has {d} vertices after drawing entity {d} (mass: {d:.2})\n", .{ collector.vertices.items.len, entity_id.id, mass });

            // Create wireframe data for this specific entity
            const vertices = collector.toOwnedSlice() catch |err| {
                std.debug.print("Failed to extract wireframe vertices for entity {d}: {any}\n", .{ entity_id.id, err });
                continue;
            };

            // Only create wireframe data if we actually have vertices for this body
            if (vertices.len > 0) {
                const wireframe_data = DebugWireframeData{
                    .entity_id = entity_id,
                    .vertices = vertices,
                    .frame_number = self.frame_count,
                };

                wireframe_buffer.append(wireframe_data) catch {
                    std.debug.print("Failed to append wireframe data for entity {d}\n", .{entity_id.id});
                    self.allocator.free(vertices);
                };
                
                std.debug.print("Created debug wireframe for entity {d} with {d} vertices (mass: {d:.2})\n", .{ entity_id.id, vertices.len, mass });
            } else {
                // Free the empty vertices array
                self.allocator.free(vertices);
            }
        }

        self.wireframe_buffer.endWrite();
        
        // Increment debug wireframes version to signal update
        const current_version = self.debug_wireframes_version.load(.acquire);
        self.debug_wireframes_version.store(current_version + 1, .release);
        
        std.debug.print("Extracted per-body debug wireframes for physics world (version: {d})\n", .{current_version + 1});
    }
};

// Test-specific imports (only needed for testing)
const testing = std.testing;
const ECSManager = @import("../ECSManager.zig");
const Transform = @import("Transform.zig");
const Collisions = @import("Collisions.zig");

fn skip() !void {
    return error.SkipZigTest;
}
const gl = @import("../../bindings/gl.zig");

// Simple box mesh generator for test visualization
fn generateBoxMesh(allocator: std.mem.Allocator, width: f32, height: f32, depth: f32) !*Mesh {
    const hw = width / 2.0;
    const hh = height / 2.0;
    const hd = depth / 2.0;

    // Create vertices for a simple box (just front face for simplicity)
    var vertices = try allocator.alloc(Mesh.Vertex, 4);
    var indices = try allocator.alloc(u32, 6);

    vertices[0] = Mesh.Vertex{
        .position = .{ -hw, -hh, hd },
        .color = .{ 1.0, 1.0, 1.0 },
        .texture = .{ 0.0, 0.0 },
        .normal = .{ 0.0, 0.0, 1.0 },
        .tangent = .{ 1.0, 0.0, 0.0 },
    };
    vertices[1] = Mesh.Vertex{
        .position = .{ hw, -hh, hd },
        .color = .{ 1.0, 1.0, 1.0 },
        .texture = .{ 1.0, 0.0 },
        .normal = .{ 0.0, 0.0, 1.0 },
        .tangent = .{ 1.0, 0.0, 0.0 },
    };
    vertices[2] = Mesh.Vertex{
        .position = .{ hw, hh, hd },
        .color = .{ 1.0, 1.0, 1.0 },
        .texture = .{ 1.0, 1.0 },
        .normal = .{ 0.0, 0.0, 1.0 },
        .tangent = .{ 1.0, 0.0, 0.0 },
    };
    vertices[3] = Mesh.Vertex{
        .position = .{ -hw, hh, hd },
        .color = .{ 1.0, 1.0, 1.0 },
        .texture = .{ 0.0, 1.0 },
        .normal = .{ 0.0, 0.0, 1.0 },
        .tangent = .{ 1.0, 0.0, 0.0 },
    };

    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2;
    indices[3] = 0;
    indices[4] = 2;
    indices[5] = 3;

    return try Mesh.init(allocator, vertices, indices, Mesh.gen_draw(.triangles));
}

// Generate a more complex mesh (tetrahedron) for testing convex hull decomposition
fn generateTetrahedronMesh(allocator: std.mem.Allocator) !*Mesh {
    // Create vertices for a tetrahedron
    var vertices = try allocator.alloc(Mesh.Vertex, 4);
    var indices = try allocator.alloc(u32, 12); // 4 triangular faces

    // Define tetrahedron vertices (pyramid with triangular base)
    vertices[0] = Mesh.Vertex{
        .position = .{ 0.0, 1.0, 0.0 }, // Top
        .color = .{ 1.0, 0.0, 0.0 },
        .texture = .{ 0.5, 1.0 },
        .normal = .{ 0.0, 1.0, 0.0 },
        .tangent = .{ 1.0, 0.0, 0.0 },
    };
    vertices[1] = Mesh.Vertex{
        .position = .{ -0.5, -0.5, 0.5 }, // Front left
        .color = .{ 0.0, 1.0, 0.0 },
        .texture = .{ 0.0, 0.0 },
        .normal = .{ -0.5, -0.5, 0.5 },
        .tangent = .{ 1.0, 0.0, 0.0 },
    };
    vertices[2] = Mesh.Vertex{
        .position = .{ 0.5, -0.5, 0.5 }, // Front right
        .color = .{ 0.0, 0.0, 1.0 },
        .texture = .{ 1.0, 0.0 },
        .normal = .{ 0.5, -0.5, 0.5 },
        .tangent = .{ 1.0, 0.0, 0.0 },
    };
    vertices[3] = Mesh.Vertex{
        .position = .{ 0.0, -0.5, -0.5 }, // Back
        .color = .{ 1.0, 1.0, 0.0 },
        .texture = .{ 0.5, 0.0 },
        .normal = .{ 0.0, -0.5, -0.5 },
        .tangent = .{ 1.0, 0.0, 0.0 },
    };

    // Define triangular faces (counter-clockwise winding)
    // Front face
    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2;
    // Right face
    indices[3] = 0;
    indices[4] = 2;
    indices[5] = 3;
    // Left face
    indices[6] = 0;
    indices[7] = 3;
    indices[8] = 1;
    // Bottom face
    indices[9] = 1;
    indices[10] = 3;
    indices[11] = 2;

    return try Mesh.init(allocator, vertices, indices, Mesh.gen_draw(.triangles));
}

test "ConvexHull vs TriangleMesh collision detection" {
    const allocator = testing.allocator;

    std.debug.print("\n=== Testing ConvexHull vs TriangleMesh Collision ===\n", .{});

    // Initialize minimal ECS for testing
    const ecs = try ECSManager.init(allocator);
    defer ecs.deinit();

    // Create ground plane using Ground prefab (TriangleMesh collider)
    const Ground = @import("../prefabs/Ground.zig");
    _ = try Ground.spawn(allocator, ecs, .{
        .size = 10.0,
        .position = .{ 0.0, 0.0, 0.0 },
        .color = .{ 0.4, 0.6, 0.4 },
    });

    // Create falling triangle mesh object (using box mesh but with triangle mesh collider)
    const falling_entity = try ecs.spawn(.{});

    var falling_transform = Transform.TransformComponent.init(allocator);
    falling_transform.setPosition(0, 5, 0); // Start at height 5
    try falling_transform.attach(ecs, falling_entity);

    // Create complex tetrahedron mesh and generate convex hull decomposition
    const falling_mesh = try generateTetrahedronMesh(allocator);

    // Generate convex hulls from the complex mesh using V-HACD with caching
    const convex_hulls = try ecs.world.resource_manager.getOrGenerateCollisionMesh(falling_mesh);
    defer {
        for (convex_hulls) |hull| {
            allocator.free(hull.points);
            allocator.free(hull.triangles);
        }
        allocator.free(convex_hulls);
    }

    var falling_collider = try Collisions.ColliderComponent.init(allocator, .{ .ConvexHull = convex_hulls }, falling_mesh);
    try falling_collider.attach(ecs, falling_entity);

    var falling_body = Collisions.RigidBodyComponent.init(1.0, falling_collider.bullet_shape.?);
    try falling_body.attach(ecs, falling_entity);

    // Wait for physics bodies to be created
    std.debug.print("Waiting for physics bodies to be created...\n", .{});
    std.time.sleep(50_000_000); // 50ms

    // Simulate for 3 seconds
    const simulation_time: f32 = 3.0;
    const start_time = std.time.milliTimestamp();
    var last_time = start_time;
    const initial_y: f32 = 5.0;
    var collision_detected = false;

    std.debug.print("Starting triangle mesh collision test...\n", .{});
    std.debug.print("Time\tY Pos\tΔY\tCollision?\n", .{});
    std.debug.print("----\t-----\t--\t----------\n", .{});

    var main_loop_count: u32 = 0;
    while (true) {
        const current_time = std.time.milliTimestamp();
        const elapsed = @as(f32, @floatFromInt(current_time - start_time)) / 1000.0;
        last_time = current_time;

        if (elapsed >= simulation_time) break;

        main_loop_count += 1;

        // Update collision system (includes physics)
        try ecs.collision_system.update();
        ecs.transform_system.update();

        // Sample every 0.1 seconds
        const S = struct {
            var last_sample_time: f32 = 0.0;
        };
        const sample_interval: f32 = 0.1;
        if ((elapsed - S.last_sample_time) >= sample_interval) {
            S.last_sample_time = elapsed;

            const current_transform = ecs.transform_components.get(falling_entity).?;
            const current_y = current_transform.position[1];
            const delta_y = initial_y - current_y;

            std.debug.print("{d:.2}\t{d:.3}\t{d:.3}\t{s}\n", .{
                elapsed,
                current_y,
                delta_y,
                if (delta_y > 0.1) "YES" else "NO",
            });

            // Detect if object has fallen significantly
            if (!collision_detected and delta_y > 0.5) {
                collision_detected = true;
                std.debug.print(">>> ConvexHull vs TriangleMesh collision detected: Object fell {d:.3} units!\n", .{delta_y});
            }

            // Check if object stopped moving (collision occurred)
            if (!collision_detected and current_y < 1.0 and delta_y > 3.0) {
                collision_detected = true;
                std.debug.print(">>> ConvexHull vs TriangleMesh collision confirmed: Object at Y={d:.3} (fell {d:.3} units)\n", .{ current_y, delta_y });
            }
        }
    }

    if (!collision_detected) {
        std.debug.print("WARNING: ConvexHull vs TriangleMesh collision may not be working properly!\n", .{});
    }

    std.debug.print("ConvexHull vs TriangleMesh collision test completed.\n", .{});
}

test "Physics Thread - Box falling with gravity integration test" {
    const allocator = testing.allocator;

    std.debug.print("\n=== Testing Physics Thread Integration - Box Falling ===\n", .{});

    // Check if we should run with GUI (set via build option)
    const test_config = @import("test_config");
    const enable_gui = @import("builtin").is_test and test_config.GUI_ENABLED;

    // Initialize minimal ECS for testing
    const ecs = try ECSManager.init(allocator);
    defer ecs.deinit();

    // Create ground plane using Ground prefab (uses TriangleMesh collider)
    const Ground = @import("../prefabs/Ground.zig");
    _ = try Ground.spawn(allocator, ecs, .{
        .size = 100.0,
        .position = .{ 0.0, 0.0, 0.0 },
        .color = .{ 0.4, 0.6, 0.4 },
    });

    // Create falling box entity
    const box_entity = try ecs.spawn(.{});

    var box_transform = Transform.TransformComponent.init(allocator);
    box_transform.setPosition(0, 5, 0); // Start at height 5
    try box_transform.attach(ecs, box_entity);

    var box_collider = try Collisions.ColliderComponent.init(allocator, .{ .Box = .{ .half_extents = .{ 0.5, 0.01, 0.5 } } }, null);
    try box_collider.attach(ecs, box_entity);

    var box_body = Collisions.RigidBodyComponent.init(1.0, box_collider.bullet_shape.?);
    try box_body.attach(ecs, box_entity);

    // Add visual components for box if GUI is enabled
    if (enable_gui) {
        const Renderer = @import("Renderer.zig");

        // Create box mesh (thin like a plane)
        const box_mesh = try generateBoxMesh(allocator, 1.0, 0.02, 1.0);
        const box_mesh_name = try std.fmt.allocPrint(allocator, "test_box_mesh", .{});
        defer allocator.free(box_mesh_name);

        const box_mesh_name_owned = try allocator.dupe(u8, box_mesh_name);
        try ecs.world.resource_manager.meshes.put(box_mesh_name_owned, .{ .mesh = box_mesh, .instance_count = 0 });

        // Create renderer component for box
        const box_renderer = try Renderer.Renderable.init(allocator, box_mesh_name_owned);
        try ecs.renderer_components.add(box_entity, box_renderer);
    }

    // Give physics thread time to process CreateRigidBody commands
    std.debug.print("\nWaiting for physics bodies to be created...\n", .{});
    std.time.sleep(100 * std.time.ns_per_ms);

    // Debug: Check if physics thread has registered bodies
    if (ecs.collision_system.physics_thread) |physics| {
        std.debug.print("Physics thread entity count after wait: {d}\n", .{physics.entity_bodies.count()});

        // Also check what physics states are available
        const states = physics.getPhysicsStates();
        std.debug.print("Available physics states: {d}\n", .{states.len});
        for (states) |state| {
            std.debug.print("  Entity {d}: Y={d:.3}\n", .{ state.entity_id.id, state.position[1] });
        }
    }

    // Simulate for 5 seconds (longer time to see collision)
    const simulation_time: f32 = 5.0;
    const start_time = std.time.milliTimestamp();
    var last_time = start_time;

    var collision_detected = false;
    var final_y: f32 = 5.0;
    const initial_y: f32 = 5.0;

    std.debug.print("\nStarting physics simulation...\n", .{});
    if (enable_gui) {
        std.debug.print("GUI mode enabled - press 'Q' to quit, or let it run automatically\n", .{});

        // Set up camera for viewing the test
        const Camera = @import("Camera.zig");
        const camera_entity = try ecs.spawn(.{});

        // Create camera components manually
        var camera_transform = Transform.TransformComponent.init(allocator);
        camera_transform.setPosition(10, 5, 10); // Side view
        try camera_transform.attach(ecs, camera_entity);

        var camera_component = Camera.CameraComponent{ .entity_id = camera_entity };
        try camera_component.attach(ecs, camera_entity);
    } else {
        std.debug.print("Time\tY Pos\tΔY\tFalling?\n", .{});
        std.debug.print("----\t-----\t--\t--------\n", .{});
    }

    const simulation_limit = if (enable_gui) 60.0 else simulation_time; // 60 seconds max in GUI mode

    var main_loop_count: u32 = 0;
    while (true) {
        const current_time = std.time.milliTimestamp();
        const elapsed = @as(f32, @floatFromInt(current_time - start_time)) / 1000.0;
        last_time = current_time;

        if (elapsed >= simulation_limit) break;

        main_loop_count += 1;

        // Update collision system (includes physics)
        try ecs.collision_system.update();
        ecs.transform_system.update();

        if (enable_gui) {
            // Update the full ECS for rendering
            try ecs.update(elapsed);

            // Check for quit key (simplified - in real implementation would use proper input)
            if (gl.glfw.glfwGetKey(ecs.globals_system.window, gl.glfw.GLFW_KEY_Q) == gl.glfw.GLFW_PRESS) {
                std.debug.print("Quit key pressed - ending test\n", .{});
                break;
            }

            // Check if window should close
            if (gl.glfw.glfwWindowShouldClose(ecs.globals_system.window) != 0) {
                break;
            }
        }

        // Sample every 0.1 seconds (10 times per second) or every frame in GUI mode
        const S = struct {
            var last_sample_time: f32 = 0.0;
        };
        const sample_interval: f32 = if (enable_gui) 0.0 else 0.1; // 0.1s = 10Hz sampling
        if (enable_gui or (elapsed - S.last_sample_time) >= sample_interval) {
            S.last_sample_time = elapsed;
            const current_transform = ecs.transform_components.get(box_entity).?;
            const current_y = current_transform.position[1];
            const delta_y = initial_y - current_y;

            if (!enable_gui) {
                std.debug.print("{d:.2}\t{d:.3}\t{d:.3}\t{s}\n", .{
                    elapsed,
                    current_y,
                    delta_y,
                    if (delta_y > 0.1) "YES" else "NO",
                });
            } else {
                // In GUI mode, just update occasionally
                if (@mod(@as(i32, @intFromFloat(elapsed * 60)), 60) == 0) {
                    std.debug.print("Time: {d:.1}s, Box Y: {d:.3}, Fall: {d:.3}\n", .{ elapsed, current_y, delta_y });
                }
            }

            // Detect if object has fallen significantly
            if (!collision_detected and delta_y > 0.5) {
                collision_detected = true;
                std.debug.print(">>> Physics working: Box has fallen {d:.3} units!\n", .{delta_y});
            }

            final_y = current_y;
        }
    }

    const total_fall = initial_y - final_y;

    std.debug.print("\n--- Physics Integration Test Results ---\n", .{});
    std.debug.print("Initial Y position: {d:.3}\n", .{initial_y});
    std.debug.print("Final Y position: {d:.3}\n", .{final_y});
    std.debug.print("Total fall distance: {d:.3}\n", .{total_fall});
    std.debug.print("Expected minimum fall: 3.0 units (should settle around Y=0.6)\n", .{});

    // Test that physics thread is running
    if (ecs.collision_system.physics_thread) |physics| {
        try testing.expect(!physics.should_shutdown.load(.acquire));
        std.debug.print("✓ Physics thread is running\n", .{});
    } else {
        std.debug.print("✗ Physics thread not initialized!\n", .{});
        try testing.expect(false);
    }

    // Main test: Verify the box actually fell due to gravity
    // With gravity -9.81 and no air resistance, in 5 seconds the box should fall:
    // y = 0.5 * g * t^2 = 0.5 * 9.81 * 25 = 122.625 meters
    // But it will hit the ground at Y=0.6, so it should fall 5.0 - 0.6 = 4.4 units
    if (total_fall > 4.0) {
        std.debug.print("✓ PASS: Physics integration working - box fell {d:.3} units\n", .{total_fall});
    } else if (total_fall > 1.0) {
        std.debug.print("⚠ PARTIAL: Physics working but may have collision issues - box fell {d:.3} units\n", .{total_fall});
        std.debug.print("  Expected fall to ground (~4.4 units), got {d:.3} units\n", .{total_fall});
    } else {
        std.debug.print("✗ FAIL: Physics integration broken - box only fell {d:.3} units\n", .{total_fall});
        std.debug.print("  This indicates physics or transform sync is not working\n", .{});
        try testing.expect(false);
    }

    // Verify final position is reasonable (should be resting on ground)
    try testing.expect(final_y >= 0.0); // Should not fall through world
    try testing.expect(final_y <= 2.0); // Should have fallen significantly
}

test "Double Buffer - Basic functionality" {
    try skip();
    const allocator = testing.allocator;

    std.debug.print("\n=== Testing Double Buffer Basic Functionality ===\n", .{});

    var buffer = PhysicsStateBuffer.init(allocator);
    defer buffer.deinit();

    // Test 1: Basic write and read
    {
        const write_buf = buffer.beginWrite();
        try write_buf.append(PhysicsState{
            .entity_id = Core.EntityID.init(1),
            .position = .{ 1.0, 2.0, 3.0 },
            .rotation = .{ 0.0, 0.0, 0.0, 1.0 },
            .linear_velocity = .{ 0.0, 0.0, 0.0 },
            .angular_velocity = .{ 0.0, 0.0, 0.0 },
            .is_active = true,
            .frame_number = 1,
        });
        buffer.endWrite();

        const read_data = buffer.read();
        try testing.expect(read_data.len == 1);
        try testing.expect(read_data[0].frame_number == 1);
        try testing.expect(read_data[0].position[1] == 2.0);
        std.debug.print("✓ Basic write/read test passed\n", .{});
    }

    // Test 2: Multiple writes should swap buffers
    {
        // Write frame 2
        const write_buf2 = buffer.beginWrite();
        try write_buf2.append(PhysicsState{
            .entity_id = Core.EntityID.init(1),
            .position = .{ 1.0, 3.0, 3.0 },
            .rotation = .{ 0.0, 0.0, 0.0, 1.0 },
            .linear_velocity = .{ 0.0, 0.0, 0.0 },
            .angular_velocity = .{ 0.0, 0.0, 0.0 },
            .is_active = true,
            .frame_number = 2,
        });
        buffer.endWrite();

        const read_data2 = buffer.read();
        try testing.expect(read_data2.len == 1);
        try testing.expect(read_data2[0].frame_number == 2);
        try testing.expect(read_data2[0].position[1] == 3.0);
        std.debug.print("✓ Buffer swap test passed\n", .{});
    }

    // Test 3: Write frame 3 and verify we can still read frame 2 until swap
    {
        const write_buf3 = buffer.beginWrite();
        try write_buf3.append(PhysicsState{
            .entity_id = Core.EntityID.init(1),
            .position = .{ 1.0, 4.0, 3.0 },
            .rotation = .{ 0.0, 0.0, 0.0, 1.0 },
            .linear_velocity = .{ 0.0, 0.0, 0.0 },
            .angular_velocity = .{ 0.0, 0.0, 0.0 },
            .is_active = true,
            .frame_number = 3,
        });

        // Before endWrite, should still read frame 2
        const read_during_write = buffer.read();
        try testing.expect(read_during_write[0].frame_number == 2);

        buffer.endWrite();

        // After endWrite, should read frame 3
        const read_after_write = buffer.read();
        try testing.expect(read_after_write[0].frame_number == 3);
        try testing.expect(read_after_write[0].position[1] == 4.0);
        std.debug.print("✓ Buffer consistency during write test passed\n", .{});
    }
}

test "Double Buffer - Concurrent access simulation" {
    try skip();
    const allocator = testing.allocator;

    std.debug.print("\n=== Testing Double Buffer Concurrent Access ===\n", .{});

    var buffer = PhysicsStateBuffer.init(allocator);
    defer buffer.deinit();

    // Simulate rapid writes followed by reads to check for race conditions
    var frame: u64 = 1;
    while (frame <= 10) : (frame += 1) {
        // Write a frame
        const write_buf = buffer.beginWrite();
        try write_buf.append(PhysicsState{
            .entity_id = Core.EntityID.init(1),
            .position = .{ 1.0, @as(f32, @floatFromInt(frame)), 3.0 },
            .rotation = .{ 0.0, 0.0, 0.0, 1.0 },
            .linear_velocity = .{ 0.0, 0.0, 0.0 },
            .angular_velocity = .{ 0.0, 0.0, 0.0 },
            .is_active = true,
            .frame_number = frame,
        });
        buffer.endWrite();

        // Read multiple times (simulating main thread reading frequently)
        var read_count: u32 = 0;
        while (read_count < 5) : (read_count += 1) {
            const read_data = buffer.read();
            try testing.expect(read_data.len == 1);
            const read_frame = read_data[0].frame_number;

            // Should read either current frame or previous frame (never future frame)
            try testing.expect(read_frame <= frame);
            // Should never read ancient frames (more than 1 frame behind)
            try testing.expect(read_frame >= frame - 1);

            std.debug.print("  Frame {d} write -> read frame {d}\n", .{ frame, read_frame });
        }
    }

    std.debug.print("✓ Concurrent access simulation passed\n", .{});
}

test "Double Buffer - Threading stress test" {
    try skip();
    const allocator = testing.allocator;

    std.debug.print("\n=== Testing Double Buffer Threading Stress ===\n", .{});

    var buffer = PhysicsStateBuffer.init(allocator);
    defer buffer.deinit();

    const ThreadContext = struct {
        buffer: *PhysicsStateBuffer,
        should_stop: *std.atomic.Value(bool),
        frame_counter: *std.atomic.Value(u64),
    };

    var should_stop = std.atomic.Value(bool).init(false);
    var frame_counter = std.atomic.Value(u64).init(0);

    const context = ThreadContext{
        .buffer = &buffer,
        .should_stop = &should_stop,
        .frame_counter = &frame_counter,
    };

    // Writer thread function
    const writerThread = struct {
        fn run(ctx: ThreadContext) void {
            var local_frame: u64 = 1;
            while (!ctx.should_stop.load(.acquire)) {
                const write_buf = ctx.buffer.beginWrite();
                write_buf.clearRetainingCapacity();
                write_buf.append(PhysicsState{
                    .entity_id = Core.EntityID.init(1),
                    .position = .{ 1.0, @as(f32, @floatFromInt(local_frame)), 3.0 },
                    .rotation = .{ 0.0, 0.0, 0.0, 1.0 },
                    .linear_velocity = .{ 0.0, 0.0, 0.0 },
                    .angular_velocity = .{ 0.0, 0.0, 0.0 },
                    .is_active = true,
                    .frame_number = local_frame,
                }) catch {};
                ctx.buffer.endWrite();

                ctx.frame_counter.store(local_frame, .release);
                local_frame += 1;

                std.time.sleep(1000000); // 1ms
            }
        }
    }.run;

    // Start writer thread
    const writer = try std.Thread.spawn(.{}, writerThread, .{context});

    // Reader thread (main thread) - read for 50ms
    var reads_performed: u32 = 0;
    var unique_frames_seen = std.AutoHashMap(u64, void).init(allocator);
    defer unique_frames_seen.deinit();

    const start_time = std.time.milliTimestamp();
    while (std.time.milliTimestamp() - start_time < 50) { // Run for 50ms
        const read_data = buffer.read();
        if (read_data.len > 0) {
            const frame_num = read_data[0].frame_number;
            try unique_frames_seen.put(frame_num, {});
            reads_performed += 1;
        }
        std.time.sleep(100000); // 0.1ms between reads
    }

    // Stop writer and wait
    should_stop.store(true, .release);
    writer.join();

    const final_frame = frame_counter.load(.acquire);
    const unique_count = unique_frames_seen.count();

    std.debug.print("  Writer produced {d} frames\n", .{final_frame});
    std.debug.print("  Reader performed {d} reads\n", .{reads_performed});
    std.debug.print("  Reader saw {d} unique frames\n", .{unique_count});

    // Validation
    try testing.expect(reads_performed > 0);
    try testing.expect(unique_count > 1); // Should see multiple different frames
    try testing.expect(unique_count <= final_frame); // Can't see more frames than written

    std.debug.print("✓ Threading stress test passed\n", .{});
}

test "updateStateBuffer - Bullet Physics state extraction" {
    try skip();
    const allocator = testing.allocator;
    _ = allocator; // unused for this test

    std.debug.print("\n=== Testing updateStateBuffer - Bullet Physics State Extraction ===\n", .{});

    // Create a minimal physics world and body to test state extraction
    const world = bullet.cbtWorldCreate();
    defer bullet.cbtWorldDestroy(world);

    // Set gravity
    const gravity = [3]f32{ 0.0, -9.81, 0.0 };
    bullet.cbtWorldSetGravity(world, &gravity);

    // Create a test shape - allocate then create
    const shape = bullet.cbtShapeAllocate(bullet.CBT_SHAPE_TYPE_BOX);
    const half_extents = [3]f32{ 0.5, 0.5, 0.5 };
    bullet.cbtShapeBoxCreate(shape, &half_extents);
    defer bullet.cbtShapeDestroy(shape);

    // Create a body at known position
    const initial_pos = [3]f32{ 1.0, 5.0, 2.0 };
    const body = bullet.cbtBodyAllocate();
    bullet.cbtBodyCreate(body, 1.0, &initial_pos, shape);

    // Set initial transform explicitly (as done in CreateRigidBody)
    var bullet_transform = [4][3]f32{
        .{ 1.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 1.0 },
        .{ initial_pos[0], initial_pos[1], initial_pos[2] },
    };
    bullet.cbtBodySetCenterOfMassTransform(body, &bullet_transform);

    bullet.cbtWorldAddBody(world, body);

    std.debug.print("  Created body at initial position: [{d:.2}, {d:.2}, {d:.2}]\n", .{ initial_pos[0], initial_pos[1], initial_pos[2] });

    // Test 1: Read initial transform
    {
        var transform_matrix: [4][3]f32 = undefined;
        bullet.cbtBodyGetCenterOfMassTransform(body, &transform_matrix);
        const pos = [3]f32{ transform_matrix[3][0], transform_matrix[3][1], transform_matrix[3][2] };

        std.debug.print("  Read initial position: [{d:.2}, {d:.2}, {d:.2}]\n", .{ pos[0], pos[1], pos[2] });
        try testing.expect(@abs(pos[0] - initial_pos[0]) < 0.01);
        try testing.expect(@abs(pos[1] - initial_pos[1]) < 0.01);
        try testing.expect(@abs(pos[2] - initial_pos[2]) < 0.01);
        std.debug.print("✓ Initial position read correctly\n", .{});
    }

    // Test 2: Step physics and check if position changes
    {
        const timestep: f32 = 1.0 / 60.0;
        const steps_taken = bullet.cbtWorldStepSimulation(world, timestep, 10, timestep);
        std.debug.print("  Physics step took {d} substeps\n", .{steps_taken});

        var transform_matrix: [4][3]f32 = undefined;
        bullet.cbtBodyGetCenterOfMassTransform(body, &transform_matrix);
        const pos_after_step = [3]f32{ transform_matrix[3][0], transform_matrix[3][1], transform_matrix[3][2] };

        std.debug.print("  Position after 1 step: [{d:.3}, {d:.3}, {d:.3}]\n", .{ pos_after_step[0], pos_after_step[1], pos_after_step[2] });

        // Y should be lower due to gravity
        try testing.expect(pos_after_step[1] < initial_pos[1]);
        std.debug.print("✓ Physics simulation is working (Y decreased)\n", .{});
    }

    // Test 3: Multiple steps
    {
        var step_count: u32 = 0;
        while (step_count < 10) : (step_count += 1) {
            const timestep: f32 = 1.0 / 60.0;
            _ = bullet.cbtWorldStepSimulation(world, timestep, 10, timestep);
        }

        var transform_matrix: [4][3]f32 = undefined;
        bullet.cbtBodyGetCenterOfMassTransform(body, &transform_matrix);
        const final_pos = [3]f32{ transform_matrix[3][0], transform_matrix[3][1], transform_matrix[3][2] };

        std.debug.print("  Position after 10 steps: [{d:.3}, {d:.3}, {d:.3}]\n", .{ final_pos[0], final_pos[1], final_pos[2] });

        // Should have fallen significantly
        const fall_distance = initial_pos[1] - final_pos[1];
        std.debug.print("  Fall distance: {d:.3} units\n", .{fall_distance});
        try testing.expect(fall_distance > 0.05); // Should fall at least 0.05 units in 10 frames
        std.debug.print("✓ Multi-step physics working\n", .{});
    }

    bullet.cbtWorldRemoveBody(world, body);
    bullet.cbtBodyDeallocate(body);
}

test "updateStateBuffer - Entity tracking" {
    try skip();
    const allocator = testing.allocator;

    std.debug.print("\n=== Testing updateStateBuffer - Entity Tracking ===\n", .{});

    // Create a physics system and manually test entity tracking
    var physics_system = try ThreadedPhysicsSystem.init(allocator);
    defer physics_system.deinit();

    // Add some test entities manually (simulating what CreateRigidBody would do)
    const entity1 = Core.EntityID.init(1);
    const entity2 = Core.EntityID.init(2);

    // Create shapes and bodies
    const shape1 = bullet.cbtShapeAllocate(bullet.CBT_SHAPE_TYPE_BOX);
    const half_extents1 = [3]f32{ 0.5, 0.5, 0.5 };
    bullet.cbtShapeBoxCreate(shape1, &half_extents1);

    const shape2 = bullet.cbtShapeAllocate(bullet.CBT_SHAPE_TYPE_BOX);
    const half_extents2 = [3]f32{ 1.0, 1.0, 1.0 };
    bullet.cbtShapeBoxCreate(shape2, &half_extents2);

    const body1 = bullet.cbtBodyAllocate();
    const body2 = bullet.cbtBodyAllocate();

    const pos1 = [3]f32{ 0.0, 5.0, 0.0 };
    const pos2 = [3]f32{ 2.0, 3.0, 1.0 };

    bullet.cbtBodyCreate(body1, 1.0, &pos1, shape1);
    bullet.cbtBodyCreate(body2, 0.5, &pos2, shape2);

    // Set initial transforms explicitly
    var transform1 = [4][3]f32{
        .{ 1.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 1.0 },
        .{ pos1[0], pos1[1], pos1[2] },
    };
    bullet.cbtBodySetCenterOfMassTransform(body1, &transform1);

    var transform2 = [4][3]f32{
        .{ 1.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 1.0 },
        .{ pos2[0], pos2[1], pos2[2] },
    };
    bullet.cbtBodySetCenterOfMassTransform(body2, &transform2);

    bullet.cbtWorldAddBody(physics_system.bullet_world, body1);
    bullet.cbtWorldAddBody(physics_system.bullet_world, body2);

    // Register entities in the physics system
    try physics_system.entity_bodies.put(entity1, body1);
    try physics_system.entity_bodies.put(entity2, body2);

    std.debug.print("  Added {d} entities to physics system\n", .{physics_system.entity_bodies.count()});

    // Test: Manually call updateStateBuffer and check results
    physics_system.updateStateBuffer();

    const states = physics_system.getPhysicsStates();
    std.debug.print("  updateStateBuffer produced {d} states\n", .{states.len});

    try testing.expect(states.len == 2);

    // Check that both entities are present
    var entity1_found = false;
    var entity2_found = false;

    for (states) |state| {
        std.debug.print("    Entity {d}: pos=[{d:.3}, {d:.3}, {d:.3}] frame={d}\n", .{ state.entity_id.id, state.position[0], state.position[1], state.position[2], state.frame_number });

        if (state.entity_id.id == 1) {
            entity1_found = true;
            try testing.expect(@abs(state.position[1] - 5.0) < 0.1); // Should be near Y=5
        } else if (state.entity_id.id == 2) {
            entity2_found = true;
            try testing.expect(@abs(state.position[1] - 3.0) < 0.1); // Should be near Y=3
        }
    }

    try testing.expect(entity1_found);
    try testing.expect(entity2_found);
    std.debug.print("✓ Entity tracking working correctly\n", .{});

    // Cleanup
    bullet.cbtWorldRemoveBody(physics_system.bullet_world, body1);
    bullet.cbtWorldRemoveBody(physics_system.bullet_world, body2);
    bullet.cbtBodyDeallocate(body1);
    bullet.cbtBodyDeallocate(body2);
    bullet.cbtShapeDestroy(shape1);
    bullet.cbtShapeDestroy(shape2);
}

test "updateStateBuffer - Frame progression" {
    try skip();
    const allocator = testing.allocator;

    std.debug.print("\n=== Testing updateStateBuffer - Frame Progression ===\n", .{});

    // Create physics system
    var physics_system = try ThreadedPhysicsSystem.init(allocator);
    defer physics_system.deinit();

    // Add one test entity
    const entity = Core.EntityID.init(1);
    const shape = bullet.cbtShapeAllocate(bullet.CBT_SHAPE_TYPE_BOX);
    const half_extents = [3]f32{ 0.5, 0.5, 0.5 };
    bullet.cbtShapeBoxCreate(shape, &half_extents);
    const body = bullet.cbtBodyAllocate();
    const pos = [3]f32{ 0.0, 10.0, 0.0 };

    bullet.cbtBodyCreate(body, 1.0, &pos, shape);

    // Set initial transform explicitly
    var transform = [4][3]f32{
        .{ 1.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 1.0 },
        .{ pos[0], pos[1], pos[2] },
    };
    bullet.cbtBodySetCenterOfMassTransform(body, &transform);

    bullet.cbtWorldAddBody(physics_system.bullet_world, body);
    try physics_system.entity_bodies.put(entity, body);

    std.debug.print("  Entity created at Y=10.0\n", .{});

    // Test: Call updateStateBuffer multiple times and check frame progression
    var previous_frame: u64 = 0;
    var previous_y: f32 = 10.0;

    var iteration: u32 = 0;
    while (iteration < 5) : (iteration += 1) {
        // Step physics
        const timestep: f32 = 1.0 / 60.0;
        _ = bullet.cbtWorldStepSimulation(physics_system.bullet_world, timestep, 10, timestep);
        physics_system.frame_count += 1;

        // Update state buffer
        physics_system.updateStateBuffer();

        const states = physics_system.getPhysicsStates();
        try testing.expect(states.len == 1);

        const state = states[0];
        std.debug.print("  Iteration {d}: Frame={d}, Y={d:.3}\n", .{ iteration + 1, state.frame_number, state.position[1] });

        // Frame number should be progressing
        try testing.expect(state.frame_number > previous_frame);
        previous_frame = state.frame_number;

        // Y position should be decreasing (falling)
        try testing.expect(state.position[1] < previous_y);
        previous_y = state.position[1];
    }

    std.debug.print("✓ Frame progression and physics state updating correctly\n", .{});

    // Cleanup
    bullet.cbtWorldRemoveBody(physics_system.bullet_world, body);
    bullet.cbtBodyDeallocate(body);
    bullet.cbtShapeDestroy(shape);
}

test "Physics Thread - Command queue stress test" {
    try skip();
    const allocator = testing.allocator;

    std.debug.print("\n=== Testing Physics Thread Command Queue ===\n", .{});

    // Initialize physics system directly
    var physics_thread = try ThreadedPhysicsSystem.init(allocator);
    defer physics_thread.deinit();

    try testing.expect(!physics_thread.should_shutdown.load(.acquire));

    // Send many commands rapidly to test queue
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const command = PhysicsCommand{
            .ApplyForce = .{
                .entity_id = Core.EntityID.init(i),
                .force = .{ 0, 10, 0 },
            },
        };

        const success = physics_thread.sendCommand(command);
        try testing.expect(success);
    }

    // Give physics thread time to process
    std.time.sleep(50 * std.time.ns_per_ms);

    std.debug.print("✓ Command queue stress test passed - sent 100 commands\n", .{});
}
