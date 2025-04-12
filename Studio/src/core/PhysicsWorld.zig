// PhysicsWorld.zig
const std = @import("std");
const Thread = std.Thread;
const PhysicsBody = @import("PhysicsBody.zig");

const Self = @This();

allocator: std.mem.Allocator,
bodies: std.ArrayList(*PhysicsBody),
mutex: Thread.Mutex = Thread.Mutex{},
physics_thread: ?Thread = null,
running: bool = true,
physics_rate: f32 = 240.0, // Hz
gravity: [3]f32 = .{ 0.0, -9.81, 0.0 },

pub fn init(allocator: std.mem.Allocator) !*Self {
    var world = try allocator.create(Self);

    world.* = Self{
        .allocator = allocator,
        .bodies = std.ArrayList(*PhysicsBody).init(allocator),
    };

    return world;
}

pub fn deinit(self: *Self) void {
    // Stop physics thread if running
    if (self.physics_thread != null) {
        self.running = false;
        self.physics_thread.?.join();
    }

    // Note: we don't own the bodies, just track them
    self.bodies.deinit();
    self.allocator.destroy(self);
}

pub fn addBody(self: *Self, body: *PhysicsBody) !void {
    self.mutex.lock();
    defer self.mutex.unlock();

    try self.bodies.append(body);
}

pub fn removeBody(self: *Self, body: *PhysicsBody) void {
    self.mutex.lock();
    defer self.mutex.unlock();

    // Find and remove the body
    for (self.bodies.items, 0..) |b, i| {
        if (b == body) {
            _ = self.bodies.swapRemove(i);
            break;
        }
    }
}

// This can be called from main thread for single-threaded physics
pub fn update(self: *Self, dt: f32) void {
    self.mutex.lock();
    defer self.mutex.unlock();

    // First apply gravity to all dynamic bodies
    for (self.bodies.items) |body| {
        if (body.body_type == .Dynamic) {
            const gravity_force = .{
                body.mass * self.gravity[0],
                body.mass * self.gravity[1],
                body.mass * self.gravity[2],
            };

            body.applyForce(.{ .x = gravity_force[0], .y = gravity_force[1], .z = gravity_force[2] });
        }
    }

    // Update all physics bodies
    for (self.bodies.items) |body| {
        body.update(dt);
    }

    // Detect and resolve collisions between bodies
    self.detectAndResolveCollisions();
}

// Starts physics in separate thread
pub fn startPhysicsThread(self: *Self) !void {
    self.running = true;
    self.physics_thread = try Thread.spawn(.{}, physicsThreadFn, .{self});
}

fn physicsThreadFn(self: *Self) void {
    const time_step = 1.0 / self.physics_rate;
    var last_time = std.time.milliTimestamp();

    while (self.running) {
        const current_time = std.time.milliTimestamp();
        const elapsed = @intToFloat(f32, current_time - last_time) / 1000.0;

        if (elapsed >= time_step) {
            self.update(time_step);
            last_time = current_time;
        } else {
            // Sleep a bit to avoid spinning too fast
            std.time.sleep(1 * std.time.ns_per_ms);
        }
    }
}

// Simplified collision detection and resolution
fn detectAndResolveCollisions(self: *Self) void {
    // For each pair of dynamic bodies
    for (self.bodies.items, 0..) |body_a, i| {
        if (body_a.body_type == .Static) continue;

        // Check ground collision as a simple example
        if (body_a.node.position[1] < 0.0) {
            // Simple ground collision
            body_a.node.position[1] = 0.0;

            // Reflect velocity with restitution
            if (body_a.velocity.y() < 0) {
                body_a.velocity = .{
                    .x = body_a.velocity.x(),
                    .y = -body_a.velocity.y() * body_a.restitution,
                    .z = body_a.velocity.z(),
                };
            }
        }

        // Check other bodies (simplified)
        var j: usize = i + 1;
        while (j < self.bodies.items.len) : (j += 1) {
            const body_b = self.bodies.items[j];
            if (body_b.body_type == .Static) continue;

            // Very simple sphere collision detection
            // In a real implementation, you'd use proper collision shapes
            const pos_a = .{
                .x = body_a.node.position[0],
                .y = body_a.node.position[1],
                .z = body_a.node.position[2],
            };

            const pos_b = .{
                .x = body_b.node.position[0],
                .y = body_b.node.position[1],
                .z = body_b.node.position[2],
            };

            const radius_a = 0.5; // Simplified - should be body property
            const radius_b = 0.5;

            const distance = pos_a.subtract(pos_b).length();
            const min_distance = radius_a + radius_b;

            if (distance < min_distance) {
                // Collision detected - implement resolution here
                // ...
            }
        }
    }
}
