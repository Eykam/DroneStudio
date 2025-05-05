// src/ecs/components/PhysicsBody.zig
const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const Transform = @import("../components/Transform.zig");
const TransfromComponent = Transform.TransformComponent;

const Vec3 = Math.Vec3;

pub const BodyType = enum {
    Static,
    Dynamic,
    Kinematic,
};

pub const PhysicsComponent = struct {
    const Self = @This();

    velocity: [3]f32 = .{ 0, 0, 0 },
    acceleration: [3]f32 = .{ 0, 0, 0 },
    mass: f32 = 1.0,
    body_type: BodyType = .Dynamic,
    use_gravity: bool = true,
    is_grounded: bool = false,

    pub fn init(body_type: BodyType) Self {
        return .{
            .body_type = body_type,
        };
    }

    pub fn applyForce(self: *Self, force: [3]f32) void {
        if (self.body_type != .Dynamic) return;

        const force_x = force[0] / self.mass;
        const force_y = force[1] / self.mass;
        const force_z = force[2] / self.mass;

        self.acceleration[0] += force_x;
        self.acceleration[1] += force_y;
        self.acceleration[2] += force_z;
    }
};

const GRAVITY: f32 = 9.81;

pub const PhysicsSystem = struct {
    const Self = @This();

    world: *Core.World,
    transform_components: *SparseSet(TransfromComponent),
    physics_components: *SparseSet(PhysicsComponent),

    pub fn init(world: *Core.World, transform_components: *SparseSet(TransfromComponent), physics_components: *SparseSet(PhysicsComponent)) Self {
        return .{
            .world = world,
            .transform_components = transform_components,
            .physics_components = physics_components,
        };
    }

    pub fn update(self: *Self, dt: f32) void {
        var physics_iter = self.physics_components.iterator();

        while (physics_iter.next()) |tuple| {
            const entity_id = tuple.entity_id;
            const physics = tuple.component;

            // Skip static bodies
            if (physics.body_type == .Static) continue;

            // Apply gravity
            if (physics.use_gravity and !physics.is_grounded) {
                physics.acceleration[1] -= GRAVITY;
            }

            // Update velocity
            physics.velocity[0] += physics.acceleration[0] * dt;
            physics.velocity[1] += physics.acceleration[1] * dt;
            physics.velocity[2] += physics.acceleration[2] * dt;

            // Reset acceleration
            physics.acceleration = .{ 0, 0, 0 };

            // Apply velocity to transform
            if (self.transform_components.get(entity_id)) |transform| {
                transform.position[0] += physics.velocity[0] * dt;
                transform.position[1] += physics.velocity[1] * dt;
                transform.position[2] += physics.velocity[2] * dt;

                transform.updateLocalTransform();
            }

            // Apply damping
            physics.velocity[0] *= 0.98;
            physics.velocity[1] *= 0.98;
            physics.velocity[2] *= 0.98;

            // Ground collision check (simplified)
            if (physics.body_type == .Dynamic) {
                if (self.transform_components.get(entity_id)) |transform| {
                    if (transform.position[1] <= 0.0) {
                        transform.position[1] = 0.0;
                        physics.velocity[1] = 0.0;
                        physics.is_grounded = true;
                    } else {
                        physics.is_grounded = false;
                    }

                    transform.updateLocalTransform();
                }
            }
        }
    }
};
