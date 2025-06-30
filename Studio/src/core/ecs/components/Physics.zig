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

// Simplified Physics component - only stores info needed by Bullet physics
pub const PhysicsComponent = struct {
    const Self = @This();

    mass: f32 = 1.0,
    body_type: BodyType = .Dynamic,

    pub fn init(body_type: BodyType) Self {
        return .{
            .body_type = body_type,
        };
    }
};
