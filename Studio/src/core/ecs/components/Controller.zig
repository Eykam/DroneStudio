// src/ecs/components/Controller.zig
const std = @import("std");
const gl = @import("../../bindings/gl.zig");
const Core = @import("../Core.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const Globals = @import("Globals.zig");
const Transform = @import("../components/Transform.zig");
const Physics = @import("../components/Physics.zig");
const Collisions = @import("Collisions.zig");
const ECSManager = @import("../ECSManager.zig");

const GlobalsComponent = Globals.GlobalsComponent;
const PhysicsComponent = Physics.PhysicsComponent;
const TransformComponent = Transform.TransformComponent;

const glfw = gl.glfw;

pub const ControllerComponent = struct {
    const Self = @This();

    const KeyBinding = struct {
        key: c_int,
        onPressed: *const fn (
            eid: Core.EntityID,
            transform: *Transform.TransformComponent,
            rigid_body: ?*Collisions.RigidBodyComponent,
            collision_system: ?*Collisions.CollisionSystem, 
            dt: f32,
        ) void,
    };

    pub const MouseMoveFn = *const fn (eid: Core.EntityID, tf: *TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*Collisions.CollisionSystem, dx: f32, dy: f32, dt: f64) void;

    move_speed: f32 = 5.0,
    mouse_move: ?MouseMoveFn = null,
    key_bindings: std.ArrayList(KeyBinding),

    pub fn init(allocator: std.mem.Allocator) Self {
        const controller = Self{
            .key_bindings = std.ArrayList(KeyBinding).init(allocator),
        };

        return controller;
    }

    pub fn deinit(self: *Self) void {
        self.key_bindings.deinit();
    }


    pub fn attach(self: *ControllerComponent, ecs: *ECSManager, eid: Core.EntityID) !void {
        if (ecs.control_system.active_controller_eid == null) {
            ecs.control_system.active_controller_eid = eid;
        }

        try ecs.controller_components.add(eid, self.*);
    }
};

pub const ControlSystem = struct {
    const Self = @This();

    world: *Core.World,
    globals: *GlobalsComponent,
    transform_components: *SparseSet(TransformComponent),
    rigid_body_components: *SparseSet(Collisions.RigidBodyComponent),
    controller_components: *SparseSet(ControllerComponent),
    active_controller_eid: ?Core.EntityID = null,
    collision_system: ?*Collisions.CollisionSystem = null,

    pub fn init(
        world: *Core.World,
        globals: *GlobalsComponent,
        transform_components: *SparseSet(TransformComponent),
        rigid_body_components: *SparseSet(Collisions.RigidBodyComponent),
        controller_components: *SparseSet(ControllerComponent),
    ) Self {
        return .{
            .world = world,
            .globals = globals,
            .transform_components = transform_components,
            .rigid_body_components = rigid_body_components,
            .controller_components = controller_components,
        };
    }

    pub fn update(self: *Self, dt: f64) void {
        if (self.globals.menu) return;

        if (self.active_controller_eid) |eid| {
            const ctrl = self.controller_components.get(eid).?;
            const tf = self.transform_components.get(eid).?;
            
            // In new architecture, rigid body components are on the same entity as controller
            const rigid_body = self.rigid_body_components.get(eid);

            for (ctrl.key_bindings.items) |kb| {
                if (self.globals.keys[@intCast(kb.key)]) {
                    std.debug.print("Key {} pressed for eid {d}\n", .{ kb.key, eid.id });
                    kb.onPressed(eid, tf, rigid_body, self.collision_system, @floatCast(dt));
                }
            }

            if (ctrl.mouse_move) |fnPtr| {
                if (self.globals.mouse_dx != 0 or self.globals.mouse_dy != 0)
                    fnPtr(eid, tf, rigid_body, self.collision_system, @floatCast(self.globals.mouse_dx), @floatCast(self.globals.mouse_dy), dt);
            }
        }

        self.globals.mouse_dx = 0;
        self.globals.mouse_dy = 0;
    }
};
