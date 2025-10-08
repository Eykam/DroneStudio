// src/ecs/components/Transform.zig
const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const ECSManager = @import("../ECSManager.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;

const Vec3 = Math.Vec3;
const Quaternion = Math.Quaternion;
const Mat4 = Math.Mat4;

pub const TransformComponent = struct {
    const Self = @This();

    position: [3]f32 = .{ 0, 0, 0 },
    rotation: Quaternion = Quaternion.identity(),
    scale: [3]f32 = .{ 1, 1, 1 },
    local_transform: Mat4 = Mat4.identity(),
    world_transform: Mat4 = Mat4.identity(),
    parent: ?Core.EntityID = null,
    children: std.ArrayList(Core.EntityID),
    changed_this_frame: bool = true,

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .children = std.ArrayList(Core.EntityID).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.children.deinit();
    }

    fn markDirty(self: *Self) void {
        self.changed_this_frame = true;
    }

    fn resetFlag(self: *Self) void {
        self.changed_this_frame = false;
    }

    pub fn setPosition(self: *Self, x: f32, y: f32, z: f32) void {
        self.position = .{ x, y, z };
        self.updateLocalTransform();
    }

    pub fn setRotation(self: *Self, q: Quaternion) void {
        self.rotation = q.normalize();
        self.updateLocalTransform();
    }

    pub fn setScale(self: *Self, x: f32, y: f32, z: f32) void {
        self.scale = .{ x, y, z };
        self.updateLocalTransform();
    }

    pub fn translate(self: *Self, translation: [3]f32) void {
        self.position = .{
            self.position[0] + translation[0],
            self.position[1] + translation[1],
            self.position[2] + translation[2],
        };
        self.updateLocalTransform();
    }

    pub fn rotate(self: *Self, q: Quaternion) void {
        self.rotation = self.rotation.multiply(q).normalize();
        self.updateLocalTransform();
    }

    pub fn rotateWithEuler(self: *Self, pitch: f32, yaw: f32, roll: f32) void {
        const q = Quaternion.from_euler(pitch, yaw, roll);
        self.rotation = self.rotation.multiply(q).normalize();
        self.updateLocalTransform();
    }

    pub fn updateLocalTransform(self: *Self) void {
        var transform = Mat4.identity();

        transform = transform.multiply(Mat4.scaling(self.scale[0], self.scale[1], self.scale[2]));
        transform = transform.multiply(self.rotation.to_mat4());
        transform = transform.multiply(Mat4.translation(self.position[0], self.position[1], self.position[2]));

        self.local_transform = transform;
        self.markDirty();
    }

    pub fn attach(self: *TransformComponent, ecs: *ECSManager, eid: Core.EntityID) !void {
        try ecs.transform_components.add(eid, self.*);
    }
};

pub const TransformSystem = struct {
    const Self = @This();

    world: *Core.World,
    transform_components: *SparseSet(TransformComponent),

    pub fn init(world: *Core.World, transform_components: *SparseSet(TransformComponent)) Self {
        return .{
            .world = world,
            .transform_components = transform_components,
        };
    }

    pub fn update(self: *Self) void {
        self.updateWorldTransformsRecursive(null);
    }

    fn updateWorldTransformsRecursive(self: *Self, parent_id: ?Core.EntityID) void {
        var transform_iter = self.transform_components.iterator();

        while (transform_iter.next()) |tuple| {
            const entity_id = tuple.entity_id;
            const transform = tuple.component;

            // If this is a root entity (no parent) or the direct child of the current parent
            if ((parent_id == null and transform.parent == null) or
                (parent_id != null and transform.parent != null and transform.parent.?.id == parent_id.?.id))
            {

                // Update world transform based on parent
                if (transform.parent) |parent_entity_id| {
                    if (self.transform_components.get(parent_entity_id)) |parent_transform| {
                        transform.world_transform = transform.local_transform.multiply(parent_transform.world_transform);
                    } else {
                        transform.world_transform = transform.local_transform;
                    }
                } else {
                    transform.world_transform = transform.local_transform;
                }

                // Recursively update all children
                self.updateWorldTransformsRecursive(entity_id);
                // transform.resetFlag();
            }
        }
    }

    pub fn addChild(self: *Self, parent_id: Core.EntityID, child_id: Core.EntityID) !void {
        if (self.transform_components.get(parent_id)) |parent_transform| {
            if (self.transform_components.get(child_id)) |child_transform| {
                // Check if child already has a different parent
                if (child_transform.parent != null and child_transform.parent.?.id != parent_id.id) {
                    if (self.transform_components.get(child_transform.parent.?)) |old_parent| {
                        // Remove from old parent's children list
                        for (old_parent.children.items, 0..) |id, i| {
                            if (id.id == child_id.id) {
                                _ = old_parent.children.swapRemove(i);
                                break;
                            }
                        }
                    }
                }

                // Set new parent
                child_transform.parent = parent_id;

                // Add to parent's children list if not already there
                var already_child = false;
                for (parent_transform.children.items) |id| {
                    if (id.id == child_id.id) {
                        already_child = true;
                        break;
                    }
                }

                if (!already_child) {
                    try parent_transform.children.append(child_id);
                }

                // child_transform.updateLocalTransform();
            }
        }
    }

    pub fn getParent(self: *Self, entity_id: Core.EntityID) ?Core.EntityID {
        if (self.transform_components.get(entity_id)) |transform| {
            return transform.parent;
        }
        return null;
    }
};
