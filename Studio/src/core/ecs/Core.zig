// src/ecs/Core.zig
const std = @import("std");
const ResourceManager = @import("ResourceManager.zig");

// A unique identifier for entities
pub const EntityID = struct {
    id: u64,

    pub fn init(id: u64) EntityID {
        return .{ .id = id };
    }
};

// Base component interface
pub const Component = struct {
    entity_id: EntityID,

    pub fn init(entity_id: EntityID) Component {
        return .{ .entity_id = entity_id };
    }
};

// The world containing all entities and systems
pub const World = struct {
    allocator: std.mem.Allocator,
    entities: std.AutoHashMap(u64, EntityID),
    next_entity_id: u64,
    resource_manager: *ResourceManager,

    pub fn init(allocator: std.mem.Allocator, resource_manager: *ResourceManager) !*World {
        const world = try allocator.create(World);
        world.* = .{
            .allocator = allocator,
            .entities = std.AutoHashMap(u64, EntityID).init(allocator),
            .next_entity_id = 0,
            .resource_manager = resource_manager,
        };
        return world;
    }

    pub fn deinit(self: *World) void {
        self.entities.deinit();
    }

    pub fn createEntity(self: *World) !EntityID {
        const entity_id = EntityID.init(self.next_entity_id);
        self.next_entity_id += 1;
        try self.entities.put(entity_id.id, entity_id);
        return entity_id;
    }

    pub fn destroyEntity(self: *World, entity_id: EntityID) void {
        _ = self.entities.remove(entity_id.id);
    }
};
