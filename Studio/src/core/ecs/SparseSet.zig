// src/ecs/SparseSet.zig
const std = @import("std");
const Core = @import("Core.zig");
const EntityID = Core.EntityID;

pub fn SparseSet(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const ComponentEntry = struct {
            entity_id: EntityID,
            component: T,
        };

        pub const IteratorType = struct {
            entity_id: EntityID,
            component: *T,
        };

        allocator: std.mem.Allocator,
        sparse: std.AutoHashMap(u64, usize),
        dense: std.ArrayList(ComponentEntry),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .sparse = std.AutoHashMap(u64, usize).init(allocator),
                .dense = std.ArrayList(ComponentEntry).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.sparse.deinit();
            self.dense.deinit();
        }

        pub fn add(self: *Self, entity_id: EntityID, component: T) !void {
            if (self.sparse.get(entity_id.id)) |_| {
                return error.ComponentAlreadyExists;
            }

            const index = self.dense.items.len;
            try self.sparse.put(entity_id.id, index);
            try self.dense.append(.{
                .entity_id = entity_id,
                .component = component,
            });
        }

        pub fn remove(self: *Self, entity_id: EntityID) bool {
            if (self.sparse.get(entity_id.id)) |index| {
                // Swap with the last element to maintain dense array
                const last_index = self.dense.items.len - 1;
                if (index != last_index) {
                    const last_entity_id = self.dense.items[last_index].entity_id;
                    self.dense.items[index] = self.dense.items[last_index];
                    try self.sparse.put(last_entity_id.id, index);
                }

                _ = self.sparse.remove(entity_id.id);
                _ = self.dense.pop();
                return true;
            }
            return false;
        }

        pub fn get(self: Self, entity_id: EntityID) ?*T {
            if (self.sparse.get(entity_id.id)) |index| {
                return &self.dense.items[index].component;
            }
            return null;
        }

        pub fn has(self: Self, entity_id: EntityID) bool {
            return self.sparse.contains(entity_id.id);
        }

        pub fn iterator(self: *Self) ComponentIterator(T, IteratorType) {
            return .{
                .sparse_set = self,
                .index = 0,
            };
        }
    };
}

pub fn ComponentIterator(comptime T: type, comptime TEntry: type) type {
    return struct {
        const Self = @This();

        sparse_set: *SparseSet(T),
        index: usize,

        pub fn next(self: *Self) ?TEntry {
            if (self.index < self.sparse_set.dense.items.len) {
                const item = &self.sparse_set.dense.items[self.index];
                self.index += 1;
                return TEntry{ .entity_id = item.entity_id, .component = &item.component };
            }
            return null;
        }
    };
}
