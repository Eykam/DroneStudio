const std = @import("std");
const SparseSet = @import("SparseSet.zig");

const Self = @This();

allocator: std.mem.Allocator,
components: std.ArrayList(SparseSet),

pub fn init(allocator: std.mem.Allocator) *Self {
    const self = try allocator.create(Self);
    self.* = Self{
        .allocator = allocator,
        .components = std.ArrayList(SparseSet()).init(allocator),
    };
    return self;
}

pub fn deinit() *Self {}

pub fn registerComponent() void {}
pub fn unregisterComponent() void {}

pub fn getComponents() void {}
pub fn getSystems() void {}
