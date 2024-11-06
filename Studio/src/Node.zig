const std = @import("std");
const Mesh = @import("Mesh.zig");

const c = @cImport({
    @cInclude("glad/glad.h");
});

const Self = @This();

mesh: ?*Mesh,
update: ?*const fn (*Mesh, c.GLint) void,
children: std.ArrayList(*Self),

pub fn init(allocator: std.mem.Allocator, mesh: ?*Mesh) !Self {
    const update_fn = if (mesh) |mesh_pt| mesh_pt.draw else null;

    return Self{
        .mesh = mesh,
        .update = update_fn,
        .children = try std.ArrayList(*Self).initCapacity(allocator, 0),
    };
}

pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
    for (self.children.items) |child| {
        child.deinit(allocator);
    }
    self.children.deinit();
}

pub fn addChild(self: *Self, child: *Self) !void {
    try self.children.append(child);
}

pub fn update(self: *Self, uModelLoc: c.GLint) void {
    if (self.mesh) |mesh| {
        self.update(mesh, uModelLoc);
    }

    for (self.children.items) |child| {
        child.update(child.mesh, uModelLoc);
    }
}
