const std = @import("std");
const Mesh = @import("Mesh.zig");

const c = @cImport({
    @cInclude("glad/glad.h");
});

const Self = @This();

mesh: ?*Mesh,
update_mesh: ?*const fn (*Mesh, c.GLint) void,
allocator: std.mem.Allocator,
children: std.ArrayList(*Self),

pub fn init(allocator: std.mem.Allocator, mesh_opt: ?Mesh) !Self {

    // If a mesh was provided, allocate space for it
    var mesh_ptr: ?*Mesh = null;
    var update_fn: ?*const fn (*Mesh, c.GLint) void = null;

    if (mesh_opt) |mesh_val| {
        const mesh_storage = try allocator.create(Mesh);
        mesh_storage.* = mesh_val;
        mesh_ptr = mesh_storage;
        update_fn = mesh_val.draw;
    }

    const node = Self{
        .allocator = allocator,
        .mesh = mesh_ptr,
        .update_mesh = update_fn,
        .children = try std.ArrayList(*Self).initCapacity(allocator, 0),
    };

    return node;
}

pub fn deinit(self: *Self) void {
    // Free children recursively
    for (self.children.items) |child| {
        child.deinit();
    }
    self.children.deinit();

    // Free the mesh if it exists
    if (self.mesh) |mesh| {
        self.allocator.destroy(mesh);
    }

    // Finally free ourselves
    self.allocator.destroy(self);
}

pub fn addChild(self: *Self, child: *Self) !void {
    try self.children.append(child);
}

pub fn update(self: *Self, uModelLoc: c.GLint) void {
    if (self.mesh) |mesh| {
        if (self.update_mesh) |update_fn| {
            update_fn(mesh, uModelLoc);
        }
    }

    for (self.children.items) |child| {
        child.update(uModelLoc);
    }
}

pub fn debug(self: *Self) void {
    if (self.mesh) |mesh| {
        mesh.debug();
    }

    for (self.children.items) |node| {
        if (node.mesh) |mesh| {
            mesh.debug();
        }
    }
}
