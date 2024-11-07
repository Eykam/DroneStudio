const std = @import("std");
const Mesh = @import("Mesh.zig");
const Transformations = @import("Transformations.zig");
const Vec3 = Transformations.Vec3;

const c = @cImport({
    @cInclude("glad/glad.h");
});

const Self = @This();

mesh: ?*Mesh,
_update: ?*const fn (*Mesh) void,
allocator: std.mem.Allocator,
children: std.ArrayList(*Self),
parent: ?*Self = null,

// Add transformation properties
position: [3]f32 = .{ 0, 0, 0 },
rotation: [3]f32 = .{ 0, 0, 0 }, // Euler angles in radians
scale: [3]f32 = .{ 1, 1, 1 },
local_transform: [16]f32 = Transformations.identity(),
world_transform: [16]f32 = Transformations.identity(),

pub fn init(allocator: std.mem.Allocator, mesh_opt: ?Mesh) !Self {

    // If a mesh was provided, allocate space for it
    var mesh_ptr: ?*Mesh = null;
    var _update: ?*const fn (*Mesh) void = null;

    if (mesh_opt) |mesh_val| {
        const mesh_storage = try allocator.create(Mesh);
        mesh_storage.* = mesh_val;
        mesh_ptr = mesh_storage;
        _update = mesh_val.draw;
    }

    const node = Self{
        .allocator = allocator,
        .mesh = mesh_ptr,
        ._update = _update,
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

pub fn setPosition(self: *Self, x: f32, y: f32, z: f32) void {
    self.position = .{ x, y, z };
    self.updateLocalTransform();
}

pub fn setRotation(self: *Self, x: f32, y: f32, z: f32) void {
    self.rotation = .{ x, y, z };
    self.updateLocalTransform();
}

pub fn setScale(self: *Self, x: f32, y: f32, z: f32) void {
    self.scale = .{ x, y, z };
    self.updateLocalTransform();
}

pub fn addChild(self: *Self, child: *Self) !void {
    child.parent = self;
    try self.children.append(child);
}

fn updateLocalTransform(self: *Self) void {
    // (Scale -> Rotate -> Translate)
    var transform = Transformations.identity();

    const center_transform = Transformations.translate(transform, -self.position[0], -self.position[1], -self.position[2]);
    transform = center_transform;

    // Apply scale
    transform = Transformations.scale(transform, self.scale[0], self.scale[1], self.scale[2]);

    // Apply rotation around center
    transform = Transformations.rotate(transform, -self.rotation[0], Vec3{ .x = 1, .y = 0, .z = 0 });
    transform = Transformations.rotate(transform, -self.rotation[1], Vec3{ .x = 0, .y = 1, .z = 0 });
    transform = Transformations.rotate(transform, -self.rotation[2], Vec3{ .x = 0, .y = 0, .z = 1 });

    // Move back and translate to position
    const inv_center = Transformations.translate(Transformations.identity(), self.position[0], self.position[1], self.position[2]);
    transform = Transformations.multiply_matrices(transform, inv_center);
    transform = Transformations.translate(transform, self.position[0], self.position[1], self.position[2]);

    self.local_transform = transform;
}

fn updateWorldTransform(self: *Self) void {
    if (self.parent) |parent| {
        // Combine parent's world transform with our local transform
        self.world_transform = Transformations.multiply_matrices(parent.world_transform, self.local_transform);
    } else {
        // Root node - world transform is the same as local transform
        self.world_transform = self.local_transform;
    }
}

pub fn update(self: *Self, uModelLoc: c.GLint) void {
    self.updateWorldTransform();

    if (self.mesh) |mesh| {
        // Set mesh-specific uniforms
        if (uModelLoc != -1) {
            c.glUniformMatrix4fv(uModelLoc, 1, c.GL_FALSE, &self.world_transform);
        }

        if (self._update) |_update| {
            _update(mesh);
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
