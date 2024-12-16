const std = @import("std");
const Mesh = @import("Mesh.zig");
const Math = @import("Math.zig");
const Scene = @import("Pipeline.zig").Scene;
const gl = @import("gl.zig");
const Quaternion = Math.Quaternion;
const Vec3 = Math.Vec3;
const glad = gl.glad;
const glCheckError = @import("Debug.zig").glCheckError;

const Self = @This();

scene: ?*Scene = null,
mesh: ?*Mesh,
_update: ?*const fn (*Mesh) void,
allocator: std.mem.Allocator,
children: std.ArrayList(*Self),
parent: ?*Self = null,

y: ?[]u8 = null,
uv: ?[]u8 = null,
yTextureUnit: c_int = 0,
uvTextureUnit: c_int = 0,
width: ?c_int = null,
height: ?c_int = null,
texture_updated: bool = false,

// Transformation properties
position: [3]f32 = .{ 0, 0, 0 },
rotation: Quaternion = Quaternion.identity(),
scale: [3]f32 = .{ 1, 1, 1 },
local_transform: [16]f32 = Math.identity(),
world_transform: [16]f32 = Math.identity(),

pub fn init(allocator: std.mem.Allocator, mesh_opt: ?Mesh) !*Self {

    // If a mesh was provided, allocate space for it
    var mesh_ptr: ?*Mesh = null;
    var _update: ?*const fn (*Mesh) void = null;

    if (mesh_opt) |mesh_val| {
        const mesh_storage = try allocator.create(Mesh);
        mesh_storage.* = mesh_val;
        mesh_ptr = mesh_storage;
        _update = mesh_val.draw;
    }

    const node_ptr = try allocator.create(Self);

    node_ptr.* = Self{
        .allocator = allocator,
        .mesh = mesh_ptr,
        ._update = _update,
        .children = try std.ArrayList(*Self).initCapacity(allocator, 0),
    };

    if (mesh_ptr) |mesh| {
        mesh.node = node_ptr;
        glad.glGenTextures(1, &mesh.textureID.y);
        glad.glGenTextures(1, &mesh.textureID.uv);
    }

    return node_ptr;
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

    self.allocator.destroy(self);
}

pub fn setPosition(self: *Self, x: f32, y: f32, z: f32) void {
    self.position = .{ x, y, z };
    self.updateLocalTransform();
}

pub fn setRotation(self: *Self, q: Quaternion) void {
    self.rotation = Quaternion.normalize(q);
    self.updateLocalTransform();
}

pub fn setRotationEuler(self: *Self, pitch: f32, yaw: f32, roll: f32) void {
    self.rotation = Quaternion.normalize(Quaternion.fromEuler(pitch, yaw, roll));
    self.updateLocalTransform();
}

pub fn setScale(self: *Self, x: f32, y: f32, z: f32) void {
    self.scale = .{ x, y, z };
    self.updateLocalTransform();
}

pub fn addChild(self: *Self, child: *Self) !void {
    child.parent = self;

    if (self.scene) |scene| {
        child.addSceneRecursively(scene);
    }

    try self.children.append(child);
}

pub fn addSceneRecursively(self: *Self, scene: *Scene) void {
    self.scene = scene;

    self.yTextureUnit = scene.texGen.generateID();
    self.uvTextureUnit = scene.texGen.generateID();
    std.debug.print("Setting y: {d}, uv: {d}\n", .{ self.yTextureUnit, self.uvTextureUnit });

    for (self.children.items) |child| {
        child.addSceneRecursively(scene);
    }
}

// (Scale -> Rotate -> Translate)
fn updateLocalTransform(self: *Self) void {
    var transform = Math.identity();

    const center_transform = Math.translate(transform, -self.position[0], -self.position[1], -self.position[2]);
    transform = center_transform;

    // Apply scale
    transform = Math.scale(transform, self.scale[0], self.scale[1], self.scale[2]);

    // Apply rotation around center
    const rotation_matrix = Quaternion.toMatrix(self.rotation);
    transform = Math.multiply_matrices(transform, rotation_matrix);

    // Move back and translate to position
    const inv_center = Math.translate(Math.identity(), self.position[0], self.position[1], self.position[2]);
    transform = Math.multiply_matrices(transform, inv_center);
    transform = Math.translate(transform, self.position[0], self.position[1], self.position[2]);

    self.local_transform = transform;
}

fn updateWorldTransform(self: *Self) void {
    if (self.parent) |parent| {
        // Combine parent's world transform with our local transform
        self.world_transform = Math.multiply_matrices(parent.world_transform, self.local_transform);
    } else {
        // Root node - world transform is the same as local transform
        self.world_transform = self.local_transform;
    }
}

pub fn update(self: *Self, uModelLoc: glad.GLint) void {
    self.updateWorldTransform();

    if (self.mesh) |mesh| {
        // Set mesh-specific uniforms
        if (uModelLoc != -1) {
            glad.glUniformMatrix4fv(uModelLoc, 1, glad.GL_FALSE, &self.world_transform);
        }

        if (self._update) |_update| {
            try self.bindTexture();
            _update(mesh);
        }
    }

    for (self.children.items) |child| {
        child.update(uModelLoc);
    }
}

pub fn bindTexture(self: *Self) !void {
    // Generate texture objects if not already created
    if (self.texture_updated) {
        const mesh = self.*.mesh.?;

        // Bind and configure Y plane texture
        glad.glActiveTexture(@intCast(glad.GL_TEXTURE0 + self.yTextureUnit));
        glad.glBindTexture(glad.GL_TEXTURE_2D, mesh.textureID.y);
        glad.glTexImage2D(glad.GL_TEXTURE_2D, 0, glad.GL_R8, self.width.?, self.height.?, 0, glad.GL_RED, glad.GL_UNSIGNED_BYTE, self.y.?.ptr);
        glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MIN_FILTER, glad.GL_LINEAR);
        glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MAG_FILTER, glad.GL_LINEAR);

        // Bind and configure interleaved UV plane texture
        glad.glActiveTexture(@intCast(glad.GL_TEXTURE0 + self.uvTextureUnit));
        glad.glBindTexture(glad.GL_TEXTURE_2D, mesh.textureID.uv);
        glad.glTexImage2D(glad.GL_TEXTURE_2D, 0, glad.GL_RG8, @divTrunc(self.width.?, 2), @divTrunc(self.height.?, 2), 0, glad.GL_RG, glad.GL_UNSIGNED_BYTE, self.uv.?.ptr);
        glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MIN_FILTER, glad.GL_LINEAR);
        glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MAG_FILTER, glad.GL_LINEAR);

        if (self.scene) |scene| {
            glad.glUniform1i(scene.yTextureLoc, self.yTextureUnit);
            glad.glUniform1i(scene.uvTextureLoc, self.uvTextureUnit);
        }
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
