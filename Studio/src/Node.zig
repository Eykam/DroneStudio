const std = @import("std");
const Mesh = @import("Mesh.zig");
const Math = @import("Math.zig");
const Scene = @import("Pipeline.zig").Scene;
const gl = @import("bindings/gl.zig");
const Quaternion = Math.Quaternion;
const Vec3 = Math.Vec3;
const glad = gl.glad;
const glCheckError = @import("Debug.zig").glCheckError;

const Self = @This();

pub const InstanceData = struct {
    position_buffer: u32,
    color_buffer: u32,
    count: usize,
};

scene: ?*Scene = null,
mesh: ?*Mesh,
_update: ?*const fn (*Mesh) void,
arena: *std.heap.ArenaAllocator,
backing_allocator: std.mem.Allocator,
allocator: std.mem.Allocator,
children: std.ArrayList(*Self),
parent: ?*Self = null,

mutex: std.Thread.Mutex = std.Thread.Mutex{},

yTextureUnit: c_int = 0,
uvTextureUnit: c_int = 0,
depthTextureUnit: c_int = 0,
width: ?c_int = null,
height: ?c_int = null,
texture_updated: bool = false,
instance_data: ?InstanceData = null,

// Transformation properties
position: [3]f32 = .{ 0, 0, 0 },
rotation: Quaternion = Quaternion.identity(),
scale: [3]f32 = .{ 1, 1, 1 },
local_transform: [16]f32 = Math.identity(),
world_transform: [16]f32 = Math.identity(),

pub fn init(allocator: std.mem.Allocator, _vertices: ?[]Mesh.Vertex, _indices: ?[]u32, draw: ?Mesh.draw) !*Self {
    var node_arena = try allocator.create(std.heap.ArenaAllocator);
    node_arena.* = std.heap.ArenaAllocator.init(allocator);
    const node_allocator = node_arena.allocator();

    var mesh_ptr: ?*Mesh = null;
    var draw_fn: ?Mesh.draw = draw;

    if (_vertices) |vertices| {
        mesh_ptr = try Mesh.init(node_allocator, vertices, _indices, draw);
        draw_fn = mesh_ptr.?._draw;
    }

    const node_ptr = try node_allocator.create(Self);

    node_ptr.* = Self{
        .arena = node_arena,
        .backing_allocator = allocator,
        .allocator = node_allocator,
        .mesh = mesh_ptr,
        ._update = draw_fn,
        .children = try std.ArrayList(*Self).initCapacity(node_allocator, 0),
    };

    if (mesh_ptr) |mesh| {
        mesh.node = node_ptr;
    }

    return node_ptr;
}

pub fn deinit(self: *Self) void {
    const backing_allocator = self.backing_allocator;
    const arena = self.arena;

    for (self.children.items) |child| {
        child.deinit();
    }

    if (self.mesh) |mesh| {
        glad.glDeleteTextures(1, &mesh.textureID.y);
        glad.glDeleteTextures(1, &mesh.textureID.uv);
        mesh.deinit();
    }

    arena.deinit();
    backing_allocator.destroy(arena);
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
    self.depthTextureUnit = scene.texGen.generateID();
    // std.debug.print("Setting y: {d}, uv: {d}\n", .{ self.yTextureUnit, self.uvTextureUnit });

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

pub fn update(self: *Self) void {
    self.updateWorldTransform();

    if (self.mesh) |mesh| {
        // Set mesh-specific uniforms
        if (self.scene) |scene| {
            if (scene.uModelLoc != -1) {
                glad.glUniformMatrix4fv(scene.uModelLoc, 1, glad.GL_FALSE, &self.world_transform);
            }

            if (self._update) |_update| {
                try self.bindTexture();
                _update(mesh);
            }
        }
    }

    for (self.children.items) |child| {
        child.update();
    }
}

pub fn bindTexture(self: *Self) !void {
    if (!self.texture_updated) return;

    const mesh = self.*.mesh.?;

    glad.glActiveTexture(@intCast(glad.GL_TEXTURE0 + self.yTextureUnit));
    glad.glBindTexture(glad.GL_TEXTURE_2D, mesh.textureID.y);

    glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MIN_FILTER, glad.GL_LINEAR);
    glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MAG_FILTER, glad.GL_LINEAR);
    glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_WRAP_S, glad.GL_CLAMP_TO_EDGE);
    glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_WRAP_T, glad.GL_CLAMP_TO_EDGE);

    glad.glActiveTexture(@intCast(glad.GL_TEXTURE0 + self.uvTextureUnit));
    glad.glBindTexture(glad.GL_TEXTURE_2D, mesh.textureID.uv);

    glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MIN_FILTER, glad.GL_LINEAR);
    glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MAG_FILTER, glad.GL_LINEAR);
    glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_WRAP_S, glad.GL_CLAMP_TO_EDGE);
    glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_WRAP_T, glad.GL_CLAMP_TO_EDGE);

    glad.glActiveTexture(@intCast(glad.GL_TEXTURE0 + self.depthTextureUnit));
    glad.glBindTexture(glad.GL_TEXTURE_2D, mesh.textureID.depth);

    glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MIN_FILTER, glad.GL_LINEAR);
    glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MAG_FILTER, glad.GL_LINEAR);
    glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_WRAP_S, glad.GL_CLAMP_TO_EDGE);
    glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_WRAP_T, glad.GL_CLAMP_TO_EDGE);

    if (self.scene) |scene| {
        glad.glUniform1i(scene.yTextureLoc, self.yTextureUnit);
        glad.glUniform1i(scene.uvTextureLoc, self.uvTextureUnit);
        glad.glUniform1i(scene.depthTextureLoc, self.depthTextureUnit);
    }

    const err = glad.glGetError();
    if (err != glad.GL_NO_ERROR) {
        std.debug.print("GL Error in bindTexture: {}\n", .{err});
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
