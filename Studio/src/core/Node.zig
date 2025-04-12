const std = @import("std");
const Mesh = @import("Mesh.zig");
const Math = @import("Math.zig");
const Pipeline = @import("Pipeline.zig");
const gl = @import("bindings/gl.zig");

const Scene = Pipeline.Scene;
const glad = gl.glad;
const Vec3 = Math.Vec3;
const Mat4 = Math.Mat4;
const Quaternion = Math.Quaternion;

const glCheckError = @import("Debug.zig").glCheckError;

const Self = @This();

pub const InstanceData = struct {
    position_buffer: u32,
    color_buffer: u32,
    count: usize,
};

pub const TextureUnit = enum(c_uint) {
    // Material property textures
    BaseColor = 0,
    NormalMap = 1,
    MetallicRoughness = 2,
    Occlusion = 3,
    Emissive = 4,
    Specular = 5,

    // Reserved for other uses
    Shadow = 6,
    Environment = 7,
    Irradiance = 8,
    LUT = 9,

    // Helper functions
    pub fn glValue(self: TextureUnit) c_uint {
        return @as(c_uint, @intCast(glad.GL_TEXTURE0)) + @intFromEnum(self);
    }

    pub fn index(self: TextureUnit) c_int {
        return @intCast(@intFromEnum(self));
    }
};

scene: ?*Scene = null,
mesh: ?*Mesh,
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
local_transform: Mat4 = Mat4.identity(),
world_transform: Mat4 = Mat4.identity(),

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

pub fn rotateWithQuaternion(self: *Self, q: Quaternion) void {
    self.rotation = self.rotation.multiply(q).normalize();
    self.updateLocalTransform();
}

pub fn rotateWithEuler(self: *Self, pitch: f32, yaw: f32, roll: f32) void {
    const delta_q = Quaternion.from_euler(pitch, yaw, roll);
    self.rotation = Quaternion.multiply(self.rotation, delta_q).normalize();

    self.updateLocalTransform();
}

pub fn translate(self: *Self, offsetLocal: Vec3) void {
    self.position[0] += offsetLocal.x();
    self.position[1] += offsetLocal.y();
    self.position[2] += offsetLocal.z();

    self.updateLocalTransform();
}

pub fn setPosition(self: *Self, x: f32, y: f32, z: f32) void {
    self.position = .{ x, y, z };
    self.updateLocalTransform();
}

pub fn setRotation(self: *Self, q: Quaternion) void {
    self.rotation = q.normalize();
    self.updateLocalTransform();
}

pub fn setRotationEuler(self: *Self, pitch: f32, yaw: f32, roll: f32) void {
    self.rotation = Quaternion.from_euler(pitch, yaw, roll).normalize();
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

    for (self.children.items) |child| {
        child.addSceneRecursively(scene);
    }
}

//TODO: Option to set origin to simplify calculations
fn updateLocalTransform(self: *Self) void {
    var transform = Mat4.identity();

    transform = transform.translate(-self.position[0], -self.position[1], -self.position[2]);

    // Apply scale
    transform = transform.scale(self.scale[0], self.scale[1], self.scale[2]);

    // Apply rotation around center
    const rotation_matrix = self.rotation.to_mat4();
    transform = transform.multiply(rotation_matrix);

    // Move back and translate to position
    const inv_center = Mat4.translation(self.position[0], self.position[1], self.position[2]);
    transform = transform.multiply(inv_center);
    transform = transform.translate(self.position[0], self.position[1], self.position[2]);

    self.local_transform = transform;
}

//TODO: Write-through cache or some flag to determine if this needs to be recoumpte
fn updateWorldTransform(self: *Self) void {
    if (self.parent) |parent| {
        self.world_transform = self.local_transform.multiply(parent.world_transform);
    } else {
        self.world_transform = self.local_transform;
    }
}

pub fn update(self: *Self) void {
    self.updateWorldTransform();

    if (self.mesh) |mesh| {
        const use_pbr = if (mesh.flags) |flags| flags.use_pbr else false;

        if (self.scene) |scene| {
            if ((scene.rendering_mode == .PBR and use_pbr) or
                (scene.rendering_mode == .Standard and !use_pbr))
            {
                // std.debug.print("Rendering mode: {s}\n", .{@tagName(scene.rendering_mode)});
                const model_loc = if (use_pbr) scene.pbr_uModelLoc else scene.uModelLoc;
                if (model_loc != -1) {
                    glad.glUniformMatrix4fv(model_loc, 1, glad.GL_FALSE, &self.world_transform.to_array());
                }

                if (use_pbr) {
                    if (scene.pbr_useTextureLoc != -1) {
                        const use_texture = if (mesh.flags) |flags|
                            @intFromBool(flags.use_texture)
                        else
                            0;
                        glad.glUniform1i(scene.pbr_useTextureLoc, use_texture);
                    }

                    const material = mesh.material;

                    // Base color factor
                    if (scene.pbr_baseColorFactorLoc != -1) {
                        glad.glUniform4fv(scene.pbr_baseColorFactorLoc, 1, &material.baseColorFactor);
                    }

                    // Metallic factor
                    if (scene.pbr_metallicFactorLoc != -1) {
                        glad.glUniform1f(scene.pbr_metallicFactorLoc, material.metallicFactor);
                    }

                    // Roughness factor
                    if (scene.pbr_roughnessFactorLoc != -1) {
                        glad.glUniform1f(scene.pbr_roughnessFactorLoc, material.roughnessFactor);
                    }

                    // Specular-glossiness extension on
                    if (scene.pbr_useSpecularGlossinessLoc != -1) {
                        // Check if using specular-glossiness based on whether diffuseFactor is not default
                        const use_sg = !std.mem.eql(f32, &material.diffuseFactor, &[_]f32{ 1.0, 1.0, 1.0, 1.0 });
                        glad.glUniform1i(scene.pbr_useSpecularGlossinessLoc, @intFromBool(use_sg));

                        if (use_sg) {
                            if (scene.pbr_diffuseFactorLoc != -1) {
                                glad.glUniform4fv(scene.pbr_diffuseFactorLoc, 1, &material.diffuseFactor);
                            }
                            if (scene.pbr_specularFactorLoc != -1) {
                                glad.glUniform3fv(scene.pbr_specularFactorLoc, 1, &material.specularFactor);
                            }
                            if (scene.pbr_glossinessFactorLoc != -1) {
                                glad.glUniform1f(scene.pbr_glossinessFactorLoc, material.glossinessFactor);
                            }
                        }
                    }

                    // Specular extension
                    if (scene.pbr_useSpecularExtensionLoc != -1) {
                        const use_specular = material.specularStrength > 0.0;
                        glad.glUniform1i(scene.pbr_useSpecularExtensionLoc, @intFromBool(use_specular));

                        if (use_specular) {
                            if (scene.pbr_specularStrengthLoc != -1) {
                                glad.glUniform1f(scene.pbr_specularStrengthLoc, material.specularStrength);
                            }
                            if (scene.pbr_specularColorFactorLoc != -1) {
                                glad.glUniform3fv(scene.pbr_specularColorFactorLoc, 1, &material.specularColor);
                            }
                        }
                    }

                    // Emissive properties
                    if (scene.pbr_emissiveFactorLoc != -1) {
                        glad.glUniform3fv(scene.pbr_emissiveFactorLoc, 1, &material.emissiveFactor);
                    }
                    if (scene.pbr_emissiveStrengthLoc != -1) {
                        glad.glUniform1f(scene.pbr_emissiveStrengthLoc, material.emissiveStrength);
                    }

                    // Alpha properties
                    if (scene.pbr_alphaCutoffLoc != -1) {
                        glad.glUniform1f(scene.pbr_alphaCutoffLoc, material.alphaCutoff);
                    }
                    if (scene.pbr_alphaModeLoc != -1) {
                        glad.glUniform1i(scene.pbr_alphaModeLoc, @intFromEnum(material.alphaMode));
                    }

                    // Set up textures
                    // Base color texture
                    const has_base_color = material.textures.baseColor != null;
                    if (scene.pbr_hasBaseColorTextureLoc != -1) {
                        glad.glUniform1i(scene.pbr_hasBaseColorTextureLoc, @intFromBool(has_base_color));
                    }

                    if (has_base_color) {
                        glad.glActiveTexture(TextureUnit.BaseColor.glValue());
                        glad.glBindTexture(glad.GL_TEXTURE_2D, material.textures.baseColor.?);
                    }

                    // Metallic-roughness texture
                    const has_metallic_roughness = material.textures.metallicRoughness != null;
                    if (scene.pbr_hasMetallicRoughnessTextureLoc != -1) {
                        glad.glUniform1i(scene.pbr_hasMetallicRoughnessTextureLoc, @intFromBool(has_metallic_roughness));
                    }

                    if (has_metallic_roughness) {
                        glad.glActiveTexture(TextureUnit.MetallicRoughness.glValue());
                        glad.glBindTexture(glad.GL_TEXTURE_2D, material.textures.metallicRoughness.?);
                    }

                    // Normal texture
                    const has_normal = material.textures.normal != null;
                    if (scene.pbr_hasNormalTextureLoc != -1) {
                        glad.glUniform1i(scene.pbr_hasNormalTextureLoc, @intFromBool(has_normal));
                    }

                    if (has_normal) {
                        glad.glActiveTexture(TextureUnit.NormalMap.glValue());
                        glad.glBindTexture(glad.GL_TEXTURE_2D, material.textures.normal.?);
                    }

                    // Occlusion texture
                    const has_occlusion = material.textures.occlusion != null;
                    if (scene.pbr_hasOcclusionTextureLoc != -1) {
                        glad.glUniform1i(scene.pbr_hasOcclusionTextureLoc, @intFromBool(has_occlusion));
                    }

                    if (has_occlusion) {
                        glad.glActiveTexture(TextureUnit.Occlusion.glValue());
                        glad.glBindTexture(glad.GL_TEXTURE_2D, material.textures.occlusion.?);
                    }

                    // // Emissive texture
                    const has_emissive = material.textures.emissive != null;
                    if (scene.pbr_hasEmissiveTextureLoc != -1) {
                        glad.glUniform1i(scene.pbr_hasEmissiveTextureLoc, @intFromBool(has_emissive));
                    }

                    if (has_emissive) {
                        glad.glActiveTexture(TextureUnit.Emissive.glValue());
                        glad.glBindTexture(glad.GL_TEXTURE_2D, material.textures.emissive.?);
                    }

                    // For alpha blending
                    if (material.alphaMode == .BLEND) {
                        glad.glEnable(glad.GL_BLEND);
                        glad.glBlendFunc(glad.GL_SRC_ALPHA, glad.GL_ONE_MINUS_SRC_ALPHA);
                    } else {
                        glad.glDisable(glad.GL_BLEND);
                    }

                    // For double-sided rendering
                    if (material.doubleSided) {
                        glad.glDisable(glad.GL_CULL_FACE);
                    } else {
                        glad.glEnable(glad.GL_CULL_FACE);
                    }
                } else {
                    if (scene.useTextureLoc != -1) {
                        const use_texture = if (mesh.flags) |flags|
                            @intFromBool(flags.use_texture)
                        else
                            0;
                        glad.glUniform1i(scene.useTextureLoc, use_texture);
                    }

                    if (mesh.flags) |flags| {
                        if (flags.use_texture) {
                            try self.bindTexture();
                        }
                    }
                }

                // In either case, draw the mesh
                mesh._draw(mesh);

                // Reset state
                glad.glDisable(glad.GL_BLEND);
                glad.glEnable(glad.GL_CULL_FACE);
            }
        }
    }

    for (self.children.items) |child| {
        child.update();
    }
}

fn bindTexture(self: *Self) !void {
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
