// src/ecs/ResourceManager.zig
const std = @import("std");
const Mesh = @import("../Mesh.zig");
const Math = @import("../Math.zig");
const gl = @import("../bindings/gl.zig");
const ImageLoader = @import("../Image.zig");
const GLTFParser = @import("../GLTF.zig");
const GLTF = GLTFParser.GLTF;
const glad = gl.glad;

pub const PhongData = struct {
    ambientColor: [3]f32 = .{ 0.1, 0.1, 0.1 },
    diffuseColor: [4]f32 = .{ 1.0, 1.0, 1.0, 1.0 },
    specularColor: [3]f32 = .{ 1.0, 1.0, 1.0 },
    shininess: f32 = 32.0,

    diffuseTexture: ?c_uint = null,
    specularTexture: ?c_uint = null,
    normalTexture: ?c_uint = null,
};

/// For a PBR material, store typical metallic/roughness, normal, etc.
pub const PBRData = struct {
    baseColorFactor: [4]f32 = .{ 1.0, 1.0, 1.0, 1.0 },
    metallicFactor: f32 = 1.0,
    roughnessFactor: f32 = 1.0,

    // Specular-glossiness extension
    diffuseFactor: [4]f32 = .{ 1.0, 1.0, 1.0, 1.0 },
    specularFactor: [3]f32 = .{ 1.0, 1.0, 1.0 },
    glossinessFactor: f32 = 1.0,

    // Emissive parameters
    emissiveFactor: [3]f32 = .{ 0.0, 0.0, 0.0 },
    emissiveStrength: f32 = 1.0,

    // KHR_materials_specular extension
    specularColor: [3]f32 = .{ 1.0, 1.0, 1.0 },
    specularStrength: f32 = 0.0,

    // Other material properties
    doubleSided: bool = false,
    alphaMode: Mesh.AlphaMode = .OPAQUE,
    alphaCutoff: f32 = 0.5,

    pub fn format(
        self: @This(),
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "\nPBR Data (\n\tFactors: (\n\t\tbaseColorFactor: {d:.4},\n\t\tmetallicFactor: {d:.4},\n\t\troughnessFactor: {d:.4}\n\t),\n",
            .{ self.baseColorFactor, self.metallicFactor, self.roughnessFactor },
        );
        try writer.print(
            "\tSpecular-Glossiness: (\n\t\tdiffuseFactor: {d:.4},\n\t\tspecularFactor: {d:.4},\n\t\tglossinessFactor: {d:.4}\n\t),\n",
            .{ self.diffuseFactor, self.specularFactor, self.glossinessFactor },
        );
        try writer.print(
            "\tSpecular: (\n\t\tspecularColor: {d:.4},\n\t\tspecularStrength: {d:.4}\n\t)\n",
            .{ self.specularColor, self.specularStrength },
        );
        try writer.print(
            "\tEmission: (\n\t\temissiveFactor: {d:.4},\n\t\temissiveStrength: {d:.4}\n\t),\n",
            .{ self.emissiveFactor, self.emissiveStrength },
        );
        try writer.print(
            "\tOther Parameters: (\n\t\tdoubleSided: {any},\n\t\talphaMode: {s},\n\t\talphaCutoff: {d:.4}\n\t)\n",
            .{ self.doubleSided, @tagName(self.alphaMode), self.alphaCutoff },
        );
    }
};

pub const PBRTextures = struct {
    baseColor: ?c_uint = null,
    normal: ?c_uint = null,
    metallicRoughness: ?c_uint = null,
    occlusion: ?c_uint = null,
    emissive: ?c_uint = null,
    specular: ?c_uint = null,

    pub fn format(
        self: PBRTextures,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("PBRTextures (\n", .{});

        try writer.print("\tbaseColor: ", .{});
        if (self.baseColor) |texture| {
            try writer.print("{any}", .{texture});
        } else {
            try writer.print("null", .{});
        }
        try writer.print(",\n", .{});

        try writer.print("\tnormal: ", .{});
        if (self.normal) |texture| {
            try writer.print("{any}", .{texture});
        } else {
            try writer.print("null", .{});
        }
        try writer.print(",\n", .{});

        try writer.print("\tmetallicRoughness: ", .{});
        if (self.metallicRoughness) |texture| {
            try writer.print("{any}", .{texture});
        } else {
            try writer.print("null", .{});
        }
        try writer.print(",\n", .{});

        try writer.print("\tocclusion: ", .{});
        if (self.occlusion) |texture| {
            try writer.print("{any}", .{texture});
        } else {
            try writer.print("null", .{});
        }
        try writer.print(",\n", .{});

        try writer.print("\temissive: ", .{});
        if (self.emissive) |texture| {
            try writer.print("{any}", .{texture});
        } else {
            try writer.print("null", .{});
        }
        try writer.print(",\n", .{});

        try writer.print("\tspecular: ", .{});
        if (self.specular) |texture| {
            try writer.print("{any}", .{texture});
        } else {
            try writer.print("null", .{});
        }
        try writer.print("\n)\n", .{});
    }
};

pub const MaterialType = enum {
    Phong,
    PBR,
};

pub fn Material(T: MaterialType) type {
    return struct {
        matType: MaterialType = T,

        data: switch (T) {
            .Phong => PhongData,
            .PBR => PBRData,
        } = .{},

        textures: switch (T) {
            .Phong => PBRTextures,
            .PBR => PBRTextures,
        } = .{},
    };
}

pub const MaterialVariant = union(MaterialType) {
    Phong: Material(.Phong),
    PBR: Material(.PBR),

    pub fn getType(self: @This()) MaterialType {
        return switch (self) {
            .Phong => .Phong,
            .PBR => .PBR,
        };
    }
};

pub const UniformValue = union(enum) {
    Int: i32,
    Float: f32,
    Vec2: [2]f32,
    Vec3: [3]f32,
    Vec4: [4]f32,
    Mat4: [16]f32,

    pub fn setToShader(self: UniformValue, location: c_int) void {
        switch (self) {
            .Int => |v| glad.glUniform1i(location, v),
            .Float => |v| glad.glUniform1f(location, v),
            .Vec2 => |v| glad.glUniform2fv(location, 1, &v),
            .Vec3 => |v| glad.glUniform3fv(location, 1, &v),
            .Vec4 => |v| glad.glUniform4fv(location, 1, &v),
            .Mat4 => |v| glad.glUniformMatrix4fv(location, 1, glad.GL_FALSE, &v),
        }
    }
};

pub const MeshResource = struct {
    mesh: *Mesh,
    instance_count: usize,

    pub fn init(mesh: *Mesh) MeshResource {
        return .{
            .mesh = mesh,
            .instance_count = 0,
        };
    }
};

pub const TextureResource = struct {
    texture_id: c_uint,
    ref_count: usize,
    width: u32 = 0,
    height: u32 = 0,
    channels: u32 = 0,

    pub fn init(texture_id: c_uint, width: u32, height: u32, channels: u32) TextureResource {
        return .{
            .texture_id = texture_id,
            .ref_count = 1,
            .width = width,
            .height = height,
            .channels = channels,
        };
    }
};

pub const ShaderResource = struct {
    program_id: c_uint,
    ref_count: usize,
    uniforms: std.StringHashMap(c_int),

    pub fn init(allocator: std.mem.Allocator, program_id: c_uint) !ShaderResource {
        return .{
            .program_id = program_id,
            .ref_count = 1,
            .uniforms = std.StringHashMap(c_int).init(allocator),
        };
    }

    pub fn getorPutUniformLocation(self: *ShaderResource, name: []const u8) c_int {
        if (self.uniforms.get(name)) |location| {
            return location;
        }

        const location = glad.glGetUniformLocation(self.program_id, @ptrCast(name.ptr));
        if (location != -1) {
            self.uniforms.put(name, location) catch {};
        }

        return location;
    }

    pub fn cacheCommonUniforms(self: *ShaderResource, shader_type: MaterialType) void {
        _ = self.getorPutUniformLocation("uModel");
        _ = self.getorPutUniformLocation("uView");
        _ = self.getorPutUniformLocation("uProjection");
        _ = self.getorPutUniformLocation("uNormalMatrix");

        _ = self.getorPutUniformLocation("viewPos");
        _ = self.getorPutUniformLocation("ambientLight");
        _ = self.getorPutUniformLocation("useTexture");

        switch (shader_type) {
            .PBR => {
                // Cache PBR uniform locations
                _ = self.getorPutUniformLocation("baseColorFactor");
                _ = self.getorPutUniformLocation("metallicFactor");
                _ = self.getorPutUniformLocation("roughnessFactor");
                _ = self.getorPutUniformLocation("emissiveFactor");
                _ = self.getorPutUniformLocation("emissiveStrength");
                _ = self.getorPutUniformLocation("alphaCutoff");
                _ = self.getorPutUniformLocation("alphaModeEnum");
                _ = self.getorPutUniformLocation("doubleSided");
                _ = self.getorPutUniformLocation("specularColor");
                _ = self.getorPutUniformLocation("specularStrength");

                // Texture flags
                _ = self.getorPutUniformLocation("hasBaseColorTexture");
                _ = self.getorPutUniformLocation("hasNormalTexture");
                _ = self.getorPutUniformLocation("hasMetallicRoughnessTexture");
                _ = self.getorPutUniformLocation("hasOcclusionTexture");
                _ = self.getorPutUniformLocation("hasEmissiveTexture");
                _ = self.getorPutUniformLocation("hasSpecularTexture");

                // Texture samplers
                _ = self.getorPutUniformLocation("baseColorTexture");
                _ = self.getorPutUniformLocation("normalTexture");
                _ = self.getorPutUniformLocation("metallicRoughnessTexture");
                _ = self.getorPutUniformLocation("occlusionTexture");
                _ = self.getorPutUniformLocation("emissiveTexture");
                _ = self.getorPutUniformLocation("specularTexture");
            },
            .Phong => {
                // Cache Phong uniform locations
                _ = self.getorPutUniformLocation("ambientColor");
                _ = self.getorPutUniformLocation("diffuseColor");
                _ = self.getorPutUniformLocation("specularColor");
                _ = self.getorPutUniformLocation("shininess");

                // Texture flags
                _ = self.getorPutUniformLocation("hasDiffuseTexture");
                _ = self.getorPutUniformLocation("hasSpecularTexture");
                _ = self.getorPutUniformLocation("hasNormalTexture");

                // Texture samplers
                _ = self.getorPutUniformLocation("diffuseTexture");
                _ = self.getorPutUniformLocation("specularTexture");
                _ = self.getorPutUniformLocation("normalTexture");
            },
        }
    }

    pub fn deinit(self: *ShaderResource) void {
        self.uniforms.deinit();

        if (self.ref_count == 0 and self.program_id != 0) {
            glad.glDeleteProgram(self.program_id);
            self.program_id = 0;
        }
    }
};

pub const MaterialResource = struct {
    material: MaterialVariant,
    ref_count: usize,
    texture_refs: std.StringArrayHashMap([:0]const u8),
    shader_ref: ?[:0]const u8 = null,
    uniforms: std.StringHashMap(UniformValue),

    pub fn init(allocator: std.mem.Allocator, material: MaterialVariant) !MaterialResource {
        var resource = MaterialResource{
            .material = material,
            .ref_count = 1,
            .texture_refs = std.StringArrayHashMap([:0]const u8).init(allocator),
            .uniforms = std.StringHashMap(UniformValue).init(allocator),
        };

        try resource.initDefaultUniforms(allocator);
        return resource;
    }

    pub fn initDefaultUniforms(self: *MaterialResource, allocator: std.mem.Allocator) !void {
        switch (self.material) {
            .Phong => |phong| {
                try self.setUniform(allocator, "ambientColor", .{ .Vec3 = phong.data.ambientColor });
                try self.setUniform(allocator, "diffuseColor", .{ .Vec4 = phong.data.diffuseColor });
                try self.setUniform(allocator, "specularColor", .{ .Vec3 = phong.data.specularColor });
                try self.setUniform(allocator, "shininess", .{ .Float = phong.data.shininess });
                try self.setUniform(allocator, "hasDiffuseTexture", .{ .Int = @intFromBool(phong.data.diffuseTexture != null) });
                try self.setUniform(allocator, "hasSpecularTexture", .{ .Int = @intFromBool(phong.data.specularTexture != null) });
                try self.setUniform(allocator, "hasNormalTexture", .{ .Int = @intFromBool(phong.data.normalTexture != null) });
            },
            .PBR => |pbr| {
                std.debug.print("{any}", .{pbr.data});
                try self.setUniform(allocator, "baseColorFactor", .{ .Vec4 = pbr.data.baseColorFactor });
                try self.setUniform(allocator, "metallicFactor", .{ .Float = pbr.data.metallicFactor });
                try self.setUniform(allocator, "roughnessFactor", .{ .Float = pbr.data.roughnessFactor });
                try self.setUniform(allocator, "emissiveFactor", .{ .Vec3 = pbr.data.emissiveFactor });
                try self.setUniform(allocator, "emissiveStrength", .{ .Float = pbr.data.emissiveStrength });
                try self.setUniform(allocator, "alphaCutoff", .{ .Float = pbr.data.alphaCutoff });
                try self.setUniform(allocator, "doubleSided", .{ .Int = @intFromBool(pbr.data.doubleSided) });

                // Texture flags
                std.debug.print("{any}", .{pbr.textures});
                try self.setUniform(allocator, "hasBaseColorTexture", .{ .Int = @intFromBool(pbr.textures.baseColor != null) });
                try self.setUniform(allocator, "hasNormalTexture", .{ .Int = @intFromBool(pbr.textures.normal != null) });
                try self.setUniform(allocator, "hasMetallicRoughnessTexture", .{ .Int = @intFromBool(pbr.textures.metallicRoughness != null) });
                try self.setUniform(allocator, "hasOcclusionTexture", .{ .Int = @intFromBool(pbr.textures.occlusion != null) });
                try self.setUniform(allocator, "hasEmissiveTexture", .{ .Int = @intFromBool(pbr.textures.emissive != null) });
            },
        }
    }

    pub fn setShaderRef(self: *MaterialResource, allocator: std.mem.Allocator, shader_name: []const u8) !void {
        if (self.shader_ref) |old_ref| {
            allocator.free(old_ref);
        }
        self.shader_ref = try allocator.dupeZ(u8, shader_name);
    }

    pub fn setUniform(self: *MaterialResource, allocator: std.mem.Allocator, name: []const u8, value: UniformValue) !void {
        const name_copy = try allocator.dupeZ(u8, name);

        // If the uniform already exists, free the old name
        if (self.uniforms.getKey(name)) |old_name| {
            if (self.uniforms.remove(name)) {
                return error.FailedToRemoveUniform;
            }
            allocator.free(old_name);
        }

        try self.uniforms.put(name_copy, value);
    }

    // Update deinit to free uniform names
    pub fn deinit(self: *MaterialResource, allocator: std.mem.Allocator) void {
        var it = self.texture_refs.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.texture_refs.deinit();

        var uniform_it = self.uniforms.iterator();
        while (uniform_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        self.uniforms.deinit();

        if (self.shader_ref) |shader_name| {
            allocator.free(shader_name);
            self.shader_ref = null;
        }
    }
};

const Self = @This();

allocator: std.mem.Allocator,
meshes: std.StringHashMap(MeshResource),
textures: std.StringHashMap(TextureResource),
shaders: std.StringHashMap(ShaderResource),
materials: std.StringHashMap(MaterialResource),

pub fn init(allocator: std.mem.Allocator) !*Self {
    const manager = try allocator.create(Self);
    manager.* = .{
        .allocator = allocator,
        .meshes = std.StringHashMap(MeshResource).init(allocator),
        .textures = std.StringHashMap(TextureResource).init(allocator),
        .shaders = std.StringHashMap(ShaderResource).init(allocator),
        .materials = std.StringHashMap(MaterialResource).init(allocator),
    };

    try manager.initDefaultShaders(
        @embedFile("../shaders/standard_vertex.glsl"),
        @embedFile("../shaders/standard_fragment.glsl"),
        @embedFile("../shaders/pbr_vertex.glsl"),
        @embedFile("../shaders/pbr_fragment.glsl"),
    );
    return manager;
}

pub fn deinit(self: *Self) void {
    var meshes_iter = self.meshes.iterator();
    while (meshes_iter.next()) |entry| {
        self.allocator.free(entry.key_ptr.*);
        entry.value_ptr.mesh.deinit();
        self.allocator.destroy(entry.value_ptr.mesh);
    }
    self.meshes.deinit();

    var textures_iter = self.textures.iterator();
    while (textures_iter.next()) |entry| {
        self.allocator.free(entry.key_ptr.*);
        if (entry.value_ptr.texture_id != 0) {
            glad.glDeleteTextures(1, &entry.value_ptr.texture_id);
        }
    }
    self.textures.deinit();

    var shaders_iter = self.shaders.iterator();
    while (shaders_iter.next()) |entry| {
        self.allocator.free(entry.key_ptr.*);
        entry.value_ptr.deinit();
    }
    self.shaders.deinit();

    var materials_iter = self.materials.iterator();
    while (materials_iter.next()) |entry| {
        self.allocator.free(entry.key_ptr.*);
        entry.value_ptr.deinit(self.allocator);
    }
    self.materials.deinit();
}

pub fn loadMesh(self: *Self, name: []const u8, vertices: []Mesh.Vertex, indices: ?[]u32, draw_fn: ?Mesh.draw) !*Mesh {
    // Check if mesh already exists
    if (self.meshes.getPtr(name)) |resource| {
        resource.instance_count += 1;
        return resource.mesh;
    }

    // Create new mesh
    const mesh = try Mesh.init(self.allocator, vertices, indices, draw_fn);
    const mesh_name = try self.allocator.dupeZ(u8, name);
    try self.meshes.put(mesh_name, MeshResource.init(mesh));
    return mesh;
}

pub fn unloadMesh(self: *Self, name: []const u8) void {
    if (self.meshes.getPtr(name)) |resource_ptr| {
        resource_ptr.instance_count -= 1;
        if (resource_ptr.instance_count == 0) {
            if (self.meshes.remove(name)) |kv| {
                self.allocator.free(kv.key);
                kv.value.mesh.deinit();
                self.allocator.destroy(kv.value.mesh);
            }
        }
    }
}

pub fn loadTexture(self: *Self, name: []const u8, texture_id: c_uint, width: u32, height: u32, channels: u32) !void {
    // Check if texture already exists
    if (self.textures.getPtr(name)) |resource| {
        resource.ref_count += 1;
        return;
    }

    // Create new texture resource
    const texture_name = try self.allocator.dupeZ(u8, name);
    try self.textures.put(texture_name, TextureResource.init(texture_id, width, height, channels));
}

pub fn unloadTexture(self: *Self, name: []const u8) void {
    if (self.textures.getPtr(name)) |resource_ptr| {
        resource_ptr.ref_count -= 1;
        if (resource_ptr.ref_count == 0) {
            if (self.textures.remove(name)) |kv| {
                self.allocator.free(kv.key);
                if (kv.value.texture_id != 0) {
                    glad.glDeleteTextures(1, &kv.value.texture_id);
                }
            }
        }
    }
}

pub fn initDefaultShaders(self: *Self, vertex_src_standard: []const u8, fragment_src_standard: []const u8, vertex_src_pbr: []const u8, fragment_src_pbr: []const u8) !void {
    // Add standard shader if it doesn't exist
    if (!self.shaders.contains("standard_shader")) {
        try self.loadShader("standard_shader", vertex_src_standard, fragment_src_standard, .Phong);
    }

    // Add PBR shader if it doesn't exist
    if (!self.shaders.contains("pbr_shader")) {
        try self.loadShader("pbr_shader", vertex_src_pbr, fragment_src_pbr, .PBR);
    }
}

pub fn loadShader(self: *Self, name: []const u8, vertex_src: []const u8, fragment_src: []const u8, shader_type: MaterialType) !void {
    // Check if shader already exists
    if (self.shaders.getPtr(name)) |resource| {
        resource.ref_count = 1;
        return;
    }

    // Create new shader program
    const vertex_shader = glad.glCreateShader(glad.GL_VERTEX_SHADER);
    defer glad.glDeleteShader(vertex_shader);

    const fragment_shader = glad.glCreateShader(glad.GL_FRAGMENT_SHADER);
    defer glad.glDeleteShader(fragment_shader);

    // Compile vertex shader
    const vsrc_ptr: [*c]const u8 = @ptrCast(vertex_src.ptr);
    const vsrc_len: c_int = @intCast(vertex_src.len);
    glad.glShaderSource(vertex_shader, 1, &vsrc_ptr, &vsrc_len);
    glad.glCompileShader(vertex_shader);

    // Check vertex shader compilation
    var success: c_int = 0;
    glad.glGetShaderiv(vertex_shader, glad.GL_COMPILE_STATUS, &success);
    if (success == 0) {
        var infoLog: [512]u8 = undefined;
        glad.glGetShaderInfoLog(vertex_shader, 512, null, &infoLog);
        std.debug.print("Vertex shader compilation failed: {s}\n", .{infoLog});
        return error.ShaderCompilationFailed;
    }

    // Compile fragment shader
    const fsrc_ptr: [*c]const u8 = @ptrCast(fragment_src.ptr);
    const fsrc_len: c_int = @intCast(fragment_src.len);
    glad.glShaderSource(fragment_shader, 1, &fsrc_ptr, &fsrc_len);
    glad.glCompileShader(fragment_shader);

    // Check fragment shader compilation
    glad.glGetShaderiv(fragment_shader, glad.GL_COMPILE_STATUS, &success);
    if (success == 0) {
        var infoLog: [512]u8 = undefined;
        glad.glGetShaderInfoLog(fragment_shader, 512, null, &infoLog);
        std.debug.print("Fragment shader compilation failed: {s}\n", .{infoLog});
        return error.ShaderCompilationFailed;
    }

    // Link shader program
    const program = glad.glCreateProgram();
    glad.glAttachShader(program, vertex_shader);
    glad.glAttachShader(program, fragment_shader);
    glad.glLinkProgram(program);

    // Check linking
    glad.glGetProgramiv(program, glad.GL_LINK_STATUS, &success);
    if (success == 0) {
        var infoLog: [512]u8 = undefined;
        glad.glGetProgramInfoLog(program, 512, null, &infoLog);
        std.debug.print("Shader program linking failed: {s}\n", .{infoLog});
        glad.glDeleteProgram(program);
        return error.ShaderLinkingFailed;
    }

    // Store the shader resource
    const shader_name = try self.allocator.dupeZ(u8, name);
    var shader_resource = try ShaderResource.init(self.allocator, program);

    shader_resource.cacheCommonUniforms(shader_type);

    try self.shaders.put(shader_name, shader_resource);
}

pub fn getShader(self: *Self, name: []const u8) ?*ShaderResource {
    return self.shaders.getPtr(name);
}

pub fn unloadShader(self: *Self, name: []const u8) void {
    if (self.shaders.getPtr(name)) |resource_ptr| {
        resource_ptr.ref_count -= 1;
        if (resource_ptr.ref_count == 0) {
            if (self.shaders.fetchRemove(name)) |kv| {
                self.allocator.free(kv.key);
                kv.value.deinit();
            }
        }
    }
}

pub fn loadMaterial(self: *Self, name: []const u8, material: MaterialVariant, shader_name: ?[]const u8) !void {
    if (self.materials.getPtr(name)) |resource| {
        resource.ref_count += 1;
        return;
    }

    const material_name = try self.allocator.dupeZ(u8, name);
    var material_resource = try MaterialResource.init(self.allocator, material);

    std.debug.print("Material Name: {s}", .{material_name});

    // Set shader reference if provided
    if (shader_name) |shader| {
        try material_resource.setShaderRef(self.allocator, shader);

        // Increase shader reference count
        if (self.shaders.getPtr(shader)) |shader_res| {
            shader_res.ref_count += 1;
        }
    } else {
        // Use default shader based on material type
        const default_shader = switch (material.getType()) {
            .PBR => "pbr_shader",
            .Phong => "standard_shader",
        };

        try material_resource.setShaderRef(self.allocator, default_shader);

        // Increase shader reference count if it exists
        if (self.shaders.getPtr(default_shader)) |shader_res| {
            shader_res.ref_count += 1;
        }
    }

    try self.materials.put(material_name, material_resource);
}

pub fn unloadMaterial(self: *Self, name: []const u8) void {
    if (self.materials.getPtr(name)) |resource_ptr| {
        resource_ptr.ref_count -= 1;
        if (resource_ptr.ref_count == 0) {
            if (self.materials.fetchRemove(name)) |kv| {

                // Decrease shader reference count
                if (kv.value.shader_ref) |shader_name| {
                    self.unloadShader(shader_name);
                }

                // Decrease texture reference counts
                var it = kv.value.texture_refs.iterator();
                while (it.next()) |entry| {
                    self.unloadTexture(entry.value_ptr.*);
                }

                self.allocator.free(kv.key);
                kv.value.deinit(self.allocator);
            }
        }
    }
}

const CACHE_DIR = ".asset-cache";
const CACHE_MAGIC = 0x474C5446; // "GLTF" little‑endian

fn cachePath(alloc: std.mem.Allocator, gltf_path: []const u8) []const u8 {
    // scene.gltf -> cache/scene.gltf.bin
    const modified_path = std.mem.replaceOwned(u8, alloc, gltf_path, "/", "-") catch @panic("Failed to create modifiedPath");
    return std.fmt.allocPrintZ(alloc, "{s}/{s}.bin", .{ CACHE_DIR, modified_path }) catch @panic("Failed to create cachePath");
}

fn isFresh(gltf_path: []const u8, bin_path: []const u8) bool {
    const gltf_mtime = std.fs.cwd().statFile(gltf_path) catch return false;
    const bin_mtime = std.fs.cwd().statFile(bin_path) catch return false;
    return bin_mtime.mtime >= gltf_mtime.mtime;
}

fn cacheWriteZString(writer: anytype, s: ?[:0]const u8) !void {
    // 0xFFFF_FFFF → “null”
    if (s) |str| {
        try writer.writeInt(u32, @intCast(str.len), .little);
        try writer.writeAll(str);
    } else {
        try writer.writeInt(u32, 0xFFFF_FFFF, .little);
    }
}

fn cacheReadZString(reader: anytype, alloc: std.mem.Allocator) !?[:0]const u8 {
    const len = try reader.readInt(u32, .little);
    if (len == 0xFFFF_FFFF) return null;

    const buf = try alloc.alloc(u8, len);
    _ = try reader.readAll(buf);
    return buf[0..len :0]; // slice :0 → NUL‑terminated
}

// ---------------------------------------------------------------------------
// Helpers – optional fixed‑size float arrays
// ---------------------------------------------------------------------------

fn writeOptArray(writer: anytype, comptime N: usize, val: ?[N]f32) !void {
    try writer.writeByte(if (val != null) 1 else 0);
    if (val) |v| try writer.writeAll(std.mem.asBytes(&v));
}

fn readOptArray(
    reader: anytype,
    comptime N: usize,
) !?[N]f32 {
    if (try reader.readByte() == 0) return null;

    var out: [N]f32 = undefined;
    _ = try reader.readAll(std.mem.asBytes(&out));
    return out;
}

// ---------------------------------------------------------------------------
// writeCache – serialise ModelResource into <path>.bin
// ---------------------------------------------------------------------------

pub fn writeCache(res: *GLTFParser.ModelResource, path: []const u8) !void {
    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    var w = file.writer();

    try w.writeInt(u32, CACHE_MAGIC, .little);
    try w.writeInt(u32, @intCast(res.entities.len), .little);

    for (res.entities) |e| {
        // --- variable‑length strings --------------------------------------
        try cacheWriteZString(w, e.name);
        try cacheWriteZString(w, e.mesh_name);
        try cacheWriteZString(w, e.material_name);

        // --- optional matrix ----------------------------------------------
        try w.writeByte(if (e.local_transformation != null) 1 else 0);
        if (e.local_transformation) |m|
            try w.writeAll(std.mem.asBytes(&m));

        // --- TRS arrays ----------------------------------------------------
        try writeOptArray(w, 3, e.translation);
        try writeOptArray(w, 4, e.rotation);
        try writeOptArray(w, 3, e.scale);

        // --- hierarchy -----------------------------------------------------
        try w.writeInt(i32, if (e.parent_idx) |idx| @intCast(idx) else -1, .little);

        try w.writeInt(u32, @intCast(e.children.len), .little);
        for (e.children) |c|
            try w.writeInt(u32, @intCast(c), .little);
    }
}

// ---------------------------------------------------------------------------
// readCache – load ModelResource from <path>.bin
// ---------------------------------------------------------------------------

pub fn readCache(alloc: std.mem.Allocator, path: []const u8) !*GLTFParser.ModelResource {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var r = file.reader();

    if (try r.readInt(u32, .little) != CACHE_MAGIC)
        return error.BadCache;

    const count = try r.readInt(u32, .little);

    var res = try alloc.create(GLTFParser.ModelResource);
    res.* = .{
        .model_id = try alloc.dupeZ(u8, path), // useful for debugging
        .entities = try alloc.alloc(GLTFParser.ModelResource.EntityInfo, count),
        .allocator = alloc,
    };

    var i: usize = 0;
    while (i < count) : (i += 1) {
        var e = &res.entities[i];

        // --- strings ------------------------------------------------------
        e.name = try cacheReadZString(r, alloc);
        e.mesh_name = try cacheReadZString(r, alloc);
        e.material_name = try cacheReadZString(r, alloc);

        // --- optional matrix ---------------------------------------------
        if (try r.readByte() == 1) {
            var m: Math.Mat4 = undefined;
            _ = try r.readAll(std.mem.asBytes(&m));
            e.local_transformation = m;
        } else e.local_transformation = null;

        // --- TRS arrays ---------------------------------------------------
        e.translation = try readOptArray(r, 3);
        e.rotation = try readOptArray(r, 4);
        e.scale = try readOptArray(r, 3);

        // --- hierarchy ----------------------------------------------------
        const parent_raw = try r.readInt(i32, .little);
        e.parent_idx = if (parent_raw >= 0) @as(usize, @intCast(parent_raw)) else null;

        const child_cnt = try r.readInt(u32, .little);
        const child_buf = try alloc.alloc(usize, child_cnt);
        var j: usize = 0;
        while (j < child_cnt) : (j += 1)
            child_buf[j] = @as(usize, @intCast(try r.readInt(u32, .little)));
        e.children = child_buf;
    }

    return res;
}

pub fn loadGLTFModel(self: *Self, allocator: std.mem.Allocator, filepath: []const u8) !*GLTFParser.ModelResource {
    var gltf = try GLTF.init(allocator, filepath);
    defer gltf.deinit();

    const model_id = try std.fmt.allocPrint(allocator, "model_{s}", .{filepath});
    try self.processGLTFResources(gltf, model_id);
    return GLTFParser.createModelResource(allocator, model_id, gltf);
}

pub fn loadGLTFModelCached(
    self: *Self,
    allocator: std.mem.Allocator,
    gltf_path: []const u8,
) !*GLTFParser.ModelResource {
    const bin_path = cachePath(allocator, gltf_path);
    defer allocator.free(bin_path);

    // ---------- fast path -------------------------------------------------
    // if (isFresh(gltf_path, bin_path)) {
    //     if (readCache(allocator, bin_path)) |mr| {
    //         const gltf = try GLTF.init(allocator, gltf_path);
    //         defer gltf.deinit();
    //         try self.processGLTFTextures(gltf, mr.model_id);
    //         try self.processGLTFMaterials(gltf, mr.model_id);
    //         return mr;
    //     } else |_| {
    //         std.debug.print("cache unreadable – rebuilding\n", .{});
    //     }
    // }

    // ---------- cold path -------------------------------------------------
    const mr = try self.loadGLTFModel(allocator, gltf_path);

    writeCache(mr, bin_path) catch |e|
        std.debug.print("cache write failed: {}\n", .{e});

    return mr;
}

fn processGLTFResources(self: *Self, gltf: *GLTF, model_id: []const u8) !void {
    try self.processGLTFTextures(gltf, model_id);
    try self.processGLTFMaterials(gltf, model_id);
    try self.processGLTFMeshes(gltf, model_id);
}

fn processGLTFTextures(self: *Self, gltf: *GLTF, model_id: []const u8) !void {
    if (gltf.document.images == null or gltf.document.textures == null) return;

    const allocator = self.allocator;
    // Process all images/textures
    for (gltf.document.images.?, 0..) |image, img_idx| {
        // Generate unique texture name
        const texture_name = if (image.name) |name|
            try std.fmt.allocPrint(allocator, "{s}_{s}", .{ model_id, name })
        else
            try std.fmt.allocPrint(allocator, "{s}_texture_{d}", .{ model_id, img_idx });
        defer allocator.free(texture_name);

        // Load the texture data
        var texture_id: Mesh.TextureID = Mesh.TextureID{};
        var img: ?*ImageLoader.Image = null;

        if (image.uri) |uri| {
            // Load from file
            const full_path = try std.fmt.allocPrint(allocator, "{s}{s}", .{ gltf.base_path, uri });
            defer allocator.free(full_path);

            img = try ImageLoader.Image.loadFromFile(allocator, full_path);
            defer img.?.deinit();

            texture_id = try img.?.createGLTexture();
        } else if (image.bufferView != null) {
            // Load from buffer
            img = try gltf.loadBufferViewImage(allocator, image.bufferView.?);
            defer img.?.deinit();

            texture_id = try img.?.createGLTexture();
        }

        // Register in resource manager
        if (texture_id.y != 0) {
            const width = if (img) |_img| _img.width else 0;
            const height = if (img) |_img| _img.height else 0;
            const channels = if (img) |_img| _img.getChannels() else 0;
            try self.loadTexture(texture_name, texture_id.y, width, height, channels);
        }
    }
}

fn processGLTFMaterials(self: *Self, gltf: *GLTF, model_id: []const u8) !void {
    if (gltf.document.materials == null) return;

    const allocator = self.allocator;
    for (gltf.document.materials.?, 0..) |material_def, mat_idx| {
        // Generate unique material name
        const material_name = if (material_def.name) |name|
            try std.fmt.allocPrint(allocator, "{s}_{s}", .{ model_id, name })
        else
            try std.fmt.allocPrint(allocator, "{s}_material_{d}", .{ model_id, mat_idx });
        defer allocator.free(material_name);

        // Determine material type
        const material_type: MaterialType = if (material_def.pbrMetallicRoughness != null or (material_def.extensions != null and material_def.extensions.?.KHR_materials_pbrSpecularGlossiness != null))
            MaterialType.PBR
        else
            MaterialType.Phong;

        const material = switch (material_type) {
            .PBR => try self.createPBRMaterial(material_def, gltf, model_id),
            .Phong => try self.createPhongMaterial(material_def, gltf, model_id),
        };

        // Register material in resource manager
        try self.loadMaterial(material_name, material, null);
    }
}

fn createPBRMaterial(self: *Self, material_def: GLTFParser.Material, gltf: *GLTF, model_id: []const u8) !MaterialVariant {
    const allocator = self.allocator;
    var material = Material(.PBR){};

    // Set material properties
    if (material_def.doubleSided) |double_sided| {
        material.data.doubleSided = double_sided;
    }

    if (material_def.alphaMode) |alpha_mode_str| {
        if (std.mem.eql(u8, alpha_mode_str, @tagName(Mesh.AlphaMode.MASK))) {
            material.data.alphaMode = .MASK;
        } else if (std.mem.eql(u8, alpha_mode_str, @tagName(Mesh.AlphaMode.BLEND))) {
            material.data.alphaMode = .BLEND;
        } else {
            material.data.alphaMode = .OPAQUE;
        }
    }

    if (material_def.alphaCutoff) |alpha_cutoff| {
        material.data.alphaCutoff = alpha_cutoff;
    }

    // Set emissive factor if present
    if (material_def.emissiveFactor) |emissive| {
        material.data.emissiveFactor = emissive;
    }

    // Process PBR Metallic-Roughness parameters
    if (material_def.pbrMetallicRoughness) |pbr_mr| {
        if (pbr_mr.baseColorFactor) |base_color| {
            material.data.baseColorFactor = base_color;
        }

        if (pbr_mr.metallicFactor) |metallic| {
            material.data.metallicFactor = metallic;
        }

        if (pbr_mr.roughnessFactor) |roughness| {
            material.data.roughnessFactor = roughness;
        }

        // Associate base color texture if present
        if (pbr_mr.baseColorTexture) |tex_info| {
            if (gltf.document.textures) |textures| {
                if (tex_info.index < textures.len) {
                    const texture = textures[tex_info.index];
                    if (texture.source) |source_idx| {
                        if (gltf.document.images) |images| {
                            if (source_idx < images.len) {
                                const img = images[source_idx];
                                const tex_name = if (img.name) |name|
                                    try std.fmt.allocPrint(allocator, "{s}_{s}", .{ model_id, name })
                                else
                                    try std.fmt.allocPrint(allocator, "{s}_texture_{d}", .{ model_id, source_idx });
                                defer allocator.free(tex_name);

                                if (self.textures.get(tex_name)) |tex_resource| {
                                    material.textures.baseColor = tex_resource.texture_id;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Associate metallic-roughness texture if present
        if (pbr_mr.metallicRoughnessTexture) |tex_info| {
            if (gltf.document.textures) |textures| {
                if (tex_info.index < textures.len) {
                    const texture = textures[tex_info.index];
                    if (texture.source) |source_idx| {
                        if (gltf.document.images) |images| {
                            if (source_idx < images.len) {
                                const img = images[source_idx];
                                const tex_name = if (img.name) |name|
                                    try std.fmt.allocPrint(allocator, "{s}_{s}", .{ model_id, name })
                                else
                                    try std.fmt.allocPrint(allocator, "{s}_texture_{d}", .{ model_id, source_idx });
                                defer allocator.free(tex_name);

                                if (self.textures.get(tex_name)) |tex_resource| {
                                    material.textures.metallicRoughness = tex_resource.texture_id;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Process normal map texture
    if (material_def.normalTexture) |tex_info| {
        if (gltf.document.textures) |textures| {
            if (tex_info.index < textures.len) {
                const texture = textures[tex_info.index];
                if (texture.source) |source_idx| {
                    if (gltf.document.images) |images| {
                        if (source_idx < images.len) {
                            const img = images[source_idx];
                            const tex_name = if (img.name) |name|
                                try std.fmt.allocPrint(allocator, "{s}_{s}", .{ model_id, name })
                            else
                                try std.fmt.allocPrint(allocator, "{s}_texture_{d}", .{ model_id, source_idx });
                            defer allocator.free(tex_name);

                            if (self.textures.get(tex_name)) |tex_resource| {
                                material.textures.normal = tex_resource.texture_id;
                            }
                        }
                    }
                }
            }
        }
    }

    // Process occlusion texture
    if (material_def.occlusionTexture) |tex_info| {
        if (gltf.document.textures) |textures| {
            if (tex_info.index < textures.len) {
                const texture = textures[tex_info.index];
                if (texture.source) |source_idx| {
                    if (gltf.document.images) |images| {
                        if (source_idx < images.len) {
                            const img = images[source_idx];
                            const tex_name = if (img.name) |name|
                                try std.fmt.allocPrint(allocator, "{s}_{s}", .{ model_id, name })
                            else
                                try std.fmt.allocPrint(allocator, "{s}_texture_{d}", .{ model_id, source_idx });
                            defer allocator.free(tex_name);

                            if (self.textures.get(tex_name)) |tex_resource| {
                                material.textures.occlusion = tex_resource.texture_id;
                            }
                        }
                    }
                }
            }
        }
    }

    // Process emissive texture
    if (material_def.emissiveTexture) |tex_info| {
        if (gltf.document.textures) |textures| {
            if (tex_info.index < textures.len) {
                const texture = textures[tex_info.index];
                if (texture.source) |source_idx| {
                    if (gltf.document.images) |images| {
                        if (source_idx < images.len) {
                            const img = images[source_idx];
                            const tex_name = if (img.name) |name|
                                try std.fmt.allocPrint(allocator, "{s}_{s}", .{ model_id, name })
                            else
                                try std.fmt.allocPrint(allocator, "{s}_texture_{d}", .{ model_id, source_idx });
                            defer allocator.free(tex_name);

                            if (self.textures.get(tex_name)) |tex_resource| {
                                material.textures.emissive = tex_resource.texture_id;
                            }
                        }
                    }
                }
            }
        }
    }

    // Process extensions
    if (material_def.extensions) |extensions| {
        // KHR_materials_pbrSpecularGlossiness
        if (extensions.KHR_materials_pbrSpecularGlossiness) |sg| {
            if (sg.diffuseFactor) |diffuse| {
                material.data.diffuseFactor = diffuse;
            }

            if (sg.specularFactor) |specular| {
                material.data.specularFactor = specular;
            }

            if (sg.glossinessFactor) |glossiness| {
                material.data.glossinessFactor = glossiness;
            }

            // Process diffuse texture
            if (sg.diffuseTexture) |tex_info| {
                if (gltf.document.textures) |textures| {
                    if (tex_info.index < textures.len) {
                        const texture = textures[tex_info.index];
                        if (texture.source) |source_idx| {
                            if (gltf.document.images) |images| {
                                if (source_idx < images.len) {
                                    const img = images[source_idx];
                                    const tex_name = if (img.name) |name|
                                        try std.fmt.allocPrint(allocator, "{s}_{s}", .{ model_id, name })
                                    else
                                        try std.fmt.allocPrint(allocator, "{s}_texture_{d}", .{ model_id, source_idx });
                                    defer allocator.free(tex_name);

                                    if (self.textures.get(tex_name)) |tex_resource| {
                                        // In specular-glossiness workflow, diffuse texture goes to baseColor
                                        material.textures.baseColor = tex_resource.texture_id;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Process specular-glossiness texture
            if (sg.specularGlossinessTexture) |tex_info| {
                if (gltf.document.textures) |textures| {
                    if (tex_info.index < textures.len) {
                        const texture = textures[tex_info.index];
                        if (texture.source) |source_idx| {
                            if (gltf.document.images) |images| {
                                if (source_idx < images.len) {
                                    const img = images[source_idx];
                                    const tex_name = if (img.name) |name|
                                        try std.fmt.allocPrint(allocator, "{s}_{s}", .{ model_id, name })
                                    else
                                        try std.fmt.allocPrint(allocator, "{s}_texture_{d}", .{ model_id, source_idx });
                                    defer allocator.free(tex_name);

                                    if (self.textures.get(tex_name)) |tex_resource| {
                                        // In specular-glossiness workflow, specular-glossiness maps to metallicRoughness slot
                                        material.textures.metallicRoughness = tex_resource.texture_id;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // KHR_materials_emissive_strength extension
        if (extensions.KHR_materials_emissive_strength) |es| {
            if (es.emissiveStrength) |strength| {
                material.data.emissiveStrength = strength;
            }
        }

        // KHR_materials_specular extension
        if (extensions.KHR_materials_specular) |spec| {
            if (spec.specularFactor) |factor| {
                material.data.specularStrength = factor;
            }

            if (spec.specularColorFactor) |color| {
                material.data.specularColor = color;
            }

            // Process specular texture
            if (spec.specularTexture) |tex_info| {
                if (gltf.document.textures) |textures| {
                    if (tex_info.index < textures.len) {
                        const texture = textures[tex_info.index];
                        if (texture.source) |source_idx| {
                            if (gltf.document.images) |images| {
                                if (source_idx < images.len) {
                                    const img = images[source_idx];
                                    const tex_name = if (img.name) |name|
                                        try std.fmt.allocPrint(allocator, "{s}_{s}", .{ model_id, name })
                                    else
                                        try std.fmt.allocPrint(allocator, "{s}_texture_{d}", .{ model_id, source_idx });
                                    defer allocator.free(tex_name);

                                    if (self.textures.get(tex_name)) |tex_resource| {
                                        material.textures.specular = tex_resource.texture_id;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return MaterialVariant{ .PBR = material };
}

fn createPhongMaterial(self: *Self, material_def: GLTFParser.Material, gltf: *GLTF, model_id: []const u8) !MaterialVariant {
    _ = self;
    _ = material_def;
    _ = gltf;
    _ = model_id;
    @panic("Not Implemented!");
}

fn processGLTFMeshes(self: *Self, gltf: *GLTF, model_id: []const u8) !void {
    if (gltf.document.meshes == null) return;

    const allocator = self.allocator;
    for (gltf.document.meshes.?, 0..) |mesh_def, mesh_idx| {
        for (mesh_def.primitives, 0..) |_, prim_idx| {
            // Generate unique mesh name
            const mesh_name = if (mesh_def.name) |name|
                try std.fmt.allocPrint(allocator, "{s}_{s}_prim_{d}", .{ model_id, name, prim_idx })
            else
                try std.fmt.allocPrint(allocator, "{s}_mesh_{d}_prim_{d}", .{ model_id, mesh_idx, prim_idx });
            defer allocator.free(mesh_name);

            const loaded_mesh = try gltf.loadMesh(allocator, mesh_idx);
            if (loaded_mesh) |mesh| {
                _ = try self.loadMesh(mesh_name, mesh.vertices, mesh.indices, mesh._draw);
            }
        }
    }
}

pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
    _ = fmt;
    _ = options;

    try writer.print("ResourceManager(\n", .{});

    // -- Meshes --
    try writer.print("  Meshes:\n", .{});
    var mesh_it = self.meshes.iterator();
    while (mesh_it.next()) |entry| {
        // Each entry has .key_ptr (pointer to the name string)
        // and .value_ptr (pointer to the MeshResource).
        try writer.print("\t'{s}':\n\t\t(instance_count={d})\n", .{
            entry.key_ptr.*,
            entry.value_ptr.instance_count,
        });
    }
    try writer.print("\n", .{});

    // -- Textures --
    try writer.print("  Textures:\n", .{});
    var tex_it = self.textures.iterator();
    while (tex_it.next()) |entry| {
        const texture = entry.value_ptr.*;
        try writer.print("\t'{s}:\n\t\t(texture_id={d}, ref_count={d}, size={d}x{d}, channels={d})\n", .{
            entry.key_ptr.*,
            texture.texture_id,
            texture.ref_count,
            texture.width,
            texture.height,
            texture.channels,
        });
    }
    try writer.print("\n", .{});

    // -- Shaders --
    try writer.print("  Shaders:\n", .{});
    var sh_it = self.shaders.iterator();
    while (sh_it.next()) |entry| {
        const shader = entry.value_ptr.*;
        try writer.print("\t'{s}':\n\t\t(program_id={d}, ref_count={d}, uniform_count={d})\n", .{
            entry.key_ptr.*,
            shader.program_id,
            shader.ref_count,
            shader.uniforms.count(),
        });
    }
    try writer.print("\n", .{});

    // -- Materials --
    try writer.print("  Materials:\n", .{});
    var mat_it = self.materials.iterator();
    while (mat_it.next()) |entry| {
        const material_res = entry.value_ptr.*;
        try writer.print("\t'{s}':\n\t\t(ref_count={d}, material_type={s}, shader={s}, texture_ref_count={d})\n", .{
            entry.key_ptr.*,
            material_res.ref_count,
            @tagName(material_res.material.getType()),
            if (material_res.shader_ref) |sh| sh else "null",
            material_res.texture_refs.count(),
        });

        // If you want to print out each texture ref (texture_type -> texture_name), do:
        var tex_ref_it = material_res.texture_refs.iterator();
        while (tex_ref_it.next()) |tex_entry| {
            try writer.print("\t\t\ttexture_type='{s}' => texture_name='{s}'\n", .{
                tex_entry.key_ptr.*,
                tex_entry.value_ptr.*,
            });
        }
    }

    try writer.print(")\n", .{});
}

pub fn debug(self: *Self) void {
    std.debug.print("{any}", .{self});
}
