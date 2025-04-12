const std = @import("std");
const Debug = @import("Debug.zig");
const Node = @import("Node.zig");
const gl = @import("bindings/gl.zig");
const glad = gl.glad;

const Self = @This();

pub const Vertex = struct {
    position: [3]f32,
    color: [3]f32,
    texture: ?[2]f32 = null,
    alpha: ?f32 = null,
    normal: ?[3]f32 = null, // Vertex normal
    tangent: ?[3]f32 = null, // Tangent vector
    bitangent: ?[3]f32 = null, // Bitangent vector
};

pub const Metadata = struct {
    VAO: u32 = 0,
    VBO: u32 = 0,
    IBO: u32 = 0,
};

pub const TextureID = struct {
    y: c_uint = 0,
    uv: c_uint = 0,
    depth: c_uint = 0,

    pub fn clone(self: TextureID, allocator: std.mem.Allocator) !*TextureID {
        const new_id = try allocator.create(TextureID);
        new_id.* = self;
        return new_id;
    }

    pub fn format(
        self: TextureID,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "(\n\t\t\ty: {d},\n\t\t\tuv: {d},\n\t\t\tdepth: {d}\n\t\t)",
            .{ self.y, self.uv, self.depth },
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

        try writer.print("PBRTextures {{\n", .{});

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
        try writer.print("\n", .{});

        try writer.print("}}", .{});
    }
};

pub const Material = struct {

    // Standard metallic-roughness parameters
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
    alphaMode: AlphaMode = .OPAQUE,
    alphaCutoff: f32 = 0.5,

    textures: PBRTextures = PBRTextures{},

    pub fn format(
        self: Material,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "\nMaterial (\n\tFactors: (\n\t\tbaseColorFactor: {d:.4},\n\t\tmetallicFactor: {d:.4},\n\t\troughnessFactor: {d:.4}\n\t),\n",
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
        try writer.print("PBR Textures: {any}\n)\n", .{self.textures});
    }
};

pub const AlphaMode = enum {
    OPAQUE,
    MASK,
    BLEND,
};

pub const MeshFlags = struct {
    use_texture: bool = false,
    use_depth: bool = false,
    use_alpha: bool = false,
    use_pbr: bool = false,
};

pub const draw = *const fn (mesh: *Self) void;

allocator: std.mem.Allocator,
node: ?*Node = null,
textureID: TextureID = TextureID{},
vertices: []Vertex,
indices: ?[]u32 = null,
meta: Metadata,
_draw: draw,
drawType: glad.GLenum = glad.GL_TRIANGLES,
flags: ?MeshFlags = MeshFlags{},
material: Material = Material{},

pub fn init(allocator: std.mem.Allocator, vertices: []Vertex, indices: ?[]u32, draw_fn: ?draw) !*Self {
    var mesh = try allocator.create(Self);

    mesh.allocator = allocator;
    mesh.vertices = vertices;
    mesh.indices = indices;
    mesh.meta = Metadata{};
    mesh._draw = draw_fn orelse default_draw;

    // OpenGL initialization...
    mesh.initGL() catch |err| {
        std.debug.print("Failed to Initialize Mesh in openGL => {any}", .{err});
        return err;
    };

    return mesh;
}

fn initGL(self: *Self) !void {
    // Debug output
    // std.debug.print("Initializing mesh with {} vertices\n", .{self.vertices.len});

    // Generate and check VAO
    glad.glGenVertexArrays(1, &self.meta.VAO);
    if (self.meta.VAO == 0) {
        std.debug.print("Failed to generate VAO\n", .{});
        return error.OpenGLBufferError;
    }

    // Generate and check VBO
    glad.glGenBuffers(1, &self.meta.VBO);
    if (self.meta.VBO == 0) {
        std.debug.print("Failed to generate VBO\n", .{});
        glad.glDeleteVertexArrays(1, &self.meta.VAO);
        return error.OpenGLBufferError;
    }

    // Generate and check IBO if needed
    if (self.indices != null) {
        glad.glGenBuffers(1, &self.meta.IBO);
        if (self.meta.IBO == 0) {
            std.debug.print("Failed to generate IBO\n", .{});
            glad.glDeleteBuffers(1, &self.meta.VBO);
            glad.glDeleteVertexArrays(1, &self.meta.VAO);
            return error.OpenGLBufferError;
        }
    }

    // Bind VAO first
    glad.glBindVertexArray(self.meta.VAO);

    // Setup VBO
    glad.glBindBuffer(glad.GL_ARRAY_BUFFER, self.meta.VBO);
    glad.glBufferData(
        glad.GL_ARRAY_BUFFER,
        @intCast(self.vertices.len * @sizeOf(Vertex)),
        self.vertices.ptr,
        glad.GL_STATIC_DRAW,
    );

    // Check for errors after buffer data
    const err = glad.glGetError();
    if (err != glad.GL_NO_ERROR) {
        std.debug.print("OpenGL error after buffer data: 0x{x}\n", .{err});
        glad.glDeleteBuffers(1, &self.meta.VBO);
        glad.glDeleteVertexArrays(1, &self.meta.VAO);
        return error.OpenGLBufferError;
    }

    // Setup IBO if present
    if (self.indices) |ind| {
        glad.glBindBuffer(glad.GL_ELEMENT_ARRAY_BUFFER, self.meta.IBO);
        glad.glBufferData(
            glad.GL_ELEMENT_ARRAY_BUFFER,
            @intCast(ind.len * @sizeOf(u32)),
            ind.ptr,
            glad.GL_STATIC_DRAW,
        );
    }

    // Setup vertex attributes
    // Position attribute (location 0)
    glad.glVertexAttribPointer(
        0, // location
        3, // size (vec3)
        glad.GL_FLOAT,
        glad.GL_FALSE,
        @sizeOf(Vertex),
        null,
    );
    glad.glEnableVertexAttribArray(0);

    // Color attribute (location 1)
    const color_offset = @offsetOf(Vertex, "color");
    glad.glVertexAttribPointer(
        1, // location
        3, // size (vec3)
        glad.GL_FLOAT,
        glad.GL_FALSE,
        @sizeOf(Vertex),
        @ptrFromInt(color_offset),
    );
    glad.glEnableVertexAttribArray(1);

    // Texture coordinates attribute (location 2)
    const tex_coord_offset = @offsetOf(Vertex, "texture");
    glad.glVertexAttribPointer(
        2, // location
        2, // (vec2)
        glad.GL_FLOAT,
        glad.GL_FALSE,
        @sizeOf(Vertex), // stride
        @ptrFromInt(tex_coord_offset),
    );
    glad.glEnableVertexAttribArray(2);

    // Normal attribute (location 3) - only if we have normals
    const normal_offset = @offsetOf(Vertex, "normal");
    if (self.vertices.len > 0 and self.vertices[0].normal != null) {
        glad.glVertexAttribPointer(
            3, // location
            3, // size (vec3)
            glad.GL_FLOAT,
            glad.GL_FALSE,
            @sizeOf(Vertex),
            @ptrFromInt(normal_offset),
        );
        glad.glEnableVertexAttribArray(3);
    }

    // Tangent attribute (location 4) - only if we have tangents
    const tangent_offset = @offsetOf(Vertex, "tangent");
    if (self.vertices.len > 0 and self.vertices[0].tangent != null) {
        glad.glVertexAttribPointer(
            4, // location
            3, // size (vec3)
            glad.GL_FLOAT,
            glad.GL_FALSE,
            @sizeOf(Vertex),
            @ptrFromInt(tangent_offset),
        );
        glad.glEnableVertexAttribArray(4);
    }

    // Bitangent attribute (location 5) - only if we have bitangents
    const bitangent_offset = @offsetOf(Vertex, "bitangent");
    if (self.vertices.len > 0 and self.vertices[0].bitangent != null) {
        glad.glVertexAttribPointer(
            5, // location
            3, // size (vec3)
            glad.GL_FLOAT,
            glad.GL_FALSE,
            @sizeOf(Vertex),
            @ptrFromInt(bitangent_offset),
        );
        glad.glEnableVertexAttribArray(5);
    }

    glad.glBindVertexArray(0);

    // Final error check
    const final_error = glad.glGetError();
    if (final_error != glad.GL_NO_ERROR) {
        std.debug.print("OpenGL error at end of mesh init: 0x{x}\n", .{final_error});
    }
}

pub fn deinit(self: *Self) void {
    // Free OpenGL resources
    if (self.meta.VAO != 0) glad.glDeleteVertexArrays(1, &self.meta.VAO);
    if (self.meta.VBO != 0) glad.glDeleteBuffers(1, &self.meta.VBO);
    if (self.meta.IBO != 0) glad.glDeleteBuffers(1, &self.meta.IBO);
}

pub fn gen_draw(comptime drawType: glad.GLuint) draw {
    return struct {
        pub fn default_draw(mesh: *Self) void {
            glad.glBindVertexArray(mesh.meta.VAO);

            if (mesh.indices) |indices| {
                glad.glDrawElements(drawType, @intCast(indices.len), glad.GL_UNSIGNED_INT, null);
            } else {
                glad.glDrawArrays(drawType, 0, @intCast(mesh.vertices.len));
            }
        }
    }.default_draw;
}

pub fn default_draw(mesh: *Self) void {
    glad.glBindVertexArray(mesh.meta.VAO);

    if (mesh.indices) |indices| {
        glad.glDrawElements(mesh.drawType, @intCast(indices.len), glad.GL_UNSIGNED_INT, null);
    } else {
        glad.glDrawArrays(mesh.drawType, 0, @intCast(mesh.vertices.len));
    }
}

pub fn calculateTangents(vertices: []Vertex, indices: ?[]u32) void {
    // Skip if no normals or texture coordinates, or if tangents already exist
    if (vertices.len == 0 or
        vertices[0].normal == null or
        vertices[0].texture == null or
        vertices[0].tangent != null)
    {
        return;
    }

    // We need indices for proper tangent calculation
    if (indices == null) {
        return;
    }

    // For each face (triangle), calculate tangent and bitangent
    var i: usize = 0;
    while (i < indices.?.len) : (i += 3) {
        const _i0 = indices.?[i];
        const _i1 = indices.?[i + 1];
        const _i2 = indices.?[i + 2];

        // Get vertices of the triangle
        var v0 = &vertices[_i0];
        var v1 = &vertices[_i1];
        var v2 = &vertices[_i2];

        // Get positions
        const pos0 = v0.position;
        const pos1 = v1.position;
        const pos2 = v2.position;

        // Get texture coordinates
        const uv0 = v0.texture.?;
        const uv1 = v1.texture.?;
        const uv2 = v2.texture.?;

        // Calculate edges and UV deltas
        const edge1 = [3]f32{ pos1[0] - pos0[0], pos1[1] - pos0[1], pos1[2] - pos0[2] };

        const edge2 = [3]f32{ pos2[0] - pos0[0], pos2[1] - pos0[1], pos2[2] - pos0[2] };

        const deltaUV1 = [2]f32{ uv1[0] - uv0[0], uv1[1] - uv0[1] };

        const deltaUV2 = [2]f32{ uv2[0] - uv0[0], uv2[1] - uv0[1] };

        // Calculate tangent and bitangent
        // Formula: T = (E1 * dV2 - E2 * dV1) / (dU1 * dV2 - dU2 * dV1)
        //          B = (E2 * dU1 - E1 * dU2) / (dU1 * dV2 - dU2 * dV1)

        const f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1]);

        var tangent = [3]f32{ f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]), f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]), f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2]) };

        var bitangent = [3]f32{ f * (-deltaUV2[0] * edge1[0] + deltaUV1[0] * edge2[0]), f * (-deltaUV2[0] * edge1[1] + deltaUV1[0] * edge2[1]), f * (-deltaUV2[0] * edge1[2] + deltaUV1[0] * edge2[2]) };

        // Normalize
        const tangent_length = std.math.sqrt(tangent[0] * tangent[0] +
            tangent[1] * tangent[1] +
            tangent[2] * tangent[2]);

        if (tangent_length > 0.0001) {
            tangent[0] /= tangent_length;
            tangent[1] /= tangent_length;
            tangent[2] /= tangent_length;
        }

        const bitangent_length = std.math.sqrt(bitangent[0] * bitangent[0] +
            bitangent[1] * bitangent[1] +
            bitangent[2] * bitangent[2]);

        if (bitangent_length > 0.0001) {
            bitangent[0] /= bitangent_length;
            bitangent[1] /= bitangent_length;
            bitangent[2] /= bitangent_length;
        }

        // Assign to vertices
        // In a proper implementation we would average tangents/bitangents for shared vertices
        // This simple approach just assigns them per face
        v0.tangent = tangent;
        v1.tangent = tangent;
        v2.tangent = tangent;

        v0.bitangent = bitangent;
        v1.bitangent = bitangent;
        v2.bitangent = bitangent;
    }
}

pub fn debug(self: Self) void {
    Debug.printVertexShader(self.meta.VBO, self.vertices.len) catch |err| {
        std.debug.print("Failed to debug vertex shader {any}\n", .{err});
    };
}
