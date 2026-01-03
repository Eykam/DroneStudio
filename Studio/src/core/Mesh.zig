// src/core/Mesh.zig

const std = @import("std");
const Debug = @import("Debug.zig");
const gl = @import("bindings/gl.zig");
const glad = gl.glad;

const Self = @This();

pub const DrawType = enum {
    lines,
    triangles,
    points,
    line_strip,
    triangle_strip,

    pub fn toGL(self: DrawType) glad.GLenum {
        return switch (self) {
            .lines => glad.GL_LINES,
            .triangles => glad.GL_TRIANGLES,
            .points => glad.GL_POINTS,
            .line_strip => glad.GL_LINE_STRIP,
            .triangle_strip => glad.GL_TRIANGLE_STRIP,
        };
    }
};

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
vertices: []Vertex,
indices: ?[]u32 = null,
meta: Metadata,
_draw: draw = default_draw,
drawType: glad.GLenum = glad.GL_TRIANGLES,
flags: ?MeshFlags = MeshFlags{},

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
    self.allocator.free(self.vertices);
    if (self.indices) |idx| self.allocator.free(idx);
}

/// Update the mesh vertices with new data, updating the OpenGL buffer
pub fn updateVertices(self: *Self, new_vertices: []Vertex) !void {
    // If the new vertex count is different, we need to reallocate the buffer
    if (new_vertices.len != self.vertices.len) {
        self.allocator.free(self.vertices);
        self.vertices = try self.allocator.dupe(Vertex, new_vertices);

        // Recreate the buffer with new size
        glad.glBindBuffer(glad.GL_ARRAY_BUFFER, self.meta.VBO);
        glad.glBufferData(
            glad.GL_ARRAY_BUFFER,
            @intCast(self.vertices.len * @sizeOf(Vertex)),
            self.vertices.ptr,
            glad.GL_DYNAMIC_DRAW, // Use DYNAMIC_DRAW since we're updating frequently
        );
    } else {
        // Same size - just update the data
        // Copy new vertices into our buffer
        @memcpy(self.vertices, new_vertices);

        // Update OpenGL vertex buffer using SubData
        glad.glBindBuffer(glad.GL_ARRAY_BUFFER, self.meta.VBO);
        glad.glBufferSubData(
            glad.GL_ARRAY_BUFFER,
            0,
            @intCast(self.vertices.len * @sizeOf(Vertex)),
            self.vertices.ptr,
        );
    }
}

pub fn gen_draw(comptime draw_type: DrawType) draw {
    return struct {
        pub fn default_draw(mesh: *Self) void {
            glad.glBindVertexArray(mesh.meta.VAO);

            const gl_draw_type = draw_type.toGL();
            if (mesh.indices) |indices| {
                glad.glDrawElements(gl_draw_type, @intCast(indices.len), glad.GL_UNSIGNED_INT, null);
            } else {
                glad.glDrawArrays(gl_draw_type, 0, @intCast(mesh.vertices.len));
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

        var tangent = [3]f32{
            f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]),
            f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]),
            f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2]),
        };

        var bitangent = [3]f32{
            f * (-deltaUV2[0] * edge1[0] + deltaUV1[0] * edge2[0]),
            f * (-deltaUV2[0] * edge1[1] + deltaUV1[0] * edge2[1]),
            f * (-deltaUV2[0] * edge1[2] + deltaUV1[0] * edge2[2]),
        };

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
