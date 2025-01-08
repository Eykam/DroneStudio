const std = @import("std");
const Math = @import("Math.zig");
const Debug = @import("Debug.zig");
const Node = @import("Node.zig");
const gl = @import("bindings/gl.zig");
const glad = gl.glad;

const Self = @This();

allocator: std.mem.Allocator,
node: ?*Node = null,
textureID: TextureID = TextureID{},
vertices: []Vertex,
indices: ?[]u32 = null,
meta: Metadata,
_draw: draw,
drawType: glad.GLenum = glad.GL_TRIANGLES,

pub const draw = *const fn (mesh: *Self) void;

pub const Vertex = struct {
    position: [3]f32,
    color: [3]f32,
    texture: ?[2]f32 = null,
    alpha: ?f32 = null,
};

pub const Metadata = struct {
    VAO: u32 = 0,
    VBO: u32 = 0,
    IBO: u32 = 0,
};

pub const TextureID = struct {
    y: c_uint = 0,
    uv: c_uint = 0,
};

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
    // Position attribute
    glad.glVertexAttribPointer(
        0, // location
        3, // size (vec3)
        glad.GL_FLOAT,
        glad.GL_FALSE,
        @sizeOf(Vertex),
        null,
    );
    glad.glEnableVertexAttribArray(0);

    // Color attribute
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

    // Verify the attributes are enabled
    var enabled: c_int = undefined;
    glad.glGetVertexAttribiv(0, glad.GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled);
    if (enabled == 0) {
        std.debug.print("Position attribute not enabled\n", .{});
    }
    glad.glGetVertexAttribiv(1, glad.GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled);
    if (enabled == 0) {
        std.debug.print("Color attribute not enabled\n", .{});
    }

    glad.glBindVertexArray(0);

    // Final error check
    const final_error = glad.glGetError();
    if (final_error != glad.GL_NO_ERROR) {
        std.debug.print("OpenGL error at end of mesh init: 0x{x}\n", .{final_error});
    } else {
        // std.debug.print("Mesh initialized successfully. VAO: {}, VBO: {}\n", .{ self.meta.VAO, self.meta.VBO });
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

pub fn setFaceColor(self: *Self, face_index: usize, color: [3]f32) void {
    const vertices_per_face = 3;
    const start_idx = face_index * vertices_per_face;
    const end_idx = start_idx + vertices_per_face;

    for (start_idx..end_idx) |i| {
        const vertex_idx = self.indices[i];
        self.vertices[vertex_idx].color = color;
    }

    glad.glBindBuffer(glad.GL_ARRAY_BUFFER, self.meta.VBO);
    glad.glBufferData(glad.GL_ARRAY_BUFFER, @intCast(self.vertices.len * @sizeOf(Vertex)), self.vertices.ptr, glad.GL_STATIC_DRAW);
}

pub fn setColor(self: *Self, color: [3]f32) void {
    for (self.vertices) |*vertex| {
        vertex.color = color;
    }

    glad.glBindBuffer(glad.GL_ARRAY_BUFFER, self.meta.VBO);
    glad.glBufferData(glad.GL_ARRAY_BUFFER, @intCast(self.vertices.len * @sizeOf(Vertex)), self.vertices.ptr, glad.GL_STATIC_DRAW);
}

pub fn debug(self: Self) void {
    Debug.printVertexShader(self.meta.VBO, self.vertices.len) catch |err| {
        std.debug.print("Failed to debug vertex shader {any}\n", .{err});
    };
}
