const Transformations = @import("Transformations.zig");
const Debug = @import("Debug.zig");
const std = @import("std");

const c = @cImport({
    @cInclude("glad/glad.h");
});

const Self = @This();

vertices: []Vertex,
indices: ?[]u32 = null,
meta: Metadata,
draw: draw,
drawType: c.GLenum = c.GL_TRIANGLES,

const draw = *const fn (mesh: *Self) void;

pub const Vertex = struct {
    position: [3]f32,
    color: [3]f32,
};

pub const Metadata = struct {
    VAO: u32 = 0,
    VBO: u32 = 0,
    IBO: u32 = 0,
};

pub fn default_draw(mesh: *Self) void {
    c.glBindVertexArray(mesh.meta.VAO);
    c.glBindBuffer(c.GL_ARRAY_BUFFER, mesh.meta.VBO);

    // Position attribute (location = 0)
    c.glVertexAttribPointer(0, // location
        3, // (vec3)
        c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex), // stride (size of entire vertex struct)
        null // offset for position
    );
    c.glEnableVertexAttribArray(0);

    // Color attribute (location = 1)
    const color_offset = @offsetOf(Vertex, "color");
    c.glVertexAttribPointer(1, // location
        3, // (vec3)
        c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex), // stride (size of entire vertex struct)
        @ptrFromInt(color_offset));
    c.glEnableVertexAttribArray(1);

    // Draw the mesh
    if (mesh.indices) |indices| {
        c.glDrawElements(mesh.drawType, @intCast(indices.len), c.GL_UNSIGNED_INT, null);
    } else {
        // When drawing without indices, we need to account for the full vertex struct size
        c.glDrawArrays(mesh.drawType, 0, @intCast(mesh.vertices.len));
    }

    // Disable vertex attributes
    c.glDisableVertexAttribArray(0);
    c.glDisableVertexAttribArray(1);

    // Unbind the VAO
    c.glBindVertexArray(0);
}

pub fn init(vertices: []Vertex, indices: ?[]u32, draw_fn: ?draw) !Self {
    var mesh = Self{
        .vertices = vertices,
        .indices = indices,
        .meta = Metadata{},
        .draw = draw_fn orelse default_draw,
    };

    // Initialize OpenGL buffers
    c.glGenVertexArrays(1, &mesh.meta.VAO);
    c.glGenBuffers(1, &mesh.meta.VBO);
    c.glGenBuffers(1, &mesh.meta.IBO);

    c.glBindVertexArray(mesh.meta.VAO);

    // Vertex Buffer
    c.glBindBuffer(c.GL_ARRAY_BUFFER, mesh.meta.VBO);
    c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(vertices.len * @sizeOf(Vertex)), vertices.ptr, c.GL_STATIC_DRAW);

    // Index Buffer
    if (indices) |ind| {
        c.glBindBuffer(c.GL_ELEMENT_ARRAY_BUFFER, mesh.meta.IBO);
        c.glBufferData(c.GL_ELEMENT_ARRAY_BUFFER, @intCast(ind.len * @sizeOf(u32)), ind.ptr, c.GL_STATIC_DRAW);
    }

    // Position attribute
    c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex), null);
    c.glEnableVertexAttribArray(0);

    // Color attribute
    const color_offset = @offsetOf(Vertex, "color");
    c.glVertexAttribPointer(1, 3, c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex), @ptrFromInt(color_offset));
    c.glEnableVertexAttribArray(1);

    c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
    c.glBindVertexArray(0);

    return mesh;
}

pub fn setFaceColor(self: *Self, face_index: usize, color: [3]f32) void {
    const vertices_per_face = 3;
    const start_idx = face_index * vertices_per_face;
    const end_idx = start_idx + vertices_per_face;

    for (start_idx..end_idx) |i| {
        const vertex_idx = self.indices[i];
        self.vertices[vertex_idx].color = color;
    }

    // Update vertex buffer
    c.glBindBuffer(c.GL_ARRAY_BUFFER, self.meta.VBO);
    c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(self.vertices.len * @sizeOf(Vertex)), self.vertices.ptr, c.GL_STATIC_DRAW);
}

pub fn setColor(self: *Self, color: [3]f32) void {
    for (self.vertices) |*vertex| {
        vertex.color = color;
    }

    // Update vertex buffer
    c.glBindBuffer(c.GL_ARRAY_BUFFER, self.meta.VBO);
    c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(self.vertices.len * @sizeOf(Vertex)), self.vertices.ptr, c.GL_STATIC_DRAW);
}

pub fn debug(self: Self) void {
    Debug.printVertexShader(self.meta.VBO, self.vertices.len) catch |err| {
        std.debug.print("Failed to debug vertex shader {any}\n", .{err});
    };
}
