const std = @import("std");
const Math = @import("Math.zig");
const Debug = @import("Debug.zig");
const Node = @import("Node.zig");
const gl = @import("gl.zig");
const glad = gl.glad;

const Self = @This();

node: ?*Node = null,
textureID: TextureID = TextureID{},
vertices: []Vertex,
indices: ?[]u32 = null,
meta: Metadata,
draw: draw,
drawType: glad.GLenum = glad.GL_TRIANGLES,

const draw = *const fn (mesh: *Self) void;

pub const Vertex = struct {
    position: [3]f32,
    color: [3]f32,
    texture: ?[2]f32 = null,
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

pub fn default_draw(mesh: *Self) void {
    glad.glBindVertexArray(mesh.meta.VAO);
    glad.glBindBuffer(glad.GL_ARRAY_BUFFER, mesh.meta.VBO);

    // Position attribute (location = 0)
    glad.glVertexAttribPointer(0, // location
        3, // (vec3)
        glad.GL_FLOAT, glad.GL_FALSE, @sizeOf(Vertex), // stride (size of entire vertex struct)
        null // offset for position
    );
    glad.glEnableVertexAttribArray(0);

    // Color attribute (location = 1)
    const color_offset = @offsetOf(Vertex, "color");
    glad.glVertexAttribPointer(1, // location
        3, // (vec3)
        glad.GL_FLOAT, glad.GL_FALSE, @sizeOf(Vertex), // stride (size of entire vertex struct)
        @ptrFromInt(color_offset));
    glad.glEnableVertexAttribArray(1);

    if (mesh.indices) |indices| {
        glad.glDrawElements(mesh.drawType, @intCast(indices.len), glad.GL_UNSIGNED_INT, null);
    } else {
        // When drawing without indices, we need to account for the full vertex struct size
        glad.glDrawArrays(mesh.drawType, 0, @intCast(mesh.vertices.len));
    }

    // Disable vertex attributes
    glad.glDisableVertexAttribArray(0);
    glad.glDisableVertexAttribArray(1);

    // Unbind the VAO
    glad.glBindVertexArray(0);
}

pub fn init(vertices: []Vertex, indices: ?[]u32, draw_fn: ?draw) !Self {
    var mesh = Self{
        .vertices = vertices,
        .indices = indices,
        .meta = Metadata{},
        .draw = draw_fn orelse default_draw,
    };

    // Initialize OpenGL buffers
    glad.glGenVertexArrays(1, &mesh.meta.VAO);
    glad.glGenBuffers(1, &mesh.meta.VBO);
    glad.glGenBuffers(1, &mesh.meta.IBO);

    glad.glBindVertexArray(mesh.meta.VAO);

    // Vertex Buffer
    glad.glBindBuffer(glad.GL_ARRAY_BUFFER, mesh.meta.VBO);
    glad.glBufferData(glad.GL_ARRAY_BUFFER, @intCast(vertices.len * @sizeOf(Vertex)), vertices.ptr, glad.GL_STATIC_DRAW);

    // Index Buffer
    if (indices) |ind| {
        glad.glBindBuffer(glad.GL_ELEMENT_ARRAY_BUFFER, mesh.meta.IBO);
        glad.glBufferData(glad.GL_ELEMENT_ARRAY_BUFFER, @intCast(ind.len * @sizeOf(u32)), ind.ptr, glad.GL_STATIC_DRAW);
    }

    // Position attribute
    glad.glVertexAttribPointer(0, 3, glad.GL_FLOAT, glad.GL_FALSE, @sizeOf(Vertex), null);
    glad.glEnableVertexAttribArray(0);

    // Color attribute
    const color_offset = @offsetOf(Vertex, "color");
    glad.glVertexAttribPointer(1, 3, glad.GL_FLOAT, glad.GL_FALSE, @sizeOf(Vertex), @ptrFromInt(color_offset));
    glad.glEnableVertexAttribArray(1);

    glad.glBindBuffer(glad.GL_ARRAY_BUFFER, 0);
    glad.glBindVertexArray(0);

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
