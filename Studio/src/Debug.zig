const std = @import("std");
const Vertex = @import("Mesh.zig").Vertex;

const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("glad/glad.h");
    @cInclude("GLFW/glfw3.h");
});

pub fn printVertexShader(vbo: u32, size: usize) !void {
    const allocator = std.heap.page_allocator;
    const vertexData = try allocator.alloc(Vertex, size);

    c.glBindBuffer(c.GL_ARRAY_BUFFER, vbo);
    c.glGetBufferSubData(c.GL_ARRAY_BUFFER, 0, @intCast(size * @sizeOf(Vertex)), vertexData.ptr);
    c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);

    std.debug.print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", .{});
    std.debug.print("VBO => {d}\n", .{vbo});

    if (vertexData.len < 500) {
        for (0..vertexData.len) |i| {
            std.debug.print("~ {d} : \n{s}Position : {d}\n{s}Color : {d}\n", .{ i, " " ** 4, vertexData[i].position, " " ** 4, vertexData[i].color });
        }
    }
}
