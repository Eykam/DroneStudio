const std = @import("std");
const Vertex = @import("Shape.zig").Vertex;

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

    std.debug.print("\n============================\n", .{});
    std.debug.print("Debugging ([x,y,z]) => {d}\n", .{vbo});

    for (0..vertexData.len) |i| {
        if ((i + 1) % 3 == 0) {
            std.debug.print(", {any}]\n", .{vertexData[i]});
        } else if (i % 3 == 0) {
            std.debug.print("~ {d} : [{any}, ", .{ i / 3, vertexData[i] });
        } else {
            std.debug.print("{any}", .{vertexData[i]});
        }
    }
}
