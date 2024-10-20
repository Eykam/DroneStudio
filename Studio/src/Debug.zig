const std = @import("std");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("glad/glad.h"); // Include GLAD
    @cInclude("GLFW/glfw3.h");
    // Add OpenGL function declarations if needed
});

pub fn printVertexShader(vbo: u32, size: usize) !void {
    const allocator = std.heap.page_allocator;
    const vertexData = try allocator.alloc(f32, size);

    c.glBindBuffer(c.GL_ARRAY_BUFFER, vbo);
    c.glGetBufferSubData(c.GL_ARRAY_BUFFER, 0, @intCast(size * @sizeOf(f32)), vertexData.ptr);
    c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);

    std.debug.print("\n============================\n", .{});
    std.debug.print("Debugging ([x,y,z]) => {d}\n", .{vbo});

    for (0..vertexData.len) |i| {
        if ((i + 1) % 3 == 0) {
            std.debug.print(", {d}]\n", .{vertexData[i]});
        } else if (i % 3 == 0) {
            std.debug.print("~ {d} : [{d}, ", .{ i / 3, vertexData[i] });
        } else {
            std.debug.print("{d}", .{vertexData[i]});
        }
    }
}
