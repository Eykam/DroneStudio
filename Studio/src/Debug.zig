const std = @import("std");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("glad/glad.h"); // Include GLAD
    @cInclude("GLFW/glfw3.h");
    // Add OpenGL function declarations if needed
});

fn getVertexData(vbo: u32, size: usize) ![]f32 {
    const allocator = std.heap.page_allocator;
    const data = try allocator.alloc(f32, size);
    c.glBindBuffer(c.GL_ARRAY_BUFFER, vbo);
    c.glGetBufferSubData(c.GL_ARRAY_BUFFER, 0, @intCast(size * @sizeOf(f32)), data.ptr);
    c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
    return data;
}

// fn printVertexShader() !void {
//     const vertexData = try getVertexData(triangle.VBO, triangleVertices.len);

//     for (0..vertexData.len) |i| {
//         std.debug.print("Vertex {d}: {d}\n", .{ i, vertexData[i] });
//     }
// }
