const std = @import("std");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("glad/glad.h");
    @cInclude("GLFW/glfw3.h");
});

pub fn glCheckError(string: []const u8) void {
    const err = c.glGetError();
    if (err != c.GL_NO_ERROR) {
        std.debug.print("OpenGL Error at {s}: {}\n", .{ string, err });
    }
}
