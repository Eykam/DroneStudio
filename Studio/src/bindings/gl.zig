const std = @import("std");

pub const glfw = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("GLFW/glfw3.h");
});

pub const glad = @cImport({
    @cInclude("glad/glad.h");
});
