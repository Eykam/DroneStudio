const std = @import("std");
const Mesh = @import("../../Mesh.zig");
const gl = @import("../../bindings/gl.zig");
const glad = gl.glad;

pub const FrameBuffer = struct {
    fbo: c_uint,
    texture: c_uint,
    depth_buffer: c_uint,
    width: c_int,
    height: c_int,

    pub fn init(width: c_int, height: c_int) !FrameBuffer {
        var fb: FrameBuffer = undefined;
        fb.width = width;
        fb.height = height;

        // Create framebuffer
        glad.glGenFramebuffers(1, &fb.fbo);
        glad.glBindFramebuffer(glad.GL_FRAMEBUFFER, fb.fbo);

        // Create texture to render to
        glad.glGenTextures(1, &fb.texture);
        glad.glBindTexture(glad.GL_TEXTURE_2D, fb.texture);
        glad.glTexImage2D(glad.GL_TEXTURE_2D, 0, glad.GL_RGBA, width, height, 0, glad.GL_RGBA, glad.GL_UNSIGNED_BYTE, null);
        glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MIN_FILTER, glad.GL_LINEAR);
        glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MAG_FILTER, glad.GL_LINEAR);
        glad.glBindTexture(glad.GL_TEXTURE_2D, 0);

        // Attach texture to framebuffer
        glad.glFramebufferTexture2D(glad.GL_FRAMEBUFFER, glad.GL_COLOR_ATTACHMENT0, glad.GL_TEXTURE_2D, fb.texture, 0);

        // Create depth buffer
        glad.glGenRenderbuffers(1, &fb.depth_buffer);
        glad.glBindRenderbuffer(glad.GL_RENDERBUFFER, fb.depth_buffer);
        glad.glRenderbufferStorage(glad.GL_RENDERBUFFER, glad.GL_DEPTH_COMPONENT, width, height);
        glad.glBindRenderbuffer(glad.GL_RENDERBUFFER, 0);

        // Attach depth buffer to framebuffer
        glad.glFramebufferRenderbuffer(glad.GL_FRAMEBUFFER, glad.GL_DEPTH_ATTACHMENT, glad.GL_RENDERBUFFER, fb.depth_buffer);

        const status = glad.glCheckFramebufferStatus(glad.GL_FRAMEBUFFER);
        if (status != glad.GL_FRAMEBUFFER_COMPLETE) {
            std.debug.print("Framebuffer is not complete! Status: 0x{X}\n", .{status});
            switch (status) {
                glad.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT => std.debug.print("Incomplete attachment\n", .{}),
                glad.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => std.debug.print("Missing attachment\n", .{}),
                glad.GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER => std.debug.print("Incomplete draw buffer\n", .{}),
                glad.GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER => std.debug.print("Incomplete read buffer\n", .{}),
                glad.GL_FRAMEBUFFER_UNSUPPORTED => std.debug.print("Unsupported format combination\n", .{}),
                else => std.debug.print("Unknown framebuffer status error\n", .{}),
            }
            return error.FramebufferIncomplete;
        }

        // Unbind framebuffer
        glad.glBindFramebuffer(glad.GL_FRAMEBUFFER, 0);

        return fb;
    }

    pub fn deinit(self: *FrameBuffer) void {
        glad.glDeleteFramebuffers(1, &self.fbo);
        glad.glDeleteTextures(1, &self.texture);
        glad.glDeleteRenderbuffers(1, &self.depth_buffer);
    }
};

pub const Viewport = struct {
    const Self = @This();

    name: [:0]const u8, // Name of the viewport
    fbo: FrameBuffer, // Framebuffer for rendering this viewport
    shader_program: c_uint,
    enabled: bool,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, width: i32, height: i32) !Self {
        // Create a copy of the name string
        const name_copy = try allocator.dupeZ(u8, name);

        // Initialize the framebuffer
        const fbo = try FrameBuffer.init(width, height);

        // Create shader program for rendering viewport
        const shader_program = try createShaderProgram("../../shaders/miniview_vertex.glsl", "../../shaders/miniview_fragment.glsl");

        const viewport = Self{
            .name = name_copy,
            .fbo = fbo,
            .shader_program = shader_program,
            .enabled = true,
            .allocator = allocator,
        };

        return viewport;
    }

    pub fn deinit(self: *Self) void {
        // Free OpenGL resources
        self.fbo.deinit();
        glad.glDeleteProgram(self.shader_program);

        // Free the name
        self.allocator.free(self.name);
    }

    // pub fn render(self: *Self, window_width: f32, window_height: f32) void {
    //     if (!self.enabled) return;

    //     // Calculate viewport dimensions in pixels
    //     const x = self.position[0] * window_width;
    //     const y = self.position[1] * window_height;
    //     const width = self.size[0] * window_width;
    //     const height = self.size[1] * window_height;

    //     // Save current OpenGL state
    //     var last_viewport: [4]c_int = undefined;
    //     glad.glGetIntegerv(glad.GL_VIEWPORT, &last_viewport);

    //     var last_program: c_int = 0;
    //     glad.glGetIntegerv(glad.GL_CURRENT_PROGRAM, &last_program);

    //     var last_blend_enabled: c_int = 0;
    //     glad.glGetIntegerv(glad.GL_BLEND, &last_blend_enabled);

    //     var last_depth_test_enabled: c_int = 0;
    //     glad.glGetIntegerv(glad.GL_DEPTH_TEST, &last_depth_test_enabled);

    //     // Disable depth test for 2D rendering
    //     glad.glDisable(glad.GL_DEPTH_TEST);

    //     // Enable blending for transparency
    //     glad.glEnable(glad.GL_BLEND);
    //     glad.glBlendFunc(glad.GL_SRC_ALPHA, glad.GL_ONE_MINUS_SRC_ALPHA);

    //     // Set viewport
    //     glad.glViewport(
    //         @intFromFloat(x),
    //         @intFromFloat(y),
    //         @intFromFloat(width),
    //         @intFromFloat(height),
    //     );

    //     // Use the shader program
    //     glad.glUseProgram(self.shader_program);

    //     // Bind texture
    //     glad.glActiveTexture(glad.GL_TEXTURE0);
    //     glad.glBindTexture(glad.GL_TEXTURE_2D, self.fbo.texture);

    //     // Set texture uniform
    //     const textureLoc = glad.glGetUniformLocation(self.shader_program, "viewTexture");
    //     if (textureLoc != -1) {
    //         glad.glUniform1i(textureLoc, 0);
    //     }

    //     // Draw the quad using the mesh's draw function
    //     self.mesh._draw(self.mesh);

    //     // Add border
    //     const border_width: f32 = 2.0;
    //     const border_color = imgui.igColorConvertFloat4ToU32(.{ .x = 1.0, .y = 1.0, .z = 1.0, .w = 0.8 });

    //     const draw_list = imgui.igGetWindowDrawList();
    //     imgui.ImDrawList_AddRect(
    //         draw_list,
    //         .{ .x = x, .y = y },
    //         .{ .x = x + width, .y = y + height },
    //         border_color,
    //         0.0,
    //         imgui.ImDrawFlags_None,
    //         border_width,
    //     );

    //     // Draw the Viewport name in the top-left corner
    //     const text_padding = 5.0;
    //     imgui.ImDrawList_AddText_Vec2(
    //         draw_list,
    //         .{ .x = x + text_padding, .y = y + text_padding },
    //         imgui.igColorConvertFloat4ToU32(.{ .x = 1.0, .y = 1.0, .z = 1.0, .w = 1.0 }),
    //         self.name.ptr,
    //         null,
    //     );

    //     // Restore previous OpenGL state
    //     if (last_depth_test_enabled == glad.GL_TRUE) {
    //         glad.glEnable(glad.GL_DEPTH_TEST);
    //     } else {
    //         glad.glDisable(glad.GL_DEPTH_TEST);
    //     }

    //     if (last_blend_enabled == glad.GL_TRUE) {
    //         glad.glEnable(glad.GL_BLEND);
    //     } else {
    //         glad.glDisable(glad.GL_BLEND);
    //     }

    //     glad.glViewport(
    //         last_viewport[0],
    //         last_viewport[1],
    //         last_viewport[2],
    //         last_viewport[3],
    //     );
    //     glad.glUseProgram(@intCast(last_program));
    // }
};

fn readShaderSource(comptime path: []const u8) ![]const u8 {
    const file: []const u8 = @embedFile(path);
    std.debug.print("Source Length: {}\n", .{file.len});
    return file;
}

fn compileShader(shaderType: u32, source: []const u8) !u32 {
    const shader = glad.glCreateShader(shaderType);
    if (shader == 0) {
        std.debug.print("Failed to compile shader\n", .{});
        return error.UnableToCreateShader;
    }

    const src_ptr: [*c]const u8 = @ptrCast(@alignCast(source.ptr));
    const src_len = source.len;

    std.debug.print("Compiling Shader...\n", .{});
    glad.glShaderSource(shader, 1, @ptrCast(&src_ptr), @ptrCast(&src_len));
    glad.glCompileShader(shader);

    // Check for compilation errors
    var success: u32 = 0;
    glad.glGetShaderiv(shader, glad.GL_COMPILE_STATUS, @alignCast(@ptrCast(&success)));
    if (success == 0) {
        var infoLog: [512]u8 = undefined;
        glad.glGetShaderInfoLog(shader, 512, null, &infoLog);
        std.debug.print("ERROR::SHADER::COMPILATION_FAILED\n{any}\n", .{infoLog});
        return error.ShaderCompilationFailed;
    }

    return shader;
}

pub fn createShaderProgram(comptime vertexPath: []const u8, comptime fragmentPath: []const u8) !u32 {
    std.debug.print("Initializing Vertex Shader...\n", .{});
    std.debug.print("Reading Vertex Shader from Source...\n", .{});
    const vertexSource = try readShaderSource(vertexPath);
    const vertexShader = compileShader(glad.GL_VERTEX_SHADER, vertexSource) catch |err| {
        std.debug.print("Failed to read vertex shader '{s}': {any}\n", .{ vertexPath, err });
        return error.ShaderCompilationFailed;
    };
    std.debug.print("\n", .{});

    std.debug.print("Initializing Fragment Shader...\n", .{});
    std.debug.print("Reading Fragment Shader from Source...\n", .{});
    const fragmentSource = try readShaderSource(fragmentPath);
    const fragmentShader = compileShader(glad.GL_FRAGMENT_SHADER, fragmentSource) catch |err| {
        std.debug.print("Failed to read fragment shader '{s}': {any}\n", .{ fragmentPath, err });
        return error.ShaderCompilationFailed;
    };
    std.debug.print("\n", .{});

    std.debug.print("Creating ShaderProgram...\n", .{});
    const shaderProgram = glad.glCreateProgram();
    if (shaderProgram == 0) {
        std.debug.print("Failed to create Shader Program\n", .{});
        return error.UnableToCreateProgram;
    }

    std.debug.print("Attaching ShaderProgram to openGL...\n", .{});
    glad.glAttachShader(shaderProgram, vertexShader);
    glad.glAttachShader(shaderProgram, fragmentShader);
    glad.glLinkProgram(shaderProgram);

    // Check for linking errors
    var success: u32 = 0;
    glad.glGetProgramiv(shaderProgram, glad.GL_LINK_STATUS, @ptrCast(@alignCast(&success)));

    if (success == 0) {
        var infoLog: [512]u8 = undefined;
        glad.glGetProgramInfoLog(shaderProgram, 512, null, &infoLog);
        std.debug.print("ERROR::PROGRAM::LINKING_FAILED\n{any}\n", .{infoLog});
        return error.ShaderLinkingFailed;
    }

    std.debug.print("Running cleanup on Shaders...\n", .{});
    // Shaders can be deleted after linking
    glad.glDeleteShader(vertexShader);
    glad.glDeleteShader(fragmentShader);

    return shaderProgram;
}
