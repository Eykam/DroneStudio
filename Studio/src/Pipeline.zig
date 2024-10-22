// src/Scene.zig
const std = @import("std");
const Shape = @import("Shape.zig");
const Transformations = @import("Transformations.zig");
const Vec3 = Transformations.Vec3;
const File = std.fs.File;
const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("glad/glad.h"); // Ensure GLAD is included
    @cInclude("GLFW/glfw3.h");
});

const GSLWError = error{ FailedToCreateWindow, FailedToInitialize };
const ShaderError = error{ UnableToCreateShader, ShaderCompilationFailed, UnableToCreateProgram, ShaderLinkingFailed, UnableToCreateWindow };

// Function to read shader source from a file
inline fn readShaderSource(comptime path: []const u8) ![]const u8 {
    const file: []const u8 = @embedFile(path);
    std.debug.print("Source Length: {}\n", .{file.len});
    return file;
}

// Function to compile a shader
fn compileShader(shaderType: u32, source: []const u8) !u32 {
    const shader = c.glCreateShader(shaderType);
    if (shader == 0) {
        std.debug.print("Failed to compile shader\n", .{});
        return ShaderError.UnableToCreateShader;
    }

    const src_ptr: [*c]const u8 = @ptrCast(@alignCast(source.ptr));
    const src_len = source.len;

    std.debug.print("Compiling Shader...\n", .{});
    c.glShaderSource(shader, 1, @ptrCast(&src_ptr), @ptrCast(&src_len));
    c.glCompileShader(shader);

    // Check for compilation errors
    var success: u32 = 0;
    c.glGetShaderiv(shader, c.GL_COMPILE_STATUS, @alignCast(@ptrCast(&success)));
    if (success == 0) {
        var infoLog: [512]u8 = undefined;
        c.glGetShaderInfoLog(shader, 512, null, &infoLog);
        std.debug.print("ERROR::SHADER::COMPILATION_FAILED\n{any}\n", .{infoLog});
        return ShaderError.ShaderCompilationFailed;
    }

    return shader;
}

// Function to create a shader program
pub fn createShaderProgram(comptime vertexPath: []const u8, comptime fragmentPath: []const u8) !u32 {
    std.debug.print("Initializing Vertex Shader...\n", .{});
    std.debug.print("Reading Vertex Shader from Source...\n", .{});
    const vertexSource = try readShaderSource(vertexPath);
    const vertexShader = compileShader(c.GL_VERTEX_SHADER, vertexSource) catch |err| {
        std.debug.print("Failed to read vertex shader '{s}': {any}\n", .{ vertexPath, err });
        return ShaderError.ShaderCompilationFailed;
    };
    std.debug.print("\n", .{});

    std.debug.print("Initializing Fragment Shader...\n", .{});
    std.debug.print("Reading Fragment Shader from Source...\n", .{});
    const fragmentSource = try readShaderSource(fragmentPath);
    const fragmentShader = compileShader(c.GL_FRAGMENT_SHADER, fragmentSource) catch |err| {
        std.debug.print("Failed to read fragment shader '{s}': {any}\n", .{ fragmentPath, err });
        return ShaderError.ShaderCompilationFailed;
    };
    std.debug.print("\n", .{});

    std.debug.print("Creating ShaderProgram...\n", .{});
    const shaderProgram = c.glCreateProgram();
    if (shaderProgram == 0) {
        std.debug.print("Failed to create Shader Program\n", .{});
        return ShaderError.UnableToCreateProgram;
    }

    std.debug.print("Attaching ShaderProgram to openGL...\n", .{});
    c.glAttachShader(shaderProgram, vertexShader);
    c.glAttachShader(shaderProgram, fragmentShader);
    c.glLinkProgram(shaderProgram);

    // Check for linking errors
    var success: u32 = 0;
    c.glGetProgramiv(shaderProgram, c.GL_LINK_STATUS, @ptrCast(@alignCast(&success)));

    if (success == 0) {
        var infoLog: [512]u8 = undefined;
        c.glGetProgramInfoLog(shaderProgram, 512, null, &infoLog);
        std.debug.print("ERROR::PROGRAM::LINKING_FAILED\n{any}\n", .{infoLog});
        return ShaderError.ShaderLinkingFailed;
    }

    std.debug.print("Running cleanup on Shaders...\n", .{});
    // Shaders can be deleted after linking
    c.glDeleteShader(vertexShader);
    c.glDeleteShader(fragmentShader);

    return shaderProgram;
}

fn checkOpenGLError(caller: []const u8) void {
    var err: u32 = c.glGetError();
    while (err != c.GL_NO_ERROR) {
        std.debug.print("OpenGL Error: {x} from Caller: {s}\n", .{ err, caller });
        err = c.glGetError();
    }
}

pub const AppState = struct {
    camera_pos: Vec3,
    camera_front: Vec3,
    camera_up: Vec3,
    rotation_x: f32 = 0.0,
    rotation_y: f32 = 0.0,
    last_mouse_x: f64 = 0.0,
    last_mouse_y: f64 = 0.0,
    first_mouse: bool = true,
    zoom: f32 = 90.0,
    keys: [1024]bool = .{false} ** 1024,
    last_frame_time: f64 = 0.0, // Time of the last frame
    delta_time: f32 = 0.0, // Time between current frame and last frame
};

pub const Scene = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    objects: std.StringHashMap(Shape.Object),
    shaderProgram: u32,
    width: f32,
    height: f32,
    appState: AppState,

    pub fn init(allocator: std.mem.Allocator, window: ?*c.struct_GLFWwindow) !Self {
        if (window == null) {
            std.debug.print("Failed to create GLFW window\n", .{});
            return GSLWError.FailedToCreateWindow;
        }

        var width: i32 = undefined;
        var height: i32 = undefined;
        c.glfwGetWindowSize(window.?, &width, &height);

        // Make the window's context current
        c.glfwMakeContextCurrent(window);
        c.glfwSwapInterval(1);

        // Initialize OpenGL Loader
        std.debug.print("Loading Glad...\n", .{});
        if (c.gladLoadGLLoader(@ptrCast(&c.glfwGetProcAddress)) == 0) {
            std.debug.print("Failed to initialize GLAD\n", .{});
            return GSLWError.FailedToInitialize;
        }

        std.debug.print("Initializing viewport...\n\n", .{});
        c.glViewport(0, 0, width, height);
        c.glEnable(c.GL_DEPTH_TEST);

        const shaderProgram = try createShaderProgram("shaders/vertex_shader.glsl", "shaders/fragment_shader.glsl");
        c.glUseProgram(shaderProgram);

        var currentProgram: u32 = 0;
        c.glGetIntegerv(c.GL_CURRENT_PROGRAM, @ptrCast(&currentProgram));
        if (currentProgram != shaderProgram) {
            std.debug.print("Shader program not active!\n", .{});
        }

        checkOpenGLError("Uniform Setup");

        var numActiveAttributes: c_int = 0;
        c.glGetProgramiv(shaderProgram, c.GL_ACTIVE_ATTRIBUTES, &numActiveAttributes);

        std.debug.print("\nGetting Attributes\n", .{});
        std.debug.print("Number of active vertex attributes: {}\n", .{numActiveAttributes});

        for (0..@intCast(numActiveAttributes)) |i| {
            var nameBuffer: [256]u8 = undefined;
            var length: c_int = 0;
            var size: c_int = 0;
            var var_type: c_uint = 0;

            c.glGetActiveAttrib(
                shaderProgram,
                @intCast(i),
                nameBuffer.len,
                &length,
                &size,
                &var_type,
                &nameBuffer,
            );

            std.debug.print(" - Attribute {d}: {s} == Size: {d} == Type: {x}\n", .{ i, nameBuffer[0..@as(u32, @intCast(length))], size, var_type });
        }

        c.glDepthFunc(c.GL_LESS);

        return Self{
            .allocator = allocator,
            .objects = std.StringHashMap(Shape.Object).init(allocator),
            .shaderProgram = shaderProgram,
            .width = @floatFromInt(width),
            .height = @floatFromInt(height),
            .appState = AppState{
                .camera_pos = Vec3{ .x = 0.0, .y = 1.0, .z = 0.0 },
                .camera_front = Vec3{ .x = 0.0, .y = 0.0, .z = -1.0 },
                .camera_up = Vec3{ .x = 0.0, .y = 1.0, .z = 0.0 },
            },
        };
    }

    pub fn deinit(self: *Self) void {
        // Delete OpenGL resources for each object
        var it = self.objects.iterator();
        while (it.next()) |entry| {
            const object = entry.value_ptr;
            c.glDeleteVertexArrays(1, &object.meta.VAO);
            c.glDeleteBuffers(1, &object.meta.VBO);
            if (object.indices.len > 0) {
                c.glDeleteBuffers(1, &object.meta.EBO);
            }
        }

        self.objects.deinit();

        c.glDeleteProgram(self.shaderProgram);
    }

    pub fn setupCallbacks(self: *Self, window: ?*c.struct_GLFWwindow) void {
        if (window == null) return;

        c.glfwSetWindowUserPointer(window, @ptrCast(self));

        const ptr = c.glfwGetWindowUserPointer(window);
        if (ptr == null) {
            std.debug.print("Failed to set window user pointer\n", .{});
            return;
        }

        // Set callbacks
        _ = c.glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
        _ = c.glfwSetCursorPosCallback(window, mouseCallback);
        _ = c.glfwSetKeyCallback(window, keyCallback);
        _ = c.glfwSetScrollCallback(window, scrollCallback); // Set the new scroll callback

        const current_mode = c.glfwGetInputMode(window, c.GLFW_CURSOR);
        if (current_mode != c.GLFW_CURSOR_DISABLED) {
            // Try setting it again
            c.glfwSetInputMode(window, c.GLFW_CURSOR, c.GLFW_CURSOR_DISABLED);

            // Check if it worked this time
            const new_mode = c.glfwGetInputMode(window, c.GLFW_CURSOR);
            if (new_mode != c.GLFW_CURSOR_DISABLED) {
                std.debug.print("Failed to disable cursor in setupCallbacks\n", .{});
            }
        }
    }

    pub fn updateProjection(self: *Self) [16]f32 {
        return Transformations.perspective(self.appState.zoom, self.width / self.height, 0.1, 100.0);
    }

    pub fn addObject(self: *Self, name: []const u8, object: Shape.Object) !void {
        try self.objects.put(name, object);
    }

    pub fn render(self: *Self, window: ?*c.struct_GLFWwindow) void {
        c.glClearColor(0.15, 0.15, 0.15, 1.0);
        c.glClear(c.GL_COLOR_BUFFER_BIT | c.GL_DEPTH_BUFFER_BIT);

        // Use the shader program
        c.glUseProgram(self.shaderProgram);

        const target = Vec3.add(self.appState.camera_pos, self.appState.camera_front);
        var view = Transformations.lookAt(self.appState.camera_pos, target, self.appState.camera_up);

        // Use the current projection matrix that accounts for window size
        const currentProjection = self.updateProjection();

        // Set common uniforms
        const viewLoc = c.glGetUniformLocation(self.shaderProgram, "uView");
        const projectionLoc = c.glGetUniformLocation(self.shaderProgram, "uProjection");
        if (viewLoc != -1) {
            c.glUniformMatrix4fv(viewLoc, 1, c.GL_FALSE, &view);
        }
        if (projectionLoc != -1) {
            c.glUniformMatrix4fv(projectionLoc, 1, c.GL_FALSE, &currentProjection);
        }

        // Iterate through all objects in the hash map
        var it = self.objects.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.draw) |drawFunction| {
                drawFunction();
            } else {
                const object = entry.value_ptr.*;

                // Set object-specific uniforms
                const modelLoc = c.glGetUniformLocation(self.shaderProgram, "uModel");
                const colorLoc = c.glGetUniformLocation(self.shaderProgram, "uColor");

                if (modelLoc != -1) {
                    c.glUniformMatrix4fv(modelLoc, 1, c.GL_FALSE, &object.modelMatrix);
                }
                if (colorLoc != -1) {
                    c.glUniform3f(colorLoc, object.color.r, object.color.g, object.color.b);
                }

                // Bind the object's VAO
                c.glBindVertexArray(object.meta.VAO);

                // Draw the object
                if (object.indices.len > 0) {
                    c.glDrawElements(object.drawType, @intCast(object.indices.len), c.GL_UNSIGNED_INT, null);
                } else {
                    c.glDrawArrays(object.drawType, 0, @intCast(object.vertices.len / 3));
                }

                // Unbind the VAO
                c.glBindVertexArray(0);
            }

            // Swap front and back buffers
        }

        c.glfwSwapBuffers(window);
        c.glfwPollEvents();
    }
};

fn framebufferSizeCallback(window: ?*c.GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
    if (window == null) return;

    const user_ptr = c.glfwGetWindowUserPointer(window);
    if (user_ptr == null) {
        std.debug.print("Error: Window user pointer is null in framebufferSizeCallback\n", .{});
        return;
    }

    // Retrieve the Scene instance from the user pointer
    const scene = @as(*Scene, @ptrCast(@alignCast(c.glfwGetWindowUserPointer(window))));

    // Update the viewport
    c.glViewport(0, 0, width, height);

    // Update Scene's width and height
    scene.width = @floatFromInt(width);
    scene.height = @floatFromInt(height);
}

fn mouseCallback(window: ?*c.struct_GLFWwindow, xpos: f64, ypos: f64) callconv(.C) void {
    if (window == null) return;

    // Only process mouse input if window is focused
    if (c.glfwGetWindowAttrib(window, c.GLFW_FOCUSED) != c.GLFW_TRUE) {
        return;
    }

    // Retrieve the Scene instance from the user pointer

    const scene = @as(*Scene, @ptrCast(@alignCast(c.glfwGetWindowUserPointer(window))));

    if (scene.appState.first_mouse) {
        scene.appState.last_mouse_x = xpos;
        scene.appState.last_mouse_y = ypos;
        scene.appState.first_mouse = false;
        return;
    }

    const xoffset = xpos - scene.appState.last_mouse_x;
    const yoffset = scene.appState.last_mouse_y - ypos; // Reversed Y

    scene.appState.last_mouse_x = xpos;
    scene.appState.last_mouse_y = ypos;

    const sensitivity_y: f32 = 7;
    const sensitivity_x: f32 = sensitivity_y * scene.width / scene.width;

    scene.appState.rotation_x += @as(f32, @floatCast(xoffset)) * sensitivity_x;
    scene.appState.rotation_y += @as(f32, @floatCast(yoffset)) * sensitivity_y;

    const yaw = Transformations.radians(scene.appState.rotation_x);
    const pitch = Transformations.radians(scene.appState.rotation_y);

    scene.appState.camera_front = Vec3.from_angles(yaw, pitch);
}

fn keyCallback(window: ?*c.struct_GLFWwindow, key: c_int, scancode: c_int, action: c_int, mods: c_int) callconv(.C) void {
    if (window == null) return;

    _ = scancode;
    _ = mods;

    // Retrieve the Scene instance from the user pointer
    const scene = @as(*Scene, @ptrCast(@alignCast(c.glfwGetWindowUserPointer(window))));

    if (key < 0 or key >= 1024) return; // Prevent out-of-bounds

    if (action == c.GLFW_PRESS) {
        scene.appState.keys[@intCast(key)] = true;
    } else if (action == c.GLFW_RELEASE) {
        scene.appState.keys[@intCast(key)] = false;
    }

    if (action == c.GLFW_PRESS or action == c.GLFW_REPEAT) {
        switch (key) {
            c.GLFW_KEY_ESCAPE => {
                c.glfwSetWindowShouldClose(window, 1);
            },
            c.GLFW_KEY_UP => {
                // Zoom in (narrower FOV)
                scene.appState.zoom -= 1.0;
                if (scene.appState.zoom < 1.0) scene.appState.zoom = 1.0;
            },
            c.GLFW_KEY_DOWN => {
                // Zoom out (wider FOV)
                scene.appState.zoom += 1.0;
            },
            else => {},
        }
    }
    // Handle additional keys here
}

fn scrollCallback(window: ?*c.struct_GLFWwindow, xoffset: f64, yoffset: f64) callconv(.C) void {
    if (window == null) return;

    _ = xoffset;

    const scene = @as(*Scene, @ptrCast(@alignCast(c.glfwGetWindowUserPointer(window))));

    // Define zoom sensitivity
    const zoomSensitivity: f32 = 0.1;
    const newZoom = scene.appState.zoom - @as(f32, @floatCast(yoffset)) * zoomSensitivity * scene.appState.zoom;

    // Clamp the zoom level to prevent extreme zooming

    if (newZoom < 1.0) {
        scene.appState.zoom = 1.0;
    } else if (newZoom >= 120) {
        scene.appState.zoom = 120;
    } else {
        scene.appState.zoom = newZoom;
    }

    std.debug.print("yOffset: {d}\n", .{yoffset});
    std.debug.print("Zoom Level: {d}\n", .{newZoom});
}
