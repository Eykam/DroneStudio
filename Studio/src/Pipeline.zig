// src/Scene.zig
const std = @import("std");
const Shape = @import("Shape.zig");
const Transformations = @import("Transformations.zig");
const Vec3 = Transformations.Vec3;
const File = std.fs.File;
const Camera = @import("Camera.zig");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("glad/glad.h"); // Ensure GLAD is included
    @cInclude("GLFW/glfw3.h");
});

const GSLWError = error{ FailedToCreateWindow, FailedToInitialize };
const ShaderError = error{ UnableToCreateShader, ShaderCompilationFailed, UnableToCreateProgram, ShaderLinkingFailed, UnableToCreateWindow };

pub const AppState = struct {
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
    meshes: std.StringHashMap(Shape.Mesh),
    shaderProgram: u32,
    width: f32,
    height: f32,
    appState: AppState,
    camera: Camera,

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
            .meshes = std.StringHashMap(Shape.Mesh).init(allocator),
            .shaderProgram = shaderProgram,
            .width = @floatFromInt(width),
            .height = @floatFromInt(height),
            .appState = AppState{},
            .camera = Camera.init(null, null),
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.meshes.iterator();
        while (it.next()) |entry| {
            const mesh = entry.value_ptr;
            c.glDeleteVertexArrays(1, &mesh.meta.VAO);
            c.glDeleteBuffers(1, &mesh.meta.VBO);
            if (mesh.indices) |indices| {
                _ = indices;
                c.glDeleteBuffers(1, &mesh.meta.IBO);
            }
        }

        self.meshes.deinit();

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

        _ = c.glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
        _ = c.glfwSetCursorPosCallback(window, mouseCallback);
        _ = c.glfwSetKeyCallback(window, keyCallback);
        _ = c.glfwSetScrollCallback(window, scrollCallback);

        const current_mode = c.glfwGetInputMode(window, c.GLFW_CURSOR);
        if (current_mode != c.GLFW_CURSOR_DISABLED) {
            c.glfwSetInputMode(window, c.GLFW_CURSOR, c.GLFW_CURSOR_DISABLED);

            const new_mode = c.glfwGetInputMode(window, c.GLFW_CURSOR);
            if (new_mode != c.GLFW_CURSOR_DISABLED) {
                std.debug.print("Failed to disable cursor in setupCallbacks\n", .{});
            }
        }
    }

    pub fn updateProjection(self: *Self) [16]f32 {
        return Transformations.perspective(self.appState.zoom, self.width / self.height, 0.1, 100.0);
    }

    pub fn getMeshNames(self: Self) void {
        var it = self.meshes.iterator();

        while (it.next()) |entry| {
            std.debug.print("{d} => {s}\n", .{ entry.value_ptr.*.meta.VBO, entry.key_ptr.* });
        }
    }

    pub fn addMesh(self: *Self, name: []const u8, mesh: Shape.Mesh) !void {
        try self.meshes.put(name, mesh);
    }

    pub fn render(self: *Self, window: ?*c.struct_GLFWwindow) void {
        c.glClearColor(0.15, 0.15, 0.15, 1.0);
        c.glClear(c.GL_COLOR_BUFFER_BIT | c.GL_DEPTH_BUFFER_BIT);

        c.glUseProgram(self.shaderProgram);

        var view = self.camera.get_view_matrix();

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

        // Iterate through all meshes in the hash map
        var it = self.meshes.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.draw) |drawFunction| {
                drawFunction(entry.value_ptr.*);
            } else {
                const mesh = entry.value_ptr.*;

                // Set mesh-specific uniforms
                const modelLoc = c.glGetUniformLocation(self.shaderProgram, "uModel");

                if (modelLoc != -1) {
                    c.glUniformMatrix4fv(modelLoc, 1, c.GL_FALSE, &mesh.modelMatrix);
                }

                c.glBindVertexArray(mesh.meta.VAO);
                c.glBindBuffer(c.GL_ARRAY_BUFFER, mesh.meta.VBO);

                // Position attribute (location = 0)
                c.glVertexAttribPointer(0, // location
                    3, // (vec3)
                    c.GL_FLOAT, c.GL_FALSE, @sizeOf(Shape.Vertex), // stride (size of entire vertex struct)
                    null // offset for position
                );
                c.glEnableVertexAttribArray(0);

                // Color attribute (location = 1)
                const color_offset = @offsetOf(Shape.Vertex, "color");
                c.glVertexAttribPointer(1, // location
                    3, // (vec3)
                    c.GL_FLOAT, c.GL_FALSE, @sizeOf(Shape.Vertex), // stride (size of entire vertex struct)
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
        }

        c.glfwSwapBuffers(window);
        c.glfwPollEvents();
    }

    pub fn processInput(self: *Self, debug: bool) void {
        const sprinting = self.appState.keys[@as(usize, c.GLFW_KEY_LEFT_SHIFT)];

        // Forward
        if (self.appState.keys[@as(usize, c.GLFW_KEY_W)]) {
            self.camera.move_forward(self.appState.delta_time, sprinting, debug);
        }

        // Left
        if (self.appState.keys[@as(usize, c.GLFW_KEY_A)]) {
            self.camera.move_left(self.appState.delta_time, sprinting, debug);
        }

        // Backward
        if (self.appState.keys[@as(usize, c.GLFW_KEY_S)]) {
            self.camera.move_backward(self.appState.delta_time, sprinting, debug);
        }

        // Right
        if (self.appState.keys[@as(usize, c.GLFW_KEY_D)]) {
            self.camera.move_right(self.appState.delta_time, sprinting, debug);
        }

        // Zoom controls can remain in the keyCallback if they are discrete actions
    }
};

inline fn readShaderSource(comptime path: []const u8) ![]const u8 {
    const file: []const u8 = @embedFile(path);
    std.debug.print("Source Length: {}\n", .{file.len});
    return file;
}

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

fn getCurrentMonitor(window: ?*c.struct_GLFWwindow) ?*c.GLFWmonitor {
    if (window == null) return null;

    var monitor_count: i32 = undefined;
    const monitors = c.glfwGetMonitors(&monitor_count);
    if (monitors == null or monitor_count == 0) return null;

    // Get window position
    var win_x: i32 = undefined;
    var win_y: i32 = undefined;
    c.glfwGetWindowPos(window, &win_x, &win_y);

    // Get window size
    var win_width: i32 = undefined;
    var win_height: i32 = undefined;
    c.glfwGetWindowSize(window, &win_width, &win_height);

    // Find the monitor that contains the window center
    const center_x = win_x + @divTrunc(win_width, 2);
    const center_y = win_y + @divTrunc(win_height, 2);

    var i: usize = 0;
    while (i < @as(i32, @intCast(monitor_count))) : (i += 1) {
        const mon = monitors[i];
        var mx: i32 = undefined;
        var my: i32 = undefined;
        var mw: i32 = undefined;
        var mh: i32 = undefined;
        c.glfwGetMonitorWorkarea(mon, &mx, &my, &mw, &mh);

        if (center_x >= mx and center_x < mx + mw and
            center_y >= my and center_y < my + mh)
        {
            return mon;
        }
    }

    // Default to primary monitor if no match
    return c.glfwGetPrimaryMonitor();
}

// Window creation with proper monitor handling
pub fn createWindow() ?*c.GLFWwindow {
    c.glfwWindowHint(c.GLFW_FOCUSED, c.GLFW_TRUE);
    c.glfwWindowHint(c.GLFW_FOCUS_ON_SHOW, c.GLFW_TRUE);
    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_OPENGL_API);
    c.glfwWindowHint(c.GLFW_CONTEXT_CREATION_API, c.GLFW_NATIVE_CONTEXT_API);

    const width: i32 = @intFromFloat(1920 * 0.75);
    const height: i32 = @intFromFloat(1080 * 0.75);

    const window = c.glfwCreateWindow(width, height, "Drone Studio", null, null) orelse return null;

    // Make context current immediately after window creation
    c.glfwMakeContextCurrent(window);

    // Set up cursor mode BEFORE changing monitor settings
    c.glfwSetInputMode(window, c.GLFW_CURSOR, c.GLFW_CURSOR_DISABLED);
    if (c.glfwRawMouseMotionSupported() == c.GLFW_TRUE) {
        c.glfwSetInputMode(window, c.GLFW_RAW_MOUSE_MOTION, c.GLFW_TRUE);
    }

    const monitor = getCurrentMonitor(window);
    if (monitor != null) {
        // Get monitor position and video mode
        var x: i32 = undefined;
        var y: i32 = undefined;
        c.glfwGetMonitorPos(monitor, &x, &y);

        const video_mode = c.glfwGetVideoMode(monitor);
        if (video_mode != null) {
            // Correctly dereference the video mode pointer
            const mode_width = video_mode.*.width;
            const mode_height = video_mode.*.height;
            const mode_refresh = video_mode.*.refreshRate;

            c.glfwSetWindowMonitor(window, monitor, 0, 0, mode_width, mode_height, mode_refresh);
        }
    }

    // Force focus and raise window
    c.glfwFocusWindow(window);
    c.glfwShowWindow(window);

    return window;
}

fn framebufferSizeCallback(window: ?*c.GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
    if (window == null) return;

    const user_ptr = c.glfwGetWindowUserPointer(window);
    if (user_ptr == null) {
        std.debug.print("Error: Window user pointer is null in framebufferSizeCallback\n", .{});
        return;
    }

    const scene = @as(*Scene, @ptrCast(@alignCast(c.glfwGetWindowUserPointer(window))));

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

    const aspectRatio: f32 = scene.width / scene.height;

    scene.camera.process_mouse_movement(xoffset, yoffset, aspectRatio, false);
}

fn keyCallback(window: ?*c.struct_GLFWwindow, key: c_int, scancode: c_int, action: c_int, mods: c_int) callconv(.C) void {
    if (window == null) return;

    _ = scancode;
    _ = mods;

    const scene = @as(*Scene, @ptrCast(@alignCast(c.glfwGetWindowUserPointer(window))));

    if (key < 0 or key >= 1024) return;

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

    const zoomSensitivity: f32 = 0.1;
    const newZoom = scene.appState.zoom - @as(f32, @floatCast(yoffset)) * zoomSensitivity * scene.appState.zoom;

    // Clamp the zoom level to prevent weird behavior at larger FOV's
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
