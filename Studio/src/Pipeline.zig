// src/Scene.zig
const std = @import("std");
const Shape = @import("Shape.zig");
const Math = @import("Math.zig");
const Node = @import("Node.zig");
const Vec3 = Math.Vec3;
const File = std.fs.File;
const Camera = @import("Camera.zig");
const gl = @import("bindings/gl.zig");
const glfw = gl.glfw;
const glad = gl.glad;
const globals = @import("Globals.zig");

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
    nodes: std.StringHashMap(*Node),
    shaderProgram: u32,
    width: f32,
    height: f32,
    appState: AppState,
    camera: Camera,
    uModelLoc: glad.GLint,
    uViewLoc: glad.GLint,
    uProjectionLoc: glad.GLint,
    useTextureLoc: glad.GLint,
    yTextureLoc: glad.GLint,
    uvTextureLoc: glad.GLint,
    useInstancingLoc: glad.GLint,
    texGen: TextureGenerator = TextureGenerator{},

    last_projection: [16]f32 = undefined,
    projection_dirty: bool = true,

    frame_count: u64 = 0,
    last_fps_time: f64 = 0,
    frame_times: [120]f64 = .{0} ** 120,
    frame_time_index: usize = 0,

    pub fn init(allocator: std.mem.Allocator, window: ?*glfw.struct_GLFWwindow) !*Self {
        if (window == null) {
            std.debug.print("Failed to create GLFW window\n", .{});
            return GSLWError.FailedToCreateWindow;
        }

        var width: i32 = undefined;
        var height: i32 = undefined;
        glfw.glfwGetWindowSize(window.?, &width, &height);

        // Make the window's context current
        glfw.glfwMakeContextCurrent(window);
        glfw.glfwSwapInterval(0);

        // Initialize OpenGL Loader
        std.debug.print("Loading Glad...\n", .{});
        if (glad.gladLoadGLLoader(@ptrCast(&glfw.glfwGetProcAddress)) == 0) {
            std.debug.print("Failed to initialize GLAD\n", .{});
            return GSLWError.FailedToInitialize;
        }

        std.debug.print("Initializing viewport...\n\n", .{});
        glad.glViewport(0, 0, width, height);
        glad.glEnable(glad.GL_DEPTH_TEST);

        const shaderProgram = try createShaderProgram("shaders/vertex_shader.glsl", "shaders/fragment_shader.glsl");
        glad.glUseProgram(shaderProgram);

        var currentProgram: u32 = 0;
        glad.glGetIntegerv(glad.GL_CURRENT_PROGRAM, @ptrCast(&currentProgram));
        if (currentProgram != shaderProgram) {
            std.debug.print("Shader program not active!\n", .{});
        }

        checkOpenGLError("Uniform Setup");

        var numActiveAttributes: c_int = 0;
        glad.glGetProgramiv(shaderProgram, glad.GL_ACTIVE_ATTRIBUTES, &numActiveAttributes);

        std.debug.print("\nGetting Attributes\n", .{});
        std.debug.print("Number of active vertex attributes: {}\n", .{numActiveAttributes});

        for (0..@intCast(numActiveAttributes)) |i| {
            var nameBuffer: [256]u8 = undefined;
            var length: c_int = 0;
            var size: c_int = 0;
            var var_type: c_uint = 0;

            glad.glGetActiveAttrib(
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

        glad.glDepthFunc(glad.GL_LESS);

        // Cache uniform locations
        const uModelLoc = glad.glGetUniformLocation(shaderProgram, "uModel");
        const uViewLoc = glad.glGetUniformLocation(shaderProgram, "uView");
        const uProjectionLoc = glad.glGetUniformLocation(shaderProgram, "uProjection");
        const useTextureLoc = glad.glGetUniformLocation(shaderProgram, "useTexture");
        const yTextureLoc = glad.glGetUniformLocation(shaderProgram, "yTexture");
        const uvTextureLoc = glad.glGetUniformLocation(shaderProgram, "uvTexture");
        const useInstancingLoc = glad.glGetUniformLocation(shaderProgram, "uUseInstancing");

        if (uModelLoc == -1 or uViewLoc == -1 or uProjectionLoc == -1) {
            std.debug.print("Failed to get one or more uniform locations\n", .{});
            // Handle error appropriately
        }

        const scene = try allocator.create(Scene);

        scene.* = Self{
            .allocator = allocator,
            .nodes = std.StringHashMap(*Node).init(allocator),
            .shaderProgram = shaderProgram,
            .width = @floatFromInt(width),
            .height = @floatFromInt(height),
            .appState = AppState{},
            .camera = Camera.init(null, null),
            .uModelLoc = uModelLoc,
            .uViewLoc = uViewLoc,
            .uProjectionLoc = uProjectionLoc,
            .useTextureLoc = useTextureLoc,
            .yTextureLoc = yTextureLoc,
            .uvTextureLoc = uvTextureLoc,
            .useInstancingLoc = useInstancingLoc,
        };

        return scene;
    }

    pub fn deinit(self: *Self) void {
        var it = self.nodes.iterator();
        while (it.next()) |entry| {
            const curr_mesh = entry.value_ptr.*.mesh;
            if (curr_mesh) |mesh| {
                glad.glDeleteVertexArrays(1, mesh.meta.VAO);
                glad.glDeleteBuffers(1, mesh.meta.VBO);

                if (mesh.indices) |indices| {
                    _ = indices;
                    glad.glDeleteBuffers(1, mesh.meta.IBO);
                }
            }
        }

        self.nodes.deinit();
        glad.glDeleteProgram(self.shaderProgram);
    }

    pub fn setupCallbacks(self: *Self, window: ?*glfw.struct_GLFWwindow) void {
        if (window == null) return;

        glfw.glfwSetWindowUserPointer(window, @ptrCast(self));

        const ptr = glfw.glfwGetWindowUserPointer(window);
        if (ptr == null) {
            std.debug.print("Failed to set window user pointer\n", .{});
            return;
        }

        _ = glfw.glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
        _ = glfw.glfwSetCursorPosCallback(window, mouseCallback);
        _ = glfw.glfwSetKeyCallback(window, keyCallback);
        _ = glfw.glfwSetScrollCallback(window, scrollCallback);

        const current_mode = glfw.glfwGetInputMode(window, glfw.GLFW_CURSOR);
        if (current_mode != glfw.GLFW_CURSOR_DISABLED) {
            glfw.glfwSetInputMode(window, glfw.GLFW_CURSOR, glfw.GLFW_CURSOR_DISABLED);

            const new_mode = glfw.glfwGetInputMode(window, glfw.GLFW_CURSOR);
            if (new_mode != glfw.GLFW_CURSOR_DISABLED) {
                std.debug.print("Failed to disable cursor in setupCallbacks\n", .{});
            }
        }
    }

    pub fn updateProjection(self: *Self) [16]f32 {
        return Math.perspective(self.appState.zoom, self.width / self.height, 0.1, 100.0);
    }

    pub fn getSceneGraph(self: Self) void {
        var it = self.nodes.iterator();

        std.debug.print("\nGetting Nodes...\n", .{});

        while (it.next()) |entry| {
            const curr_node = entry.value_ptr.*;

            std.debug.print("\n=================================\nNode: {s}", .{entry.key_ptr.*});
            curr_node.debug();
        }
    }

    pub fn addNode(self: *Self, name: []const u8, node: *Node) !void {
        node.addSceneRecursively(self);
        try self.nodes.put(name, node);
    }

    pub fn render(self: *Self, window: ?*glfw.struct_GLFWwindow) void {
        // Start frame timing
        const frame_start = glfw.glfwGetTime();

        glad.glClearColor(0.15, 0.15, 0.15, 1.0);
        glad.glClear(glad.GL_COLOR_BUFFER_BIT | glad.GL_DEPTH_BUFFER_BIT);

        var view = self.camera.get_view_matrix();
        if (self.uViewLoc != -1) {
            glad.glUniformMatrix4fv(self.uViewLoc, 1, glad.GL_FALSE, &view);
        }

        if (self.projection_dirty) {
            const currentProjection = self.updateProjection();
            if (self.uProjectionLoc != -1) {
                glad.glUniformMatrix4fv(self.uProjectionLoc, 1, glad.GL_FALSE, &currentProjection);
            }
            self.projection_dirty = false;
        }

        // Batch similar draw calls if possible
        var it = self.nodes.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.update();
        }

        // Ensure all GL commands are submitted before swap
        glad.glFlush();

        glfw.glfwSwapBuffers(window);
        glfw.glfwPollEvents();

        // Frame timing and FPS calculation
        const frame_end = glfw.glfwGetTime();
        const frame_time = frame_end - frame_start;

        // Store frame time in circular buffer
        self.frame_times[self.frame_time_index] = frame_time * 1000.0; // Convert to ms
        self.frame_time_index = (self.frame_time_index + 1) % 120;

        // Calculate and print average frame time every second
        self.frame_count += 1;
        if (frame_end - self.last_fps_time >= 1.0) {
            var sum: f64 = 0;
            for (self.frame_times) |time| {
                sum += time;
            }
            const avg_frame_time = sum / @as(f64, @floatFromInt(self.frame_times.len));

            std.debug.print("Avg Frame Time: {d:.2}ms, FPS: {d:.1}\n", .{ avg_frame_time, 1000.0 / avg_frame_time });

            self.frame_count = 0;
            self.last_fps_time = frame_end;
        }
    }

    pub fn processInput(self: *Self, debug: bool) void {
        _ = debug;

        const sprinting = self.appState.keys[@as(usize, glfw.GLFW_KEY_LEFT_SHIFT)];
        const velocity = self.camera.speed * self.appState.delta_time *
            (if (sprinting) @as(f32, 2.0) else @as(f32, 1.0));

        // Pre-calculate movement vectors once per frame if needed
        var movement = Vec3{ .x = 0, .y = 0, .z = 0 };

        if (self.appState.keys[@as(usize, glfw.GLFW_KEY_W)]) {
            movement = Vec3.add(movement, Vec3.scale(self.camera.front, velocity));
        }
        if (self.appState.keys[@as(usize, glfw.GLFW_KEY_S)]) {
            movement = Vec3.sub(movement, Vec3.scale(self.camera.front, velocity));
        }
        if (self.appState.keys[@as(usize, glfw.GLFW_KEY_D)]) {
            movement = Vec3.add(movement, Vec3.scale(self.camera.right, velocity));
        }
        if (self.appState.keys[@as(usize, glfw.GLFW_KEY_A)]) {
            movement = Vec3.sub(movement, Vec3.scale(self.camera.right, velocity));
        }

        self.camera.position = Vec3.add(self.camera.position, movement);

        // Zoom controls can remain in the keyCallback if they are discrete actions
    }
};

inline fn readShaderSource(comptime path: []const u8) ![]const u8 {
    const file: []const u8 = @embedFile(path);
    std.debug.print("Source Length: {}\n", .{file.len});
    return file;
}

fn compileShader(shaderType: u32, source: []const u8) !u32 {
    const shader = glad.glCreateShader(shaderType);
    if (shader == 0) {
        std.debug.print("Failed to compile shader\n", .{});
        return ShaderError.UnableToCreateShader;
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
        return ShaderError.ShaderCompilationFailed;
    }

    return shader;
}

pub fn createShaderProgram(comptime vertexPath: []const u8, comptime fragmentPath: []const u8) !u32 {
    std.debug.print("Initializing Vertex Shader...\n", .{});
    std.debug.print("Reading Vertex Shader from Source...\n", .{});
    const vertexSource = try readShaderSource(vertexPath);
    const vertexShader = compileShader(glad.GL_VERTEX_SHADER, vertexSource) catch |err| {
        std.debug.print("Failed to read vertex shader '{s}': {any}\n", .{ vertexPath, err });
        return ShaderError.ShaderCompilationFailed;
    };
    std.debug.print("\n", .{});

    std.debug.print("Initializing Fragment Shader...\n", .{});
    std.debug.print("Reading Fragment Shader from Source...\n", .{});
    const fragmentSource = try readShaderSource(fragmentPath);
    const fragmentShader = compileShader(glad.GL_FRAGMENT_SHADER, fragmentSource) catch |err| {
        std.debug.print("Failed to read fragment shader '{s}': {any}\n", .{ fragmentPath, err });
        return ShaderError.ShaderCompilationFailed;
    };
    std.debug.print("\n", .{});

    std.debug.print("Creating ShaderProgram...\n", .{});
    const shaderProgram = glad.glCreateProgram();
    if (shaderProgram == 0) {
        std.debug.print("Failed to create Shader Program\n", .{});
        return ShaderError.UnableToCreateProgram;
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
        return ShaderError.ShaderLinkingFailed;
    }

    std.debug.print("Running cleanup on Shaders...\n", .{});
    // Shaders can be deleted after linking
    glad.glDeleteShader(vertexShader);
    glad.glDeleteShader(fragmentShader);

    return shaderProgram;
}

fn checkOpenGLError(caller: []const u8) void {
    var err: u32 = glad.glGetError();
    while (err != glad.GL_NO_ERROR) {
        std.debug.print("OpenGL Error: {x} from Caller: {s}\n", .{ err, caller });
        err = glad.glGetError();
    }
}

fn getCurrentMonitor(window: ?*glfw.struct_GLFWwindow) ?*glfw.GLFWmonitor {
    if (window == null) return null;

    var monitor_count: i32 = undefined;
    const monitors = glfw.glfwGetMonitors(&monitor_count);
    if (monitors == null or monitor_count == 0) return null;

    // Get window position
    var win_x: i32 = undefined;
    var win_y: i32 = undefined;
    glfw.glfwGetWindowPos(window, &win_x, &win_y);

    // Get window size
    var win_width: i32 = undefined;
    var win_height: i32 = undefined;
    glfw.glfwGetWindowSize(window, &win_width, &win_height);

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
        glfw.glfwGetMonitorWorkarea(mon, &mx, &my, &mw, &mh);

        if (center_x >= mx and center_x < mx + mw and
            center_y >= my and center_y < my + mh)
        {
            return mon;
        }
    }

    // Default to primary monitor if no match
    return glfw.glfwGetPrimaryMonitor();
}

pub fn createWindow() ?*glfw.GLFWwindow {
    glfw.glfwWindowHint(glfw.GLFW_FOCUSED, glfw.GLFW_TRUE);
    glfw.glfwWindowHint(glfw.GLFW_FOCUS_ON_SHOW, glfw.GLFW_TRUE);
    glfw.glfwWindowHint(glfw.GLFW_CLIENT_API, glfw.GLFW_OPENGL_API);
    glfw.glfwWindowHint(glfw.GLFW_CONTEXT_CREATION_API, glfw.GLFW_NATIVE_CONTEXT_API);

    glfw.glfwWindowHint(glfw.GLFW_DOUBLEBUFFER, glfw.GLFW_TRUE);
    glfw.glfwWindowHint(glfw.GLFW_SRGB_CAPABLE, glfw.GLFW_TRUE);

    const width: i32 = @intFromFloat(1920 * 0.75);
    const height: i32 = @intFromFloat(1080 * 0.75);

    const window = glfw.glfwCreateWindow(width, height, "Drone Studio", null, null) orelse return null;

    // Make context current immediately after window creation
    glfw.glfwMakeContextCurrent(window);
    glfw.glfwSwapInterval(1);

    // Set up cursor mode BEFORE changing monitor settings
    glfw.glfwSetInputMode(window, glfw.GLFW_CURSOR, glfw.GLFW_CURSOR_DISABLED);
    if (glfw.glfwRawMouseMotionSupported() == glfw.GLFW_TRUE) {
        glfw.glfwSetInputMode(window, glfw.GLFW_RAW_MOUSE_MOTION, glfw.GLFW_TRUE);
    }

    const monitor = getCurrentMonitor(window);
    if (monitor != null) {
        // Get monitor position and video mode
        var x: i32 = undefined;
        var y: i32 = undefined;
        glfw.glfwGetMonitorPos(monitor, &x, &y);

        const video_mode = glfw.glfwGetVideoMode(monitor);
        if (video_mode != null) {
            // Correctly dereference the video mode pointer
            const mode_width = video_mode.*.width;
            const mode_height = video_mode.*.height;
            const mode_refresh = video_mode.*.refreshRate;

            glfw.glfwSetWindowMonitor(window, monitor, 0, 0, mode_width, mode_height, mode_refresh);
        }
    }

    // Force focus and raise window
    glfw.glfwFocusWindow(window);
    glfw.glfwShowWindow(window);

    return window;
}

fn framebufferSizeCallback(window: ?*glfw.GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
    if (window == null) return;

    const user_ptr = glfw.glfwGetWindowUserPointer(window);
    if (user_ptr == null) {
        std.debug.print("Error: Window user pointer is null in framebufferSizeCallback\n", .{});
        return;
    }

    const scene = @as(*Scene, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

    glad.glViewport(0, 0, width, height);

    // Update Scene's width and height
    scene.width = @floatFromInt(width);
    scene.height = @floatFromInt(height);
}

fn mouseCallback(window: ?*glfw.struct_GLFWwindow, xpos: f64, ypos: f64) callconv(.C) void {
    if (window == null) return;

    // Only process mouse input if window is focused
    if (glfw.glfwGetWindowAttrib(window, glfw.GLFW_FOCUSED) != glfw.GLFW_TRUE) {
        return;
    }

    const scene = @as(*Scene, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

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

fn keyCallback(window: ?*glfw.struct_GLFWwindow, key: c_int, scancode: c_int, action: c_int, mods: c_int) callconv(.C) void {
    if (window == null) return;

    _ = scancode;
    _ = mods;

    const scene = @as(*Scene, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

    if (key < 0 or key >= 1024) return;

    if (action == glfw.GLFW_PRESS) {
        scene.appState.keys[@intCast(key)] = true;
    } else if (action == glfw.GLFW_RELEASE) {
        scene.appState.keys[@intCast(key)] = false;
    }

    if (action == glfw.GLFW_PRESS or action == glfw.GLFW_REPEAT) {
        switch (key) {
            glfw.GLFW_KEY_ESCAPE => {
                glfw.glfwIconifyWindow(window);
            },
            glfw.GLFW_KEY_RIGHT_BRACKET => {
                glfw.glfwDestroyWindow(window);
            },
            glfw.GLFW_KEY_UP => {
                // Zoom in (narrower FOV)
                scene.appState.zoom -= 1.0;
                if (scene.appState.zoom < 1.0) scene.appState.zoom = 1.0;
            },
            glfw.GLFW_KEY_DOWN => {
                // Zoom out (wider FOV)
                scene.appState.zoom += 1.0;
            },
            glfw.GLFW_KEY_P => {
                globals.PAUSED = !globals.PAUSED;
            },

            else => {},
        }
    }

    // Handle additional keys here
}

fn scrollCallback(window: ?*glfw.struct_GLFWwindow, xoffset: f64, yoffset: f64) callconv(.C) void {
    if (window == null) return;

    _ = xoffset;

    const scene = @as(*Scene, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

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

    scene.projection_dirty = true;

    std.debug.print("yOffset: {d}\n", .{yoffset});
    std.debug.print("Zoom Level: {d}\n", .{newZoom});
}

pub const TextureGenerator = struct {
    const Self = @This();
    count: c_int = 0,

    pub fn generateID(self: *Self) c_int {
        defer self.count += 1;
        return self.count;
    }
};
