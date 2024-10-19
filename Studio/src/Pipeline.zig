// src/Scene.zig
const std = @import("std");
const Shape = @import("Shape.zig");
const File = std.fs.File;
const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("glad/glad.h"); // Ensure GLAD is included
    @cInclude("GLFW/glfw3.h");
});

const GSLWError = error{ FailedToCreateWindow, FailedToInitialize };
const ShaderError = error{ UnableToCreateShader, ShaderCompilationFailed, UnableToCreateProgram, ShaderLinkingFailed, UnableToCreateWindow };

// Function to read shader source from a file
fn readShaderSource(path: []const u8) ![]const u8 {
    const allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile(path, .{ .mode = File.OpenMode.read_only });
    defer file.close();

    const file_size = try file.getEndPos();
    const buffer = try allocator.alloc(u8, file_size);
    const status = try file.readAll(buffer);
    std.debug.print("Source Length: {}\n", .{status});
    return buffer;
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
pub fn createShaderProgram(vertexPath: []const u8, fragmentPath: []const u8) !u32 {
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

pub const Scene = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    objects: std.StringHashMap(Shape.Object),
    shaderProgram: u32,

    pub fn init(allocator: std.mem.Allocator, window: ?*c.struct_GLFWwindow) !Self {
        if (window == null) {
            std.debug.print("Failed to create GLFW window\n", .{});
            return GSLWError.FailedToCreateWindow;
        }

        // Make the window's context current
        c.glfwMakeContextCurrent(window);
        c.glfwSwapInterval(0);

        // Initialize OpenGL loader (optional, depending on usage)
        // You might need to load OpenGL function pointers here if using modern OpenGL
        std.debug.print("Loading Glad...\n", .{});
        if (c.gladLoadGLLoader(@ptrCast(&c.glfwGetProcAddress)) == 0) {
            std.debug.print("Failed to initialize GLAD\n", .{});
            return GSLWError.FailedToInitialize;
        }

        std.debug.print("Initializing viewport...\n\n", .{});
        // Viewport setup
        c.glViewport(0, 0, 1920 * 0.75, 1080 * 0.75);
        // Enable depth testing if needed
        c.glEnable(c.GL_DEPTH_TEST);

        const shaderProgram = try createShaderProgram("src/shaders/vertex_shader.glsl", "src/shaders/fragment_shader.glsl");
        defer c.glDeleteProgram(shaderProgram);

        // Set uniform color for the grid (e.g., white)

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
        };
    }

    pub fn deinit(self: Self) void {
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
    }

    pub fn addObject(self: *Self, name: []const u8, object: Shape.Object) !void {
        try self.objects.put(name, object);
    }

    pub fn render(self: *Self, window: ?*c.struct_GLFWwindow, view: [16]f32, projection: [16]f32) void {
        c.glClearColor(0.15, 0.15, 0.15, 1.0);
        c.glClear(c.GL_COLOR_BUFFER_BIT | c.GL_DEPTH_BUFFER_BIT);
        // Use the shader program
        c.glUseProgram(self.shaderProgram);

        // Set common uniforms
        const viewLoc = c.glGetUniformLocation(self.shaderProgram, "uView");
        const projectionLoc = c.glGetUniformLocation(self.shaderProgram, "uProjection");
        if (viewLoc != -1) {
            c.glUniformMatrix4fv(viewLoc, 1, c.GL_FALSE, &view);
        }
        if (projectionLoc != -1) {
            c.glUniformMatrix4fv(projectionLoc, 1, c.GL_FALSE, &projection);
        }

        // Iterate through all objects in the hash map
        var it = self.objects.iterator();
        while (it.next()) |entry| {
            // const key = entry.key_ptr.*;
            const object = entry.value_ptr.*;

            // std.debug.print("Rendering: {s} => VAO: {d}\n", .{ key, object.meta.VAO });

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
                c.glDrawElements(c.GL_TRIANGLES, @intCast(object.indices.len), c.GL_UNSIGNED_INT, null);
            } else {
                c.glDrawArrays(c.GL_TRIANGLES, 0, @intCast(object.vertices.len / 3));
            }

            // Unbind the VAO
            c.glBindVertexArray(0);

            // Swap front and back buffers
            c.glfwSwapBuffers(window);
            c.glfwPollEvents();
        }
    }
};
