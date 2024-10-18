const std = @import("std");
const File = std.fs.File;
const Transformations = @import("Transformations.zig");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("glad/glad.h"); // Include GLAD
    @cInclude("GLFW/glfw3.h");
    // Add OpenGL function declarations if needed
});

const ShaderError = error{ UnableToCreateShader, ShaderCompilationFailed, UnableToCreateProgram, ShaderLinkingFailed };

inline fn flatten(comptime N: usize, arr: [N][3]f32) [N * 3]f32 {
    var flat: [N * 3]f32 = undefined;
    var idx: usize = 0;

    for (arr) |vertex| {
        flat[idx] = vertex[0];
        flat[idx + 1] = vertex[1];
        flat[idx + 2] = vertex[2];
        idx += 3;
    }

    return flat;
}

// Function to generate grid vertices
inline fn generateGridVertices(comptime gridSize: i32, comptime spacing: f32) [gridSize * 4][3]f32 {
    // Calculate total number of lines (X and Z axes)
    const totalLines = gridSize * 2; // 2 axes

    // Define a fixed-size array to hold all vertices
    // Each line has 2 vertices, each with 3 components (x, y, z)
    var vertices: [totalLines * 2][3]f32 = undefined;

    var index: usize = 0;

    // Generate lines parallel to X-axis (varying Z)
    for (0..gridSize) |i| {
        // Start vertex
        vertices[index] = [_]f32{
            -1.0 * @as(f32, @floatFromInt(i)) * spacing, // x
            0.0, // y
            @as(f32, @floatFromInt(gridSize)) * spacing, // z
        };
        index += 1;

        // End vertex
        vertices[index] = [_]f32{
            @as(f32, @floatFromInt(i)) * spacing, // x
            0.0, // y
            @as(f32, @floatFromInt(gridSize)) * spacing, // z
        };
        index += 1;
    }

    // Generate lines parallel to Z-axis (varying X)
    for (0..gridSize) |i| {
        // Start vertex
        vertices[index] = [_]f32{
            @as(f32, @floatFromInt(i)) * spacing, // x
            0.0, // y
            -1.0 * @as(f32, @floatFromInt(gridSize)) * spacing, // z
        };
        index += 1;

        // End vertex
        vertices[index] = [_]f32{
            @as(f32, @floatFromInt(i)) * spacing, // x
            0.0, // y
            @as(f32, @floatFromInt(gridSize)) * spacing, // z
        };
        index += 1;
    }

    return vertices;
}

const Grid = struct { VAO: u32, VBO: u32 };

// Function to create VAO and VBO for the grid
fn setupGrid(vertices: []const f32, length: u32) !Grid {
    var vao: u32 = 0;
    var vbo: u32 = 0;

    c.glGenVertexArrays(1, &vao);
    c.glGenBuffers(1, &vbo);

    c.glBindVertexArray(vao);

    c.glBindBuffer(c.GL_ARRAY_BUFFER, vbo);
    c.glBufferData(c.GL_ARRAY_BUFFER, @as(c_long, length * @sizeOf(f32)), vertices.ptr, c.GL_STATIC_DRAW);

    c.glVertexAttribPointer(0, 3, c.GL_FLOAT, 0, 3 * @sizeOf(f32), @ptrFromInt(0));
    c.glEnableVertexAttribArray(0);

    c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
    c.glBindVertexArray(0);

    return Grid{ .VAO = vao, .VBO = vbo };
}

// Function to read shader source from a file
fn readShaderSource(path: []const u8) ![]const u8 {
    const allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile(path, .{ .mode = File.OpenMode.read_only });
    defer file.close();

    const file_size = try file.getEndPos();
    const buffer = try allocator.alloc(u8, file_size);
    const status = try file.readAll(buffer);
    std.debug.print("Status: {}\n", .{status});
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
fn createShaderProgram(vertexPath: []const u8, fragmentPath: []const u8) !u32 {
    const vertexSource = try readShaderSource(vertexPath);
    const fragmentSource = try readShaderSource(fragmentPath);

    const vertexShader = compileShader(c.GL_VERTEX_SHADER, vertexSource) catch |err| {
        std.debug.print("Failed to read vertex shader '{s}': {any}\n", .{ vertexPath, err });
        return ShaderError.ShaderCompilationFailed;
    };

    const fragmentShader = compileShader(c.GL_FRAGMENT_SHADER, fragmentSource) catch |err| {
        std.debug.print("Failed to read fragment shader '{s}': {any}\n", .{ fragmentPath, err });
        return ShaderError.ShaderCompilationFailed;
    };

    const shaderProgram = c.glCreateProgram();
    if (shaderProgram == 0) {
        std.debug.print("Failed to create Shader Program\n", .{});
        return ShaderError.UnableToCreateProgram;
    }

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

    // Shaders can be deleted after linking
    c.glDeleteShader(vertexShader);
    c.glDeleteShader(fragmentShader);

    return shaderProgram;
}

// Function to check and print OpenGL errors
fn checkOpenGLError(caller: []const u8) void {
    var err: u32 = c.glGetError();
    while (err != c.GL_NO_ERROR) {
        std.debug.print("OpenGL Error: {x} from Caller: {s}\n", .{ err, caller });
        err = c.glGetError();
    }
}

fn setupTriangle(vertices: []const f32, length: u32) !Grid {
    var vao: u32 = 0;
    var vbo: u32 = 0;

    std.debug.print("Length: {d}\n", .{length});
    std.debug.print("Object address: {d}\n", .{&vertices});
    std.debug.print("First object address: {d}\n", .{&vertices[0]});

    c.glGenVertexArrays(1, &vao);
    c.glGenBuffers(1, &vbo);

    c.glBindVertexArray(vao);

    c.glBindBuffer(c.GL_ARRAY_BUFFER, vbo);
    c.glBufferData(c.GL_ARRAY_BUFFER, length * @sizeOf(f32), @ptrCast(vertices), c.GL_STATIC_DRAW);

    c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, 3 * @sizeOf(f32), null);
    c.glEnableVertexAttribArray(0);

    c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);

    c.glBindVertexArray(0);

    return Grid{ .VAO = vao, .VBO = vbo };
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    std.debug.print("Current Working Directory: {?s}\n", .{
        try std.fs.cwd().realpathAlloc(alloc, "."),
    });

    // Initialize GLFW
    if (c.glfwInit() == 0) {
        std.debug.print("Failed to initialize GLFW\n", .{});
        return;
    }
    defer c.glfwTerminate();

    // Set OpenGL version (4. Core)
    c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MAJOR, 3);
    c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MINOR, 3);
    c.glfwWindowHint(c.GLFW_OPENGL_DEBUG_CONTEXT, c.GLFW_TRUE);
    c.glfwWindowHint(c.GLFW_OPENGL_PROFILE, c.GLFW_OPENGL_CORE_PROFILE);

    // Create a windowed mode window and its OpenGL context
    const window = c.glfwCreateWindow(1920 * 0.75, 1080 * 0.75, "Drone Studio", null, null);
    if (window == null) {
        std.debug.print("Failed to create GLFW window\n", .{});
        return;
    }
    defer c.glfwDestroyWindow(window);

    // Make the window's context current
    c.glfwMakeContextCurrent(window);
    c.glfwSwapInterval(0);

    // Initialize OpenGL loader (optional, depending on usage)
    // You might need to load OpenGL function pointers here if using modern OpenGL
    if (c.gladLoadGLLoader(@ptrCast(&c.glfwGetProcAddress)) == 0) {
        std.debug.print("Failed to initialize GLAD\n", .{});
        return;
    }

    // Viewport setup
    c.glViewport(0, 0, 1920 * 0.75, 1080 * 0.75);

    // Enable depth testing if needed
    c.glEnable(c.GL_DEPTH_TEST);

    // Compile and link shaders
    const shaderProgram = try createShaderProgram("src/shaders/vertex_shader.glsl", "src/shaders/fragment_shader.glsl");
    defer c.glDeleteProgram(shaderProgram);

    // Retrieve uniform locations
    // const projectionLoc = c.glGetUniformLocation(shaderProgram, "uProjection");
    // const viewLoc = c.glGetUniformLocation(shaderProgram, "uView");
    // const modelLoc = c.glGetUniformLocation(shaderProgram, "uModel");
    // const colorLocation = c.glGetUniformLocation(shaderProgram, "uColor");

    // Generate grid vertices
    // Compile-Time Parameters
    // const gridSize: comptime_int = 10;
    // const spacing: comptime_float = 1.0;

    // Generate grid vertices at compile time
    // const gridVertices = comptime generateGridVertices(gridSize, spacing);
    // const flatVertices = comptime flatten(gridVertices.len, gridVertices);

    // // Set up grid VAO and VBO
    // const grid = try setupGrid(&flatVertices, flatVertices.len);

    // Create projection and view matrices
    // const projection = transformations.orthographic(-15.0, 15.0, -15.0, 15.0, -10.0, 10.0);
    // const eye = [3]f32{ 0.0, 5.0, 5.0 };
    // const center = [3]f32{ 0.0, 0.0, 0.0 };
    // const up = [3]f32{ 0.0, 1.0, 0.0 };
    // const view = transformations.lookAt(eye, center, up);

    // const model = [16]f32{
    //     1.0, 0.0, 0.0, 0.0,
    //     0.0, 1.0, 0.0, 0.0,
    //     0.0, 0.0, 1.0, 0.0,
    //     0.0, 0.0, 0.0, 1.0,
    // };

    // defer c.glDeleteVertexArrays(1, grid.VAO);
    // defer c.glDeleteBuffers(1, grid.VBO);

    // const identityMatrix: [16]f32 = .{
    //     1.0, 0.0, 0.0, 0.0,
    //     0.0, 1.0, 0.0, 0.0,
    //     0.0, 0.0, 1.0, 0.0,
    //     0.0, 0.0, 0.0, 1.0,
    // };
    // const projection = identityMatrix;
    // const view = identityMatrix;
    // const model = identityMatrix;

    const triangleVertices = [_]f32{
        0.0, 0.5, 0.0, // Top
        -0.5, -0.5, 0.0, // Bottom Left
        0.5, -0.5, 0.0, // Bottom Right
    };

    // Set up triangle VAO and VBO
    const triangle = try setupTriangle(&triangleVertices, triangleVertices.len);

    defer c.glDeleteVertexArrays(1, &triangle.VAO);
    defer c.glDeleteBuffers(1, &triangle.VBO);

    // Set uniform color for the grid (e.g., white)

    c.glUseProgram(shaderProgram);

    var currentProgram: u32 = 0;

    c.glGetIntegerv(c.GL_CURRENT_PROGRAM, @ptrCast(&currentProgram));
    if (currentProgram != shaderProgram) {
        std.debug.print("Shader program not active!\n", .{});
    }

    // if (projectionLoc != -1) {
    //     c.glUniformMatrix4fv(projectionLoc, 1, 0, &projection);
    // }
    // if (viewLoc != -1) {
    //     c.glUniformMatrix4fv(viewLoc, 1, 0, &view);
    // }
    // if (modelLoc != -1) {
    //     c.glUniformMatrix4fv(modelLoc, 1, 0, &model);
    // }
    // if (colorLocation != -1) {
    //     c.glUniform3f(colorLocation, 1.0, 1.0, 1.0); // White color
    // }

    // if (projectionLoc != -1) {
    //     c.glUniformMatrix4fv(projectionLoc, 1, 0, &projection);
    // } else {
    //     std.debug.print("Uniform 'uProjection' not found.\n", .{});
    // }

    // if (viewLoc != -1) {
    //     c.glUniformMatrix4fv(viewLoc, 1, 0, &view);
    // } else {
    //     std.debug.print("Uniform 'uView' not found.\n", .{});
    // }

    // if (modelLoc != -1) {
    //     c.glUniformMatrix4fv(modelLoc, 1, 0, &model);
    // } else {
    //     std.debug.print("Uniform 'uModel' not found.\n", .{});
    // }

    // if (colorLocation != -1) {
    //     c.glUniform3f(colorLocation, 1.0, 0.0, 0.0); // Red color
    // } else {
    //     std.debug.print("Uniform 'uColor' not found.\n", .{});
    // }
    checkOpenGLError("Uniform Setup");

    var numActiveAttributes: c_int = 0;
    c.glGetProgramiv(shaderProgram, c.GL_ACTIVE_ATTRIBUTES, &numActiveAttributes);
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

        std.debug.print("Attribute {d}: {s}, Size: {d}, Type: {x}\n", .{ i, nameBuffer, size, var_type });
    }

    c.glDepthFunc(c.GL_LESS);

    // Main loop
    while (c.glfwWindowShouldClose(window) == 0) {
        c.glClearColor(0.15, 0.15, 0.15, 1.0);
        c.glClear(c.GL_COLOR_BUFFER_BIT | c.GL_DEPTH_BUFFER_BIT);
        checkOpenGLError("Enable Client state");

        // Use shader program
        c.glUseProgram(shaderProgram);
        checkOpenGLError("Using Program");

        c.glBindVertexArray(triangle.VAO);
        checkOpenGLError("Binding VAO");

        // Draw the triangle
        c.glDrawArrays(c.GL_TRIANGLES, 0, 3);
        checkOpenGLError("Drawing Arrays");

        // c.glBindVertexArray(0);
        checkOpenGLError("Binding Vertexes to 0");

        // Draw grid lines
        // const numVertices: c.GLsizei = @intCast(flatVertices.len / 3);
        // c.glDrawArrays(c.GL_LINES, 0, numVertices);

        // Unbind VAO

        // Swap front and back buffers
        c.glfwSwapBuffers(window);
        c.glfwPollEvents();
    }
}
