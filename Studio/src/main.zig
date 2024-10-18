const std = @import("std");
const File = std.fs.File;
const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("glad/glad.h"); // Include GLAD
    @cInclude("GLFW/glfw3.h");
    // Add OpenGL function declarations if needed
});

const ShaderError = error{ UnableToCreateShader, ShaderCompilationFailed, UnableToCreateProgram, ShaderLinkingFailed };

// Function to generate grid vertices
fn generateGridVertices() []const f32 {
    const allocator = std.heap.page_allocator;
    const vertices = try allocator.alloc(f32, 1000);
    defer allocator.free(vertices);

    const gridSize: i32 = 10;
    const spacing: f32 = 1.0;

    // Lines parallel to X-axis (varying Z)
    for (0..gridSize) |gridItem| {
        try std.array.append(&vertices, -f32(gridItem) * spacing);
        try std.array.append(&vertices, 0.0);
        try std.array.append(&vertices, f32(gridSize) * spacing);

        try std.array.append(&vertices, f32(gridItem) * spacing);
        try std.array.append(&vertices, 0.0);
        try std.array.append(&vertices, f32(gridSize) * spacing);
    }

    // Lines parallel to Z-axis (varying X)
    for (0..gridSize) |gridItem| {
        try std.array.append(&vertices, f32(gridItem) * spacing);
        try std.array.append(&vertices, 0.0);
        try std.array.append(&vertices, -f32(gridSize) * spacing);

        try std.array.append(&vertices, f32(gridItem) * spacing);
        try std.array.append(&vertices, 0.0);
        try std.array.append(&vertices, f32(gridSize) * spacing);
    }

    return vertices;
}

const Grid = struct { VAO: *const u32, VBO: *const u32 };

// Function to create VAO and VBO for the grid
fn setupGrid(vertices: []const f32) !Grid {
    var vao: u32 = 0;
    var vbo: u32 = 0;

    c.glGenVertexArrays(1, &vao);
    c.glGenBuffers(1, &vbo);

    c.glBindVertexArray(vao);

    c.glBindBuffer(c.GL_ARRAY_BUFFER, vbo);
    c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(vertices.len * @sizeOf(f32)), vertices.ptr, c.GL_STATIC_DRAW);

    c.glVertexAttribPointer(0, 3, c.GL_FLOAT, 0, @intCast(3 * @sizeOf(f32)), @ptrFromInt(0));
    c.glEnableVertexAttribArray(0);

    c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
    c.glBindVertexArray(0);

    return Grid{ .VAO = &vao, .VBO = &vbo };
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
    if (shader == 0) return ShaderError.UnableToCreateShader;

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

    const vertexShader = try compileShader(c.GL_VERTEX_SHADER, vertexSource);
    const fragmentShader = try compileShader(c.GL_FRAGMENT_SHADER, fragmentSource);

    const shaderProgram = c.glCreateProgram();
    if (shaderProgram == 0) return ShaderError.UnableToCreateProgram;

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
    // return 0;
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
    c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MAJOR, 4);
    c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MINOR, 1);
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
    c.glViewport(0, 0, 800, 600);

    // Enable depth testing if needed
    c.glEnable(c.GL_DEPTH_TEST);

    // Compile and link shaders
    const shaderProgram = try createShaderProgram("src/shaders/vertex_shader.glsl", "src/shaders/fragment_shader.glsl");
    defer c.glDeleteProgram(shaderProgram);

    // Generate grid vertices
    // const gridVertices = generateGridVertices();

    // Set up grid VAO and VBO
    // const grid = try setupGrid(gridVertices);
    // defer c.glDeleteVertexArrays(1, grid.VAO);
    // defer c.glDeleteBuffers(1, grid.VBO);

    // Set uniform color for the grid (e.g., white)
    c.glUseProgram(shaderProgram);
    const colorLocation = c.glGetUniformLocation(shaderProgram, "uColor");
    if (colorLocation != -1) {
        c.glUniform3f(colorLocation, 1.0, 1.0, 1.0); // White color
    }

    // Main loop
    while (c.glfwWindowShouldClose(window) == 0) {
        c.glfwPollEvents();

        // Set the clear color to dark gray
        c.glClearColor(0.15, 0.15, 0.15, 1.0);
        c.glClear(c.GL_COLOR_BUFFER_BIT);

        // Use shader program
        c.glUseProgram(shaderProgram);

        // Bind the grid VAO
        // c.glBindVertexArray(grid.VAO);

        // Draw grid lines
        // const numLines: i32 = @intCast(gridVertices.len / 3);
        // c.glDrawArrays(c.GL_LINES, 0, numLines);

        // Unbind VAO
        // c.glBindVertexArray(0);

        // Swap front and back buffers
        c.glfwSwapBuffers(window);
    }
}
