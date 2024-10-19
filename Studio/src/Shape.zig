// src/Shapes.zig
const std = @import("std");
const Transformations = @import("Transformations.zig");
const Debug = @import("Debug.zig");

const c = @cImport({
    @cInclude("glad/glad.h");
});

pub const Metadata = struct {
    VAO: u32 = 0,
    VBO: u32 = 0,
    EBO: u32 = 0,
};

pub const Object = struct {
    const Self = @This();

    meta: Metadata,
    vertices: []f32,
    indices: []u32 = &.{}, // Default to empty if not provided
    modelMatrix: [16]f32 = Transformations.identity(),
    color: struct {
        r: f32 = 1.0,
        g: f32 = 1.0,
        b: f32 = 1.0,
    },

    pub fn debug(self: Self) void {
        Debug.printVertexShader(self.meta.VBO, self.vertices.len) catch |err| {
            std.debug.print("Failed to debug vertex shader {any}\n", .{err});
        };
    }
};

pub const Triangle = struct {
    const Self = @This();

    meta: Metadata = Metadata{},
    vertices: []f32 = undefined,

    pub fn init(allocator: std.mem.Allocator, newVertices: ?[]f32) !Object {
        // Initialize OpenGL buffers for the triangle

        var triangle = Self{};
        const defaultVertices = Triangle.default();

        triangle.vertices = try allocator.alloc(f32, defaultVertices.len);

        if (newVertices) |verts| {
            triangle.vertices = verts;
        } else {
            @memcpy(triangle.vertices, &defaultVertices);
        }

        const vertices = triangle.vertices;

        c.glGenVertexArrays(1, &triangle.meta.VAO);
        c.glGenBuffers(1, &triangle.meta.VBO);

        c.glBindVertexArray(triangle.meta.VAO);

        c.glBindBuffer(c.GL_ARRAY_BUFFER, triangle.meta.VBO);
        c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(vertices.len * @sizeOf(f32)), vertices.ptr, c.GL_STATIC_DRAW);

        // Vertex Attributes
        c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, 3 * @sizeOf(f32), null);
        c.glEnableVertexAttribArray(0);

        c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
        c.glBindVertexArray(0);

        return triangle.info();
    }

    pub fn deinit(self: Self, allocator: std.mem.Allocator) !void {
        allocator.destroy(self.vertices);
    }

    pub fn info(self: Self) Object {
        return Object{
            .meta = self.meta,
            .vertices = self.vertices,
            .indices = &.{}, // No indices for triangle
            .modelMatrix = Transformations.identity(),
            .color = .{ .r = 1.0, .g = 0.0, .b = 0.0 }, // Default color: Red
        };
    }

    pub inline fn default() [9]f32 {
        return .{ // Default triangle vertices
            0.0, 0.5, 0.0, // Top
            -0.5, -0.5, 0.0, // Bottom Left
            0.5, -0.5, 0.0, // Bottom Right
        };
    }
};

pub const Rectangle = struct {
    meta: Metadata,
    vertices: []f32 = &.{ // Rectangle vertices
        // Front face
        -0.5, -0.5, 0.5, // 0
        0.5, -0.5, 0.5, // 1
        0.5, 0.5, 0.5, // 2
        -0.5, 0.5, 0.5, // 3

        // Back face
        -0.5, -0.5, -0.5, // 4
        0.5, -0.5, -0.5, // 5
        0.5, 0.5, -0.5, // 6
        -0.5, 0.5, -0.5, // 7
    },
    indices: []u32 = &.{ // Rectangle indices
        // Front face
        0, 1, 2,
        2, 3, 0,

        // Right face
        1, 5, 6,
        6, 2, 1,

        // Back face
        7, 6, 5,
        5, 4, 7,

        // Left face
        4, 0, 3,
        3, 7, 4,

        // Bottom face
        4, 5, 1,
        1, 0, 4,

        // Top face
        3, 2, 6,
        6, 7, 3,
    },

    pub fn init(self: *Rectangle) !Object {
        // Initialize OpenGL buffers for the rectangle
        c.glGenVertexArrays(1, &self.meta.VAO);
        c.glGenBuffers(1, &self.meta.VBO);
        c.glGenBuffers(1, &self.meta.EBO);

        c.glBindVertexArray(self.meta.VAO);

        // Vertex Buffer
        c.glBindBuffer(c.GL_ARRAY_BUFFER, self.meta.VBO);
        c.glBufferData(c.GL_ARRAY_BUFFER, self.vertices.len * @sizeOf(f32), @ptrCast(self.vertices), c.GL_STATIC_DRAW);

        // Element Buffer
        c.glBindBuffer(c.GL_ELEMENT_ARRAY_BUFFER, self.meta.EBO);
        c.glBufferData(c.GL_ELEMENT_ARRAY_BUFFER, self.indices.len * @sizeOf(u32), @ptrCast(self.indices), c.GL_STATIC_DRAW);

        // Vertex Attributes
        c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, 3 * @sizeOf(f32), null);
        c.glEnableVertexAttribArray(0);

        c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
        c.glBindVertexArray(0);

        return self.info();
    }

    pub fn info(self: Rectangle) Object {
        return Object{
            .meta = self.meta,
            .vertices = self.vertices,
            .indices = self.indices,
            .modelMatrix = Transformations.identity(),
            .color = .{ .r = 0.0, .g = 1.0, .b = 0.0 }, // Default color: Green
        };
    }
};

pub const Grid = struct {
    const Self = @This();

    meta: Metadata,
    vertices: []f32,
    pub fn init(self: Self, allocator: std.mem.Allocator, gridSize: i32, spacing: f32) !Object {
        // Generate grid vertices
        self.vertices = Grid.generateGridVertices(allocator, gridSize, spacing);

        // Initialize OpenGL buffers for the grid
        c.glGenVertexArrays(1, &self.meta.VAO);
        c.glGenBuffers(1, &self.meta.VBO);

        c.glBindVertexArray(self.meta.VAO);

        // Vertex Buffer
        c.glBindBuffer(c.GL_ARRAY_BUFFER, self.meta.VBO);
        c.glBufferData(c.GL_ARRAY_BUFFER, self.vertices.len * @sizeOf(f32), @ptrCast(self.vertices), c.GL_STATIC_DRAW);

        // Vertex Attributes
        c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, 3 * @sizeOf(f32), null);
        c.glEnableVertexAttribArray(0);

        c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
        c.glBindVertexArray(0);

        return self.info();
    }

    pub fn deinit(self: Self, allocator: std.mem.Allocator) !void {
        allocator.destroy(self.vertices);
    }

    pub fn info(self: Self) Object {
        return Object{
            .meta = self.meta,
            .vertices = self.vertices,
            .indices = &.{}, // No indices for grid lines
            .modelMatrix = Transformations.identity(),
            .color = .{ .r = 1.0, .g = 1.0, .b = 1.0 }, // Default color: White
        };
    }

    pub inline fn generateGridVertices(allocator: std.mem.Allocator, comptime gridSize: i32, comptime spacing: f32) []f32 {
        const totalLines = gridSize * 2; // X and Z axes
        const totalVertices = totalLines * 2; // Two vertices per line
        var vertices = []f32{};

        // Allocate memory for vertices
        vertices = allocator.alloc(f32, totalVertices * 3) orelse @panic("Allocation failed");

        var index: usize = 0;

        // Generate lines parallel to X-axis (varying Z)
        for (0..gridSize) |i| {
            const z = @as(f32, @floatFromInt(i)) * spacing;
            // Start vertex
            vertices[index] = -1.0 * @as(f32, @floatFromInt(gridSize / 2)) * spacing; // x
            vertices[index + 1] = 0.0; // y
            vertices[index + 2] = z; // z
            index += 3;

            // End vertex
            vertices[index] = @as(f32, @floatFromInt(gridSize / 2)) * spacing; // x
            vertices[index + 1] = 0.0; // y
            vertices[index + 2] = z; // z
            index += 3;
        }

        // Generate lines parallel to Z-axis (varying X)
        for (0..gridSize) |i| {
            const x = @as(f32, @floatFromInt(i)) * spacing;
            // Start vertex
            vertices[index] = x; // x
            vertices[index + 1] = 0.0; // y
            vertices[index + 2] = -1 * @as(f32, @floatFromInt(gridSize / 2)) * spacing; // z
            index += 3;

            // End vertex
            vertices[index] = x; // x
            vertices[index + 1] = 0.0; // y
            vertices[index + 2] = @as(f32, @floatFromInt(gridSize / 2)) * spacing; // z
            index += 3;
        }

        return vertices;
    }

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
};
