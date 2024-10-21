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

    pub fn init(newVertices: ?[]f32) !Object {
        // Initialize OpenGL buffers for the triangle

        const defaultVertices = Self.default();
        var triangle = Self{};

        if (newVertices) |verts| {
            triangle.vertices = verts;
        } else {
            triangle.vertices = defaultVertices;
        }

        c.glGenVertexArrays(1, &triangle.meta.VAO);
        c.glGenBuffers(1, &triangle.meta.VBO);

        c.glBindVertexArray(triangle.meta.VAO);

        c.glBindBuffer(c.GL_ARRAY_BUFFER, triangle.meta.VBO);
        c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(triangle.vertices.len * @sizeOf(f32)), triangle.vertices.ptr, c.GL_STATIC_DRAW);

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

    pub inline fn default() []f32 {
        var vertices = [_]f32{ // Default triangle vertices
            0.0, 0.5, 0.0, // Top
            -0.5, -0.5, 0.0, // Bottom Left
            0.5, -0.5, 0.0, // Bottom Right
        };
        return &vertices;
    }
};

pub const Box = struct {
    const Self = @This();

    meta: Metadata = Metadata{},
    vertices: []f32 = undefined,
    indices: []u32 = undefined,

    pub fn init() !Object {
        const defaults = Self.default();
        var rect = Self{};

        rect.vertices = defaults.vertices;
        // rect.indices = defaults.indices;

        // Initialize OpenGL buffers for the rectangle
        c.glGenVertexArrays(1, &rect.meta.VAO);
        c.glGenBuffers(1, &rect.meta.VBO);
        // c.glGenBuffers(1, &rect.meta.EBO);

        c.glBindVertexArray(rect.meta.VAO);

        // Vertex Buffer
        c.glBindBuffer(c.GL_ARRAY_BUFFER, rect.meta.VBO);
        c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(rect.vertices.len * @sizeOf(f32)), rect.vertices.ptr, c.GL_STATIC_DRAW);

        // // Element Buffer
        // c.glBindBuffer(c.GL_ELEMENT_ARRAY_BUFFER, rect.meta.EBO);
        // c.glBufferData(c.GL_ELEMENT_ARRAY_BUFFER, @intCast(rect.indices.len * @sizeOf(u32)), rect.indices.ptr, c.GL_STATIC_DRAW);

        // Vertex Attributes
        c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, 3 * @sizeOf(f32), null);
        c.glEnableVertexAttribArray(0);

        c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
        c.glBindVertexArray(0);

        return rect.info();
    }

    pub fn info(self: Self) Object {
        return Object{
            .meta = self.meta,
            .vertices = self.vertices,
            .indices = self.indices,
            .modelMatrix = Transformations.identity(),
            .color = .{ .r = 0.0, .g = 1.0, .b = 0.0 }, // Default color: Green
        };
    }

    pub fn default() struct { vertices: []f32, indices: []u32 } {
        var vertices = [_]f32{ // Rectangle vertices
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
        };

        var indices = [_]u32{ // Rectangle indices
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
        };

        return .{ .vertices = &vertices, .indices = &indices };
    }
};

pub const Grid = struct {
    const Self = @This();

    meta: Metadata = Metadata{},
    vertices: []f32 = undefined,

    pub fn init(allocator: std.mem.Allocator, gridSize: ?usize, spacing: ?f32) !Object {
        // Generate grid vertices
        var grid = Self{};

        if (gridSize != null and spacing != null) {
            grid.vertices = try Grid.generateGridVertices(allocator, gridSize.?, spacing.?);
        } else {
            grid.vertices = try Grid.generateGridVertices(allocator, 10, 10);
        }

        // Initialize OpenGL buffers for the grid
        c.glGenVertexArrays(1, &grid.meta.VAO);
        c.glGenBuffers(1, &grid.meta.VBO);

        c.glBindVertexArray(grid.meta.VAO);

        // Vertex Buffer
        c.glBindBuffer(c.GL_ARRAY_BUFFER, grid.meta.VBO);
        c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(grid.vertices.len * @sizeOf(f32)), @ptrCast(grid.vertices), c.GL_STATIC_DRAW);

        // Vertex Attributes
        c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, 3 * @sizeOf(f32), null);
        c.glEnableVertexAttribArray(0);

        c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
        c.glBindVertexArray(0);

        return grid.info();
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.vertices);
        c.glDeleteBuffers(1, &self.meta.VBO);
        c.glDeleteVertexArrays(1, &self.meta.VAO);
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

    pub inline fn generateGridVertices(allocator: std.mem.Allocator, gridSize: usize, spacing: f32) ![]f32 {
        const totalLines = gridSize * 2 + 1;
        const totalVertices = totalLines * 2 * 3; // Two vertices per line, three components (x, y, z) each

        // Allocate memory for vertices
        var vertices = try allocator.alloc(f32, totalVertices);

        var index: usize = 0;
        const halfSize = gridSize;

        // Generate lines parallel to the X-axis (varying Z)
        for (0..totalLines) |i| {
            const z = @as(f32, @floatFromInt(i - halfSize)) * spacing;

            // Start vertex (left)
            vertices[index] = -@as(f32, @floatFromInt(halfSize)) * spacing; // x
            vertices[index + 1] = 0.0; // y
            vertices[index + 2] = @as(f32, z) * 1.0; // z
            index += 3;

            // End vertex (right)
            vertices[index] = @as(f32, @floatFromInt(halfSize)) * spacing; // x
            vertices[index + 1] = 0.0; // y
            vertices[index + 2] = @as(f32, z) * 1.0; // z
            index += 3;
        }

        // Generate lines parallel to the Z-axis (varying X)
        for (0..totalLines) |i| {
            const x = @as(f32, @floatFromInt(i - halfSize)) * spacing;

            // Start vertex (bottom)
            vertices[index] = @as(f32, x) * 1.0; // x
            vertices[index + 1] = 0.0; // y
            vertices[index + 2] = -@as(f32, @floatFromInt(halfSize)) * spacing; // z
            index += 3;

            // End vertex (top)
            vertices[index] = @as(f32, x) * 1.0; // x
            vertices[index + 1] = 0.0; // y
            vertices[index + 2] = @as(f32, @floatFromInt(halfSize)) * spacing; // z
            index += 3;
        }

        return vertices;
    }
};
