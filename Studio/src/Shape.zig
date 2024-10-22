// src/Shapes.zig
const std = @import("std");
const Transformations = @import("Transformations.zig");
const Vec3 = Transformations.Vec3;
const Debug = @import("Debug.zig");

const c = @cImport({
    @cInclude("glad/glad.h");
});

const DrawErrors = error{FailedToDraw};

pub const Metadata = struct {
    VAO: u32 = 0,
    VBO: u32 = 0,
    EBO: u32 = 0,
};

const draw = *const fn () void;

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
    drawType: c.GLenum = c.GL_TRIANGLES,
    draw: ?draw = null,

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

        return .{
            .vertices = &vertices,
            .indices = &indices,
        };
    }
};

pub const Axis = struct {
    const Self = @This();

    meta: Metadata = Metadata{},
    vertices: []f32 = undefined,

    pub fn init(allocator: std.mem.Allocator, position: ?Vec3, length: ?f32) !Object {
        var axis = Self{};

        if (position != null and length != null) {
            axis.vertices = try generateVertices(
                allocator,
                position.?,
                length.?,
            );
        } else {
            axis.vertices = try generateVertices(
                allocator,
                Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
                5.0,
            );
        }

        // Initialize OpenGL buffers for the grid
        c.glGenVertexArrays(1, &axis.meta.VAO);
        c.glGenBuffers(1, &axis.meta.VBO);

        c.glBindVertexArray(axis.meta.VAO);

        // Vertex Buffer
        c.glBindBuffer(c.GL_ARRAY_BUFFER, axis.meta.VBO);
        c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(axis.vertices.len * @sizeOf(f32)), @ptrCast(axis.vertices), c.GL_STATIC_DRAW);

        // Vertex Attributes
        c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, 3 * @sizeOf(f32), null);
        c.glEnableVertexAttribArray(0);

        c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
        c.glBindVertexArray(0);

        return axis.info();
    }

    pub fn draw(self: Self) void {
        // Configure depth testing
        c.glEnable(c.GL_DEPTH_TEST);
        c.glDepthFunc(c.GL_LEQUAL);

        // Enable line smoothing
        c.glEnable(c.GL_LINE_SMOOTH);
        c.glHint(c.GL_LINE_SMOOTH_HINT, c.GL_NICEST);

        // Enable blending for smooth lines
        c.glEnable(c.GL_BLEND);
        c.glBlendFunc(c.GL_SRC_ALPHA, c.GL_ONE_MINUS_SRC_ALPHA);

        // Set line width
        c.glLineWidth(@as(c.GL_FLOAT, 50.0));

        // Small offset to prevent z-fighting
        c.glEnable(c.GL_POLYGON_OFFSET_LINE);
        c.glPolygonOffset(-1.0, -1.0);

        c.glBindVertexArray(self.meta.VAO);
        c.glDrawArrays(c.GL_LINES, 0, @intCast(self.vertices.len / 3));
        c.glBindVertexArray(0);

        // Reset states
        c.glDisable(c.GL_POLYGON_OFFSET_LINE);
        c.glDisable(c.GL_BLEND);
        c.glDisable(c.GL_LINE_SMOOTH);
    }

    pub fn generateVertices(allocator: std.mem.Allocator, position: Vec3, length: f32) ![]f32 {
        const vertex_count: usize = @intFromFloat(3 * 3 * 2);
        var vertices: []f32 = try allocator.alloc(f32, vertex_count);
        var index: usize = 0;

        while (index < vertex_count) {
            vertices[index] = position.x;
            vertices[index + 1] = position.y;
            vertices[index + 2] = position.z;
            index += 3;

            switch ((index / 2) / 3) {
                0 => {
                    vertices[index] = position.x + length;
                    vertices[index + 1] = position.y;
                    vertices[index + 2] = position.z;
                },
                1 => {
                    vertices[index] = position.x;
                    vertices[index + 1] = position.y + length;
                    vertices[index + 2] = position.z;
                },
                2 => {
                    vertices[index] = position.x;
                    vertices[index + 1] = position.y;
                    vertices[index + 2] = position.z + length;
                },
                else => unreachable,
            }

            index += 3;
        }

        return vertices;
    }

    pub fn info(self: Self) Object {
        return Object{
            .meta = self.meta,
            .vertices = self.vertices,
            .modelMatrix = Transformations.identity(),
            .color = comptime .{ .r = 0 / 255, .g = 255 / 255, .b = 0 / 225 },
            .drawType = c.GL_LINES,
        };
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
            grid.vertices = try Grid.generateGridVertices(allocator, 1000, 1.0);
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
            .color = comptime .{ .r = 118.0 / 255.0, .g = 91.0 / 255.0, .b = 226.0 / 255.0 }, // Default color: White
            .drawType = c.GL_LINES,
        };
    }

    pub fn draw(self: Self) DrawErrors!void {
        // Configure depth testing
        c.glEnable(c.GL_DEPTH_TEST);
        c.glDepthFunc(c.GL_LEQUAL);

        // Enable line smoothing
        c.glEnable(c.GL_LINE_SMOOTH);
        c.glHint(c.GL_LINE_SMOOTH_HINT, c.GL_NICEST);

        // Enable blending for smooth lines
        c.glEnable(c.GL_BLEND);
        c.glBlendFunc(c.GL_SRC_ALPHA, c.GL_ONE_MINUS_SRC_ALPHA);

        // Set line width
        c.glLineWidth(1.0);

        // Small offset to prevent z-fighting
        c.glEnable(c.GL_POLYGON_OFFSET_LINE);
        c.glPolygonOffset(-1.0, -1.0);

        c.glBindVertexArray(self.meta.VAO);
        c.glDrawArrays(c.GL_LINES, 0, @intCast(self.vertices.len / 3));
        c.glBindVertexArray(0);

        // Reset states
        c.glDisable(c.GL_POLYGON_OFFSET_LINE);
        c.glDisable(c.GL_BLEND);
        c.glDisable(c.GL_LINE_SMOOTH);
    }

    pub fn generateGridVertices(allocator: std.mem.Allocator, gridSize: usize, spacing: f32) ![]f32 {
        const totalLines = gridSize * 2 + 1;
        const totalVertices = totalLines * 2 * 2;
        const vertexComponents = totalVertices * 3;

        var vertices = try allocator.alloc(f32, vertexComponents);
        var index: usize = 0;
        const halfSize = @as(f32, @floatFromInt(gridSize)) * spacing;

        const y_coord = 0.0;

        // Generate horizontal lines (with a tiny Y offset)
        var i: f32 = -halfSize;
        while (i <= halfSize) : (i += spacing) {
            vertices[index] = -halfSize * 0.05;
            vertices[index + 1] = y_coord; // Small Y offset for horizontal lines
            vertices[index + 2] = i * 0.05;
            index += 3;

            vertices[index] = halfSize * 0.05;
            vertices[index + 1] = y_coord; // Same offset
            vertices[index + 2] = i * 0.05;
            index += 3;
        }

        // Vertical lines remain at y=0
        i = -halfSize;
        while (i <= halfSize) : (i += spacing) {
            vertices[index] = i * 0.05;
            vertices[index + 1] = y_coord;
            vertices[index + 2] = -halfSize * 0.05;
            index += 3;

            vertices[index] = i * 0.05;
            vertices[index + 1] = y_coord;
            vertices[index + 2] = halfSize * 0.05;
            index += 3;
        }

        return vertices;
    }
};
