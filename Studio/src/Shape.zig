// src/Shapes.zig
const std = @import("std");
const Transformations = @import("Transformations.zig");
const Vec3 = Transformations.Vec3;
const Debug = @import("Debug.zig");

const c = @cImport({
    @cInclude("glad/glad.h");
});

const draw = *const fn (mesh: Mesh) void;
const DrawErrors = error{FailedToDraw};

pub const Metadata = struct {
    VAO: u32 = 0,
    VBO: u32 = 0,
    IBO: u32 = 0,
};

pub const Vertex = struct {
    position: [3]f32,
    color: [3]f32,
};

pub const Mesh = struct {
    const Self = @This();

    vertices: []Vertex,
    indices: ?[]u32 = null,
    meta: Metadata,
    draw: ?draw = null,
    drawType: c.GLenum = c.GL_TRIANGLES,
    modelMatrix: [16]f32 = Transformations.identity(),

    pub fn init(vertices: []Vertex, indices: ?[]u32) !Mesh {
        var mesh = Mesh{
            .vertices = vertices,
            .indices = indices,
            .meta = Metadata{},
        };

        // Initialize OpenGL buffers
        c.glGenVertexArrays(1, &mesh.meta.VAO);
        c.glGenBuffers(1, &mesh.meta.VBO);
        c.glGenBuffers(1, &mesh.meta.IBO);

        c.glBindVertexArray(mesh.meta.VAO);

        // Vertex Buffer
        c.glBindBuffer(c.GL_ARRAY_BUFFER, mesh.meta.VBO);
        c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(vertices.len * @sizeOf(Vertex)), vertices.ptr, c.GL_STATIC_DRAW);

        // Index Buffer
        if (indices) |ind| {
            c.glBindBuffer(c.GL_ELEMENT_ARRAY_BUFFER, mesh.meta.IBO);
            c.glBufferData(c.GL_ELEMENT_ARRAY_BUFFER, @intCast(ind.len * @sizeOf(u32)), ind.ptr, c.GL_STATIC_DRAW);
        }

        // Position attribute
        c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex), null);
        c.glEnableVertexAttribArray(0);

        // Color attribute
        const color_offset = @offsetOf(Vertex, "color");
        c.glVertexAttribPointer(1, 3, c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex), @ptrFromInt(color_offset));
        c.glEnableVertexAttribArray(1);

        c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
        c.glBindVertexArray(0);

        return mesh;
    }

    pub fn setFaceColor(self: *Mesh, face_index: usize, color: [3]f32) void {
        const vertices_per_face = 3;
        const start_idx = face_index * vertices_per_face;
        const end_idx = start_idx + vertices_per_face;

        for (start_idx..end_idx) |i| {
            const vertex_idx = self.indices[i];
            self.vertices[vertex_idx].color = color;
        }

        // Update vertex buffer
        c.glBindBuffer(c.GL_ARRAY_BUFFER, self.meta.VBO);
        c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(self.vertices.len * @sizeOf(Vertex)), self.vertices.ptr, c.GL_STATIC_DRAW);
    }

    pub fn setColor(self: *Mesh, color: [3]f32) void {
        for (self.vertices) |*vertex| {
            vertex.color = color;
        }

        // Update vertex buffer
        c.glBindBuffer(c.GL_ARRAY_BUFFER, self.meta.VBO);
        c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(self.vertices.len * @sizeOf(Vertex)), self.vertices.ptr, c.GL_STATIC_DRAW);
    }

    pub fn debug(self: Self) void {
        Debug.printVertexShader(self.meta.VBO, self.vertices.len) catch |err| {
            std.debug.print("Failed to debug vertex shader {any}\n", .{err});
        };
    }
};

pub const Triangle = struct {
    const Self = @This();

    mesh: Mesh,

    pub fn init(allocator: std.mem.Allocator, newVertices: ?[]Vertex) !Mesh {
        // Initialize OpenGL buffers for the triangle
        var vertices: []Vertex = undefined;

        if (newVertices) |verts| {
            vertices = try allocator.dupe(Vertex, verts);
        } else {
            // Allocate space for default vertices
            vertices = Self.default();
            // @memcpy(vertices, &defaults);
        }

        const mesh = try Mesh.init(vertices, null);

        return mesh;
    }

    pub inline fn default() []Vertex {
        var vertices = [_]Vertex{
            Vertex{
                .position = [_]f32{
                    0.0,
                    0.5,
                    0.0,
                },
                .color = [_]f32{
                    1.0,
                    0.0,
                    0.0,
                },
            },
            Vertex{
                .position = [_]f32{
                    -0.5,
                    -0.5,
                    0.0,
                },
                .color = [_]f32{
                    0.0,
                    1.0,
                    0.0,
                },
            },
            Vertex{
                .position = [_]f32{
                    0.5,
                    -0.5,
                    0.0,
                },
                .color = [_]f32{
                    0.0,
                    0.0,
                    1.0,
                },
            },
        };

        return &vertices;
    }
};

// pub const Box = struct {
//     const Self = @This();

//     meta: Metadata = Metadata{},
//     vertices: []f32 = undefined,
//     indices: []u32 = undefined,

//     pub fn init(allocator: std.mem.Allocator, pos: ?Vec3, height: ?f32, width: ?f32, depth: ?f32) !Object {
//         _ = pos;
//         _ = height;
//         _ = width;
//         _ = depth;

//         const defaults = try Self.default(allocator);
//         var rect = Self{};

//         rect.vertices = defaults.vertices;
//         rect.indices = defaults.indices;

//         // Initialize OpenGL buffers for the rectangle
//         c.glGenVertexArrays(1, &rect.meta.VAO);
//         c.glGenBuffers(1, &rect.meta.VBO);
//         c.glGenBuffers(1, &rect.meta.IBO);

//         c.glBindVertexArray(rect.meta.VAO);

//         // Vertex Buffer
//         c.glBindBuffer(c.GL_ARRAY_BUFFER, rect.meta.VBO);
//         c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(rect.vertices.len * @sizeOf(f32)), rect.vertices.ptr, c.GL_STATIC_DRAW);

//         // // Index Buffer
//         c.glBindBuffer(c.GL_ELEMENT_ARRAY_BUFFER, rect.meta.IBO);
//         c.glBufferData(c.GL_ELEMENT_ARRAY_BUFFER, @intCast(rect.indices.len * @sizeOf(u32)), rect.indices.ptr, c.GL_STATIC_DRAW);

//         // Vertex Attributes
//         c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, 3 * @sizeOf(f32), null);
//         c.glEnableVertexAttribArray(0);

//         c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
//         c.glBindVertexArray(0);

//         return rect.info();
//     }

//     pub fn info(self: Self) Object {
//         return Object{
//             .meta = self.meta,
//             .vertices = self.vertices,
//             .indices = self.indices,
//             .modelMatrix = Transformations.identity(),
//             .color = .{ .r = 0.0, .g = 1.0, .b = 0.0 }, // Default color: Green
//         };
//     }

//     pub fn default(allocator: std.mem.Allocator) !struct { vertices: []f32, indices: []u32 } {
//         var vertices: []f32 = try allocator.alloc(f32, 4 * 3 * 2);
//         vertices = @constCast(&[_]f32{ // Rectangle vertices
//             // Front face
//             -0.5, -0.5, 0.5, // 0
//             0.5, -0.5, 0.5, // 1
//             0.5, 0.5, 0.5, // 2
//             -0.5, 0.5, 0.5, // 3

//             // Back face
//             -0.5, -0.5, -0.5, // 4
//             0.5, -0.5, -0.5, // 5
//             0.5, 0.5, -0.5, // 6
//             -0.5, 0.5, -0.5, // 7
//         });

//         var indices: []u32 = try allocator.alloc(u32, 2 * 3 * 6);
//         indices = @constCast(&[_]u32{ // Rectangle indices
//             // Front face
//             0, 1, 2,
//             2, 3, 0,

//             // Right face
//             1, 5, 6,
//             6, 2, 1,

//             // Back face
//             7, 6, 5,
//             5, 4, 7,

//             // Left face
//             4, 0, 3,
//             3, 7, 4,

//             // Bottom face
//             4, 5, 1,
//             1, 0, 4,

//             // Top face
//             3, 2, 6,
//             6, 7, 3,
//         });

//         return .{
//             .vertices = vertices,
//             .indices = indices,
//         };
//     }
// };

// pub const Axis = struct {
//     const Self = @This();

//     meta: Metadata = Metadata{},
//     vertices: []f32 = undefined,

//     pub fn init(allocator: std.mem.Allocator, position: ?Vec3, length: ?f32) !Object {
//         var axis = Self{};

//         if (position != null and length != null) {
//             axis.vertices = try generateVertices(
//                 allocator,
//                 position.?,
//                 length.?,
//             );
//         } else {
//             axis.vertices = try generateVertices(
//                 allocator,
//                 Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
//                 5.0,
//             );
//         }

//         // Initialize OpenGL buffers for the grid
//         c.glGenVertexArrays(1, &axis.meta.VAO);
//         c.glGenBuffers(1, &axis.meta.VBO);

//         c.glBindVertexArray(axis.meta.VAO);

//         // Vertex Buffer
//         c.glBindBuffer(c.GL_ARRAY_BUFFER, axis.meta.VBO);
//         c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(axis.vertices.len * @sizeOf(f32)), @ptrCast(axis.vertices), c.GL_STATIC_DRAW);

//         // Vertex Attributes
//         c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, 3 * @sizeOf(f32), null);
//         c.glEnableVertexAttribArray(0);

//         c.glBindBuffer(c.GL_ARRAY_BUFFER, 0);
//         c.glBindVertexArray(0);

//         return axis.info();
//     }

//     pub fn draw(self: Self) void {
//         // Configure depth testing
//         c.glEnable(c.GL_DEPTH_TEST);
//         c.glDepthFunc(c.GL_LEQUAL);

//         // Enable line smoothing
//         c.glEnable(c.GL_LINE_SMOOTH);
//         c.glHint(c.GL_LINE_SMOOTH_HINT, c.GL_NICEST);

//         // Enable blending for smooth lines
//         c.glEnable(c.GL_BLEND);
//         c.glBlendFunc(c.GL_SRC_ALPHA, c.GL_ONE_MINUS_SRC_ALPHA);

//         // Set line width
//         c.glLineWidth(@as(c.GL_FLOAT, 50.0));

//         // Small offset to prevent z-fighting
//         c.glEnable(c.GL_POLYGON_OFFSET_LINE);
//         c.glPolygonOffset(-1.0, -1.0);

//         c.glBindVertexArray(self.meta.VAO);
//         c.glDrawArrays(c.GL_LINES, 0, @intCast(self.vertices.len / 3));
//         c.glBindVertexArray(0);

//         // Reset states
//         c.glDisable(c.GL_POLYGON_OFFSET_LINE);
//         c.glDisable(c.GL_BLEND);
//         c.glDisable(c.GL_LINE_SMOOTH);
//     }

//     pub fn generateVertices(allocator: std.mem.Allocator, position: Vec3, length: f32) ![]f32 {
//         const vertex_count: usize = @intFromFloat(3 * 3 * 2);
//         var vertices: []f32 = try allocator.alloc(f32, vertex_count);
//         var index: usize = 0;

//         while (index < vertex_count) {
//             vertices[index] = position.x;
//             vertices[index + 1] = position.y;
//             vertices[index + 2] = position.z;
//             index += 3;

//             switch ((index / 2) / 3) {
//                 0 => {
//                     vertices[index] = position.x + length;
//                     vertices[index + 1] = position.y;
//                     vertices[index + 2] = position.z;
//                 },
//                 1 => {
//                     vertices[index] = position.x;
//                     vertices[index + 1] = position.y + length;
//                     vertices[index + 2] = position.z;
//                 },
//                 2 => {
//                     vertices[index] = position.x;
//                     vertices[index + 1] = position.y;
//                     vertices[index + 2] = position.z + length;
//                 },
//                 else => unreachable,
//             }

//             index += 3;
//         }

//         return vertices;
//     }

//     pub fn info(self: Self) Object {
//         return Object{
//             .meta = self.meta,
//             .vertices = self.vertices,
//             .modelMatrix = Transformations.identity(),
//             .color = comptime .{ .r = 0 / 255, .g = 255 / 255, .b = 0 / 225 },
//             .drawType = c.GL_LINES,
//         };
//     }
// };

pub const Grid = struct {
    const Self = @This();

    mesh: Mesh,

    pub fn init(allocator: std.mem.Allocator, gridSize: ?usize, spacing: ?f32) !Mesh {
        // Generate grid vertices
        var vertices: []Vertex = undefined;

        if (gridSize != null and spacing != null) {
            vertices = try Grid.generateGridVertices(allocator, gridSize.?, spacing.?);
        } else {
            vertices = try Grid.generateGridVertices(allocator, 1000, 1.0);
        }

        var mesh: Mesh = try Mesh.init(vertices, null);
        mesh.drawType = c.GL_LINES;
        mesh.draw = Grid.draw;

        return mesh;
    }

    pub fn draw(mesh: Mesh) void {
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

        // Bind the mesh's VAO
        c.glBindVertexArray(mesh.meta.VAO);

        // Setup vertex attributes for position and color
        c.glBindBuffer(c.GL_ARRAY_BUFFER, mesh.meta.VBO);

        // Position attribute (location = 0)
        c.glVertexAttribPointer(0, // location
            3, // size (vec3)
            c.GL_FLOAT, // type
            c.GL_FALSE, // normalized
            @sizeOf(Vertex), // stride (size of entire vertex struct)
            null // offset for position (starts at beginning)
        );
        c.glEnableVertexAttribArray(0);

        // Color attribute (location = 1)
        const color_offset = @offsetOf(Vertex, "color");
        c.glVertexAttribPointer(1, // location
            3, // size (vec3)
            c.GL_FLOAT, // type
            c.GL_FALSE, // normalized
            @sizeOf(Vertex), // stride (size of entire vertex struct)
            @ptrFromInt(color_offset) // offset for color
        );
        c.glEnableVertexAttribArray(1);
        c.glDrawArrays(c.GL_LINES, 0, @intCast(mesh.vertices.len));

        // Disable vertex attributes
        c.glDisableVertexAttribArray(0);
        c.glDisableVertexAttribArray(1);

        c.glBindVertexArray(0);

        // Reset states
        c.glDisable(c.GL_POLYGON_OFFSET_LINE);
        c.glDisable(c.GL_BLEND);
        c.glDisable(c.GL_LINE_SMOOTH);
    }

    pub fn generateGridVertices(allocator: std.mem.Allocator, gridSize: usize, spacing: f32) ![]Vertex {
        const totalLines = gridSize * 2 + 1;
        const totalVertices = totalLines * 2 * 2;

        const xColor = [_]f32{
            comptime (118.0 / 255.0),
            comptime (91.0 / 255.0),
            comptime (226.0 / 255.0),
        };

        const zColor = [_]f32{
            comptime (224.0 / 255.0),
            comptime (176.0 / 255.0),
            comptime (255.0 / 255.0),
        };

        var vertices = try allocator.alloc(Vertex, totalVertices);
        var index: usize = 0;
        const halfSize = @as(f32, @floatFromInt(gridSize)) * spacing;

        const y_coord = 0.0;

        // x Lines
        var i: f32 = -halfSize;
        while (i <= halfSize) : (i += spacing) {
            vertices[index] = Vertex{
                .position = [_]f32{
                    -halfSize * 0.05,
                    y_coord,
                    i * 0.05,
                },
                .color = xColor,
            };
            index += 1;

            vertices[index] = Vertex{
                .position = [_]f32{
                    halfSize * 0.05,
                    y_coord,
                    i * 0.05,
                },
                .color = xColor,
            };

            index += 1;
        }

        // z Lines
        i = -halfSize;
        while (i <= halfSize) : (i += spacing) {
            vertices[index] = Vertex{
                .position = [_]f32{
                    i * 0.05,
                    y_coord,
                    -halfSize * 0.05,
                },
                .color = zColor,
            };
            index += 1;

            vertices[index] = Vertex{
                .position = [_]f32{
                    i * 0.05,
                    y_coord,
                    halfSize * 0.05,
                },
                .color = zColor,
            };
            index += 1;
        }

        return vertices;
    }
};
