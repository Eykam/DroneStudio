// src/Shapes.zig
const std = @import("std");
const Math = @import("Math.zig");
const Mesh = @import("Mesh.zig");
const Node = @import("Node.zig");
const gl = @import("gl.zig");
const Vertex = Mesh.Vertex;
const Vec3 = Math.Vec3;
const glad = gl.glad;

const glCheckError = @import("Debug.zig").glCheckError;

const DrawErrors = error{FailedToDraw};

pub const Triangle = struct {
    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, position: ?Vec3, newVertices: ?[]Vertex) !*Node {
        var vertices: []Vertex = undefined;

        if (newVertices) |verts| {
            vertices = try allocator.dupe(Vertex, verts);
        } else {
            vertices = Self.default(position);
        }

        const mesh = try Mesh.init(vertices, null, null);
        const node = try Node.init(allocator, mesh);

        return node;
    }

    pub inline fn default(origin: ?Vec3) []Vertex {
        var pos: Vec3 = undefined;

        if (origin) |position| {
            pos = position;
        } else {
            pos = Vec3{
                .x = 0.0,
                .y = 0.0,
                .z = 0.0,
            };
        }

        var vertices = [_]Vertex{
            Vertex{
                .position = .{ 0.0 + pos.x, 0.5 + pos.y, 0.0 + pos.z },
                .color = .{ 1.0, 0.0, 0.0 },
            },
            Vertex{
                .position = .{ -0.5 + pos.x, -0.5 + pos.y, 0.0 + pos.z },
                .color = .{ 0.0, 1.0, 0.0 },
            },
            Vertex{
                .position = .{ 0.5 + pos.x, -0.5 + pos.y, 0.0 + pos.z },
                .color = .{ 0.0, 0.0, 1.0 },
            },
        };

        return &vertices;
    }
};

pub const Box = struct {
    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, pos: ?Vec3, height: ?f32, width: ?f32, depth: ?f32) !*Node {
        _ = pos;
        _ = height;
        _ = width;
        _ = depth;

        const defaults = try Self.default(allocator);

        const mesh = try Mesh.init(defaults.vertices, defaults.indices, null);
        const node = try Node.init(allocator, mesh);

        return node;
    }

    pub fn default(allocator: std.mem.Allocator) !struct { vertices: []Vertex, indices: []u32 } {
        // (4 vertices per face * 6 faces)
        var vertices: []Vertex = try allocator.alloc(Vertex, 24);

        const front_color = [3]f32{ 0.9, 0.9, 0.9 }; // Lightest gray (front)
        const right_color = [3]f32{ 0.8, 0.8, 0.8 }; // Slightly darker
        const back_color = [3]f32{ 0.6, 0.6, 0.6 }; // Darkest (back)
        const left_color = [3]f32{ 0.75, 0.75, 0.75 }; // Medium-dark
        const bottom_color = [3]f32{ 0.7, 0.7, 0.7 }; // Darker for bottom
        const top_color = [3]f32{ 0.85, 0.85, 0.85 }; // Lighter for top

        vertices = @constCast(&[_]Vertex{
            // Front face (0-3)
            .{ .position = .{ -0.5, -0.5, 0.5 }, .color = front_color },
            .{ .position = .{ 0.5, -0.5, 0.5 }, .color = front_color },
            .{ .position = .{ 0.5, 0.5, 0.5 }, .color = front_color },
            .{ .position = .{ -0.5, 0.5, 0.5 }, .color = front_color },

            // Right face (4-7)
            .{ .position = .{ 0.5, -0.5, 0.5 }, .color = right_color },
            .{ .position = .{ 0.5, -0.5, -0.5 }, .color = right_color },
            .{ .position = .{ 0.5, 0.5, -0.5 }, .color = right_color },
            .{ .position = .{ 0.5, 0.5, 0.5 }, .color = right_color },

            // Back face (8-11)
            .{ .position = .{ -0.5, -0.5, -0.5 }, .color = back_color },
            .{ .position = .{ 0.5, -0.5, -0.5 }, .color = back_color },
            .{ .position = .{ 0.5, 0.5, -0.5 }, .color = back_color },
            .{ .position = .{ -0.5, 0.5, -0.5 }, .color = back_color },

            // Left face (12-15)
            .{ .position = .{ -0.5, -0.5, -0.5 }, .color = left_color },
            .{ .position = .{ -0.5, -0.5, 0.5 }, .color = left_color },
            .{ .position = .{ -0.5, 0.5, 0.5 }, .color = left_color },
            .{ .position = .{ -0.5, 0.5, -0.5 }, .color = left_color },

            // Bottom face (16-19)
            .{ .position = .{ -0.5, -0.5, -0.5 }, .color = bottom_color },
            .{ .position = .{ 0.5, -0.5, -0.5 }, .color = bottom_color },
            .{ .position = .{ 0.5, -0.5, 0.5 }, .color = bottom_color },
            .{ .position = .{ -0.5, -0.5, 0.5 }, .color = bottom_color },

            // Top face (20-23)
            .{ .position = .{ -0.5, 0.5, 0.5 }, .color = top_color },
            .{ .position = .{ 0.5, 0.5, 0.5 }, .color = top_color },
            .{ .position = .{ 0.5, 0.5, -0.5 }, .color = top_color },
            .{ .position = .{ -0.5, 0.5, -0.5 }, .color = top_color },
        });

        var indices: []u32 = try allocator.alloc(u32, 36); // 6 faces * 2 triangles * 3 vertices
        indices = @constCast(&[_]u32{
            // Front face
            0,  1,  2,
            2,  3,  0,

            // Right face
            4,  5,  6,
            6,  7,  4,

            // Back face
            8,  9,  10,
            10, 11, 8,

            // Left face
            12, 13, 14,
            14, 15, 12,

            // Bottom face
            16, 17, 18,
            18, 19, 16,

            // Top face
            20, 21, 22,
            22, 23, 20,
        });

        return .{
            .vertices = vertices,
            .indices = indices,
        };
    }
};

pub const Axis = struct {
    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, position: ?Vec3, length: ?f32) !*Node {
        var vertices: []Vertex = undefined;

        if (position != null and length != null) {
            vertices = try generateVertices(
                allocator,
                position.?,
                length.?,
            );
        } else {
            vertices = try generateVertices(
                allocator,
                .{ .x = 0.0, .y = 0.0, .z = 0.0 },
                5.0,
            );
        }

        var mesh = try Mesh.init(vertices, null, Self.draw);
        mesh.drawType = glad.GL_LINES;

        const node = try Node.init(allocator, mesh);

        return node;
    }

    pub fn draw(mesh: *Mesh) void {

        // Configure depth testing
        glad.glEnable(glad.GL_DEPTH_TEST);
        glad.glDepthFunc(glad.GL_LEQUAL);

        // Enable line smoothing
        glad.glEnable(glad.GL_LINE_SMOOTH);
        glad.glHint(glad.GL_LINE_SMOOTH_HINT, glad.GL_NICEST);

        // Enable blending for smooth lines
        glad.glEnable(glad.GL_BLEND);
        glad.glBlendFunc(glad.GL_SRC_ALPHA, glad.GL_ONE_MINUS_SRC_ALPHA);

        // Set line width
        glad.glLineWidth(1.0);

        // Small offset to prevent z-fighting
        glad.glEnable(glad.GL_POLYGON_OFFSET_LINE);
        glad.glPolygonOffset(-1.0, -1.0);

        glad.glBindVertexArray(mesh.meta.VAO);
        glad.glBindBuffer(glad.GL_ARRAY_BUFFER, mesh.meta.VBO);

        // Position attribute (location = 0)
        glad.glVertexAttribPointer(
            0, // location
            3, //(vec3)
            glad.GL_FLOAT,
            glad.GL_FALSE,
            @sizeOf(Vertex), // stride (size of entire vertex struct)
            null, //
        );
        glad.glEnableVertexAttribArray(0);

        // Color attribute (location = 1)
        const color_offset = @offsetOf(Vertex, "color");
        glad.glVertexAttribPointer(
            1, // location
            3, // (vec3)
            glad.GL_FLOAT,
            glad.GL_FALSE,
            @sizeOf(Vertex), // stride (size of entire vertex struct)
            @ptrFromInt(color_offset),
        );
        glad.glEnableVertexAttribArray(1);
        glad.glDrawArrays(glad.GL_LINES, 0, @intCast(mesh.vertices.len));

        // Disable vertex attributes
        glad.glDisableVertexAttribArray(0);
        glad.glDisableVertexAttribArray(1);

        glad.glBindVertexArray(0);

        // Reset states
        glad.glDisable(glad.GL_POLYGON_OFFSET_LINE);
        glad.glDisable(glad.GL_BLEND);
        glad.glDisable(glad.GL_LINE_SMOOTH);
    }

    pub fn generateVertices(allocator: std.mem.Allocator, position: Vec3, length: f32) ![]Vertex {
        const vertex_count: usize = @intFromFloat(3 * 2);
        var vertices: []Vertex = try allocator.alloc(Vertex, vertex_count);
        var index: usize = 0;

        while (index < vertex_count) {
            var color: [3]f32 = undefined;

            // check whether current loop is plotting x, y, or z and color appropriately
            switch (index % 3) {
                0 => {
                    color = .{ 1.0, 0.0, 0.0 };
                },
                1 => {
                    color = .{ 0.0, 1.0, 0.0 };
                },
                2 => {
                    color = .{ 0.0, 0.0, 1.0 };
                },
                else => unreachable,
            }

            vertices[index] = Vertex{
                .position = .{
                    position.x,
                    position.y,
                    position.z,
                },
                .color = color,
            };
            index += 1;

            switch ((index - 1) % 3) {
                0 => vertices[index] = Vertex{
                    .position = .{
                        position.x + length,
                        position.y,
                        position.z,
                    },
                    .color = color,
                },
                1 => vertices[index] = Vertex{
                    .position = .{
                        position.x,
                        position.y + length,
                        position.z,
                    },
                    .color = color,
                },
                2 => vertices[index] = Vertex{
                    .position = .{
                        position.x,
                        position.y,
                        position.z + length,
                    },
                    .color = color,
                },
                else => unreachable,
            }
            index += 1;
        }

        return vertices;
    }
};

pub const Grid = struct {
    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, gridSize: ?usize, spacing: ?f32) !*Node {
        // Generate grid vertices
        var vertices: []Vertex = undefined;

        if (gridSize != null and spacing != null) {
            vertices = try Grid.generateGridVertices(allocator, gridSize.?, spacing.?);
        } else {
            vertices = try Grid.generateGridVertices(allocator, 1000, 1.0);
        }

        var mesh: Mesh = try Mesh.init(vertices, null, Self.draw);
        mesh.drawType = glad.GL_LINES;

        const node = try Node.init(allocator, mesh);
        return node;
    }

    pub fn draw(mesh: *Mesh) void {

        // Configure depth testing
        glad.glEnable(glad.GL_DEPTH_TEST);
        glad.glDepthFunc(glad.GL_LEQUAL);

        // Enable line smoothing
        glad.glEnable(glad.GL_LINE_SMOOTH);
        glad.glHint(glad.GL_LINE_SMOOTH_HINT, glad.GL_NICEST);

        // Enable blending for smooth lines
        glad.glEnable(glad.GL_BLEND);
        glad.glBlendFunc(glad.GL_SRC_ALPHA, glad.GL_ONE_MINUS_SRC_ALPHA);

        // Set line width
        glad.glLineWidth(1.0);

        // Small offset to prevent z-fighting
        glad.glEnable(glad.GL_POLYGON_OFFSET_LINE);
        glad.glPolygonOffset(-1.0, -1.0);

        glad.glBindVertexArray(mesh.meta.VAO);
        glad.glBindBuffer(glad.GL_ARRAY_BUFFER, mesh.meta.VBO);

        // Position attribute (location = 0)
        glad.glVertexAttribPointer(
            0, // location
            3, // (vec3)
            glad.GL_FLOAT,
            glad.GL_FALSE,
            @sizeOf(Vertex), // stride (size of entire vertex struct)
            null,
        );
        glad.glEnableVertexAttribArray(0);

        // Color attribute (location = 1)
        const color_offset = @offsetOf(Vertex, "color");
        glad.glVertexAttribPointer(
            1, // location
            3, //(vec3)
            glad.GL_FLOAT,
            glad.GL_FALSE,
            @sizeOf(Vertex), // stride (size of entire vertex struct)
            @ptrFromInt(color_offset),
        );

        glad.glEnableVertexAttribArray(1);
        glad.glDrawArrays(glad.GL_LINES, 0, @intCast(mesh.vertices.len));

        // Disable vertex attributes
        glad.glDisableVertexAttribArray(0);
        glad.glDisableVertexAttribArray(1);

        glad.glBindVertexArray(0);

        // Reset states
        glad.glDisable(glad.GL_POLYGON_OFFSET_LINE);
        glad.glDisable(glad.GL_BLEND);
        glad.glDisable(glad.GL_LINE_SMOOTH);
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
                .position = .{
                    -halfSize * 0.05,
                    y_coord,
                    i * 0.05,
                },
                .color = xColor,
            };
            index += 1;

            vertices[index] = Vertex{
                .position = .{
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
                .position = .{
                    i * 0.05,
                    y_coord,
                    -halfSize * 0.05,
                },
                .color = zColor,
            };
            index += 1;

            vertices[index] = Vertex{
                .position = .{
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

pub const TexturedPlane = struct {
    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, pos: ?Vec3, width: ?f32, length: ?f32) !*Node {
        var vertices: []Vertex = undefined;
        var indices: []u32 = undefined;

        if (pos != null and width != null and length != null) {
            const result = try Self.generatePlaneVertices(allocator, pos.?, width.?, length.?);
            vertices = result.vertices;
            indices = result.indices;
        } else {
            const result = try Self.generatePlaneVertices(allocator, Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 }, 1.0, 1.0);
            vertices = result.vertices;
            indices = result.indices;
        }

        // For textured rendering, you'll want to add texture coordinate generation
        // This is a placeholder for your YUV444P texture mapping
        const mesh = try Mesh.init(vertices, indices, Self.draw);
        const node = try Node.init(allocator, mesh);

        return node;
    }

    pub fn generatePlaneVertices(allocator: std.mem.Allocator, position: Vec3, width: f32, length: f32) !struct { vertices: []Vertex, indices: []u32 } {
        const halfWidth: f32 = width / 2.0;
        const halflength: f32 = length / 2.0;

        var vertices: []Vertex = try allocator.alloc(Vertex, 4);
        vertices[0] = .{
            .position = .{ position.x - halfWidth, position.y, position.z - halflength },
            .color = .{ 0.8, 0.8, 0.8 },
            .texture = [_]f32{ 0.0, 1.0 },
        };
        vertices[1] = .{
            .position = .{ position.x + halfWidth, position.y, position.z - halflength },
            .color = .{ 0.8, 0.8, 0.8 },
            .texture = [_]f32{ 1.0, 1.0 },
        };
        vertices[2] = .{
            .position = .{ position.x + halfWidth, position.y, position.z + halflength },
            .color = .{ 0.8, 0.8, 0.8 },
            .texture = [_]f32{ 1.0, 0.0 },
        };
        vertices[3] = .{
            .position = .{ position.x - halfWidth, position.y, position.z + halflength },
            .color = .{ 0.8, 0.8, 0.8 },
            .texture = [_]f32{ 0.0, 0.0 },
        };

        var indices: []u32 = try allocator.alloc(u32, 6);
        indices = @constCast(&[_]u32{
            0, 1, 2, // First triangle
            2, 3, 0, // Second triangle
        });

        return .{
            .vertices = vertices,
            .indices = indices,
        };
    }

    pub fn draw(mesh: *Mesh) void {
        // Configure depth testing
        glad.glEnable(glad.GL_DEPTH_TEST);
        glad.glDepthFunc(glad.GL_LEQUAL);

        glad.glBindVertexArray(mesh.meta.VAO);
        glad.glBindBuffer(glad.GL_ARRAY_BUFFER, mesh.meta.VBO);

        if (mesh.node) |node| {
            if (node.y != null and node.uv != null) {
                if (node.scene) |scene| {
                    glad.glUniform1i(scene.useTextureLoc, @as(c_int, 1));
                }
            }
        }

        // Position attribute (location = 0)
        glad.glVertexAttribPointer(
            0, // location
            3, // (vec3)
            glad.GL_FLOAT,
            glad.GL_FALSE,
            @sizeOf(Vertex), // stride
            null,
        );
        glad.glEnableVertexAttribArray(0);

        // Color attribute (location = 1)
        const color_offset = @offsetOf(Vertex, "color");
        glad.glVertexAttribPointer(
            1, // location
            3, // (vec3)
            glad.GL_FLOAT,
            glad.GL_FALSE,
            @sizeOf(Vertex), // stride
            @ptrFromInt(color_offset),
        );
        glad.glEnableVertexAttribArray(1);

        // Texture coordinate attribute (location = 2)
        const tex_coord_offset = @offsetOf(Vertex, "texture");

        glad.glVertexAttribPointer(
            2, // location
            2, // (vec2)
            glad.GL_FLOAT,
            glad.GL_FALSE,
            @sizeOf(Vertex), // stride
            @ptrFromInt(tex_coord_offset),
        );
        glad.glEnableVertexAttribArray(2);

        // Draw the plane
        glad.glDrawElements(glad.GL_TRIANGLES, @intCast(mesh.indices.?.len), glad.GL_UNSIGNED_INT, null);

        if (mesh.node) |node| {
            if (node.scene) |scene| {
                glad.glUniform1i(scene.useTextureLoc, @as(c_int, 0));
            }
        }

        // Disable vertex attributes
        glad.glDisableVertexAttribArray(0);
        glad.glDisableVertexAttribArray(1);
        glad.glDisableVertexAttribArray(2);

        glad.glBindVertexArray(0);
    }
};
