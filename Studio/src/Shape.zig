// src/Shapes.zig
const std = @import("std");
const Math = @import("Math.zig");
const Mesh = @import("Mesh.zig");
const Node = @import("Node.zig");
const gl = @import("bindings/gl.zig");
const Vertex = Mesh.Vertex;
const Vec3 = Math.Vec3;
const glad = gl.glad;

const glCheckError = @import("Debug.zig").glCheckError;

const DrawErrors = error{FailedToDraw};

pub const Triangle = struct {
    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, position: ?Vec3, _vertices: ?[]Vertex) !*Node {
        const vertices: []Vertex = try allocator.dupe(Vertex, _vertices orelse &Self.default(position));

        const draw = Mesh.gen_draw(comptime glad.GL_TRIANGLES);
        const node = try Node.init(allocator, vertices, null, draw);
        return node;
    }

    pub inline fn default(origin: ?Vec3) [3]Vertex {
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

        const vertices = [_]Vertex{
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

        return vertices;
    }
};

pub const Box = struct {
    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, pos: ?Vec3, height: ?f32, width: ?f32, depth: ?f32) !*Node {
        _ = pos;
        _ = height;
        _ = width;
        _ = depth;

        const defaults = try Self.default();
        const vertices = try allocator.dupe(Vertex, &defaults.vertices);
        const indices = try allocator.dupe(u32, &defaults.indices);

        const draw = Mesh.gen_draw(comptime glad.GL_TRIANGLES);
        const node = try Node.init(allocator, vertices, indices, draw);
        return node;
    }

    pub fn default() !struct { vertices: [4 * 6]Vertex, indices: [6 * 2 * 3]u32 } {
        // (4 vertices per face * 6 faces)

        const front_color = [3]f32{ 0.9, 0.9, 0.9 }; // Lightest gray (front)
        const right_color = [3]f32{ 0.8, 0.8, 0.8 }; // Slightly darker
        const back_color = [3]f32{ 0.6, 0.6, 0.6 }; // Darkest (back)
        const left_color = [3]f32{ 0.75, 0.75, 0.75 }; // Medium-dark
        const bottom_color = [3]f32{ 0.7, 0.7, 0.7 }; // Darker for bottom
        const top_color = [3]f32{ 0.85, 0.85, 0.85 }; // Lighter for top

        const vertices = [_]Vertex{
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
        };

        // 6 faces * 2 triangles * 3 vertices
        const indices =
            [_]u32{
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
        };

        return .{
            .vertices = vertices,
            .indices = indices,
        };
    }
};

pub const Axis = struct {
    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, position: ?Vec3, length: ?f32) !*Node {
        const default_pos: Vec3 = .{ .x = 0.0, .y = 0.0, .z = 0.0 };
        const vertices: []Vertex = try generateVertices(
            allocator,
            position orelse default_pos,
            length orelse 5.0,
        );

        var node = try Node.init(allocator, vertices, null, Self.draw);
        node.mesh.?.drawType = glad.GL_LINES;

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
        const vertices: []Vertex = try Grid.generateGridVertices(
            allocator,
            gridSize orelse 1000,
            spacing orelse 1.0,
        );

        var node = try Node.init(allocator, vertices, null, Self.draw);
        node.mesh.?.drawType = glad.GL_LINES;

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
        const default_pos = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
        const plane_params = try Self.generatePlaneVertices(
            pos orelse default_pos,
            width orelse 1.0,
            length orelse 1.0,
        );
        const vertices: []Vertex = try allocator.dupe(Vertex, &plane_params.vertices);
        const indices: []u32 = try allocator.dupe(u32, &plane_params.indices);

        const node = try Node.init(allocator, vertices, indices, Self.draw);

        return node;
    }

    pub fn generatePlaneVertices(position: Vec3, width: f32, length: f32) !struct { vertices: [4]Vertex, indices: [6]u32 } {
        const halfWidth: f32 = width / 2.0;
        const halflength: f32 = length / 2.0;

        const vertices: [4]Vertex = .{
            .{
                .position = .{ position.x - halfWidth, position.y, position.z - halflength },
                .color = .{ 0.8, 0.8, 0.8 },
                .texture = [_]f32{ 0.0, 1.0 },
            },
            .{
                .position = .{ position.x + halfWidth, position.y, position.z - halflength },
                .color = .{ 0.8, 0.8, 0.8 },
                .texture = [_]f32{ 1.0, 1.0 },
            },
            .{
                .position = .{ position.x + halfWidth, position.y, position.z + halflength },
                .color = .{ 0.8, 0.8, 0.8 },
                .texture = [_]f32{ 1.0, 0.0 },
            },
            .{
                .position = .{ position.x - halfWidth, position.y, position.z + halflength },
                .color = .{ 0.8, 0.8, 0.8 },
                .texture = [_]f32{ 0.0, 0.0 },
            },
        };

        const indices =
            [6]u32{
            0, 1, 2, // First triangle
            2, 3, 0, // Second triangle
        };

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

pub const KeypointDebugger = struct {
    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, pos: [3]f32, radius: ?f32, resolution: ?u32) !*Node {
        const vertices = try Self.generateVertices(
            allocator,
            pos,
            radius orelse 0.05,
            @max(1, resolution orelse 1),
        );

        const node = try Node.init(allocator, vertices, null, Self.draw);
        return node;
    }

    pub fn draw(mesh: *Mesh) void {
        // Debug check mesh state
        if (mesh.vertices.len == 0 or mesh.vertices.len > 10000) { // reasonable maximum
            std.debug.print("Invalid vertex count: {}\n", .{mesh.vertices.len});
            return;
        }

        // std.debug.print("Drawing keypoint with {} vertices\n", .{mesh.vertices.len});

        // OpenGL state setup
        glad.glEnable(glad.GL_DEPTH_TEST);
        glad.glDepthFunc(glad.GL_LEQUAL);
        glad.glEnable(glad.GL_LINE_SMOOTH);
        glad.glHint(glad.GL_LINE_SMOOTH_HINT, glad.GL_NICEST);
        glad.glEnable(glad.GL_BLEND);
        glad.glBlendFunc(glad.GL_SRC_ALPHA, glad.GL_ONE_MINUS_SRC_ALPHA);
        glad.glLineWidth(3.0);
        glad.glEnable(glad.GL_POLYGON_OFFSET_LINE);
        glad.glPolygonOffset(-1.0, -1.0);

        // Verify VAO and VBO
        if (mesh.meta.VAO == 0 or mesh.meta.VBO == 0) {
            std.debug.print("Error: Invalid VAO or VBO\n", .{});
            return;
        }

        // Bind buffers
        glad.glBindVertexArray(mesh.meta.VAO);
        glad.glBindBuffer(glad.GL_ARRAY_BUFFER, mesh.meta.VBO);

        // Debug check buffer binding
        // const current_vao = glad.glGetvb(glad.GL_VERTEX_ARRAY_BINDING);
        // const current_vbo = glad.glGetInteger(glad.GL_ARRAY_BUFFER_BINDING);
        // std.debug.print("Current VAO: {}, Current VBO: {}\n", .{ current_vao, current_vbo });

        // Setup vertex attributes with error checking
        glad.glVertexAttribPointer(0, 3, glad.GL_FLOAT, glad.GL_FALSE, @sizeOf(Vertex), null);
        glad.glEnableVertexAttribArray(0);

        const color_offset = @offsetOf(Vertex, "color");
        glad.glVertexAttribPointer(1, 3, glad.GL_FLOAT, glad.GL_FALSE, @sizeOf(Vertex), @ptrFromInt(color_offset));
        glad.glEnableVertexAttribArray(1);

        // Draw with explicit vertex count
        const vertex_count: c_int = @intCast(mesh.vertices.len);
        // std.debug.print("Drawing with {} vertices\n", .{vertex_count});
        glad.glDrawArrays(glad.GL_LINE_LOOP, 0, vertex_count);

        // Check for OpenGL errors
        const err = glad.glGetError();
        if (err != glad.GL_NO_ERROR) {
            std.debug.print("OpenGL error in draw: 0x{x}\n", .{err});
        }

        // Cleanup
        glad.glDisableVertexAttribArray(0);
        glad.glDisableVertexAttribArray(1);
        glad.glBindVertexArray(0);
        glad.glDisable(glad.GL_POLYGON_OFFSET_LINE);
        glad.glDisable(glad.GL_BLEND);
        glad.glDisable(glad.GL_LINE_SMOOTH);
    }

    pub fn generateVertices(alloc: std.mem.Allocator, pos: [3]f32, radius: f32, resolution: u32) ![]Vertex {
        const num_steps = 360 * resolution;
        const step_size = 360.0 / @as(f32, @floatFromInt(num_steps));

        const vertices = try alloc.alloc(Vertex, num_steps);

        // std.debug.print("Generating {} vertices for keypoint at {d},{d},{d}\n", .{ num_steps, pos[0], pos[1], pos[2] });

        for (1..num_steps + 1) |step| {
            const theta_deg = @as(f32, @floatFromInt(step)) * step_size;
            const theta_rad = theta_deg * (std.math.pi / 180.0);
            vertices[step - 1] = Vertex{
                .position = [_]f32{
                    pos[0] + radius * @sin(theta_rad),
                    pos[1],
                    pos[2] + radius * @cos(theta_rad),
                },
                .color = [_]f32{ 255 / 255, 0 / 255, 0 / 255 },
            };
        }

        return vertices;
    }
};
