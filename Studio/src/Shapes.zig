pub const Rectangle = struct {
    const Self = @This();

    vertices: []f32 = .{
        // Positions
        // Front face
        -0.5, -0.5, 0.5, // Bottom Left
        0.5, -0.5, 0.5, // Bottom Right
        0.5, 0.5, 0.5, // Top Right
        -0.5, 0.5, 0.5, // Top Left

        // Back face
        -0.5, -0.5, -0.5, // Bottom Left
        0.5, -0.5, -0.5, // Bottom Right
        0.5, 0.5, -0.5, // Top Right
        -0.5, 0.5, -0.5, // Top Left
    },

    indices: []u32 = .{
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

    pub fn init(self: Self) !void {
        _ = self;
    }
};
