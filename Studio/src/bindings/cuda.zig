const std = @import("std");

pub const KeyPoint = extern struct {
    x: f32,
    y: f32,
};

// External CUDA function declarations
extern "cuda_keypoint_detector" fn cuda_init_detector() c_int;
extern "cuda_keypoint_detector" fn cuda_register_gl_buffers(position_buffer: c_uint, color_buffer: c_uint, max_keypoints: c_int) c_int;
extern "cuda_keypoint_detector" fn cuda_detect_keypoints_gl(
    y_plane: [*]const u8,
    uv_plane: [*]const u8,
    width: c_int,
    height: c_int,
    y_linesize: c_int,
    uv_linesize: c_int,
    threshold: u8,
    num_keypoints: *c_int,
    image_width: f32,
    image_height: f32,
) c_int;
extern "cuda_keypoint_detector" fn cuda_unregister_gl_buffers() void;
extern "cuda_keypoint_detector" fn cuda_cleanup_detector() void;

pub const CudaKeypointDetector = struct {
    const Self = @This();

    // Track whether we're using GL interop
    gl_interop_enabled: bool,

    pub fn init() !Self {
        if (cuda_init_detector() != 0) {
            return error.CudaInitFailed;
        }
        return Self{
            .gl_interop_enabled = false,
        };
    }

    pub fn enableGLInterop(self: *Self, position_buffer: u32, color_buffer: u32, max_keypoints: u32) !void {
        if (cuda_register_gl_buffers(
            position_buffer,
            color_buffer,
            @intCast(max_keypoints),
        ) != 0) {
            return error.CudaGLRegistrationFailed;
        }
        self.gl_interop_enabled = true;
    }

    pub fn detectKeypoints(
        self: *Self,
        y_plane: [*c]u8,
        uv_plane: [*c]u8,
        width: u32,
        height: u32,
        y_linesize: u32,
        uv_linesize: u32,
        threshold: u8,
    ) !usize {
        var num_keypoints: c_int = 0;
        _ = self;

        std.debug.print(
            "\n\nCalling CUDA detector with dims: {}x{}, linestride: {}\n",
            .{ width, height, y_linesize },
        );

        // Use GL interop path
        const result = cuda_detect_keypoints_gl(
            y_plane,
            uv_plane,
            @intCast(width),
            @intCast(height),
            @intCast(y_linesize),
            @intCast(uv_linesize),
            threshold,
            &num_keypoints,
            @floatFromInt(width),
            @floatFromInt(height),
        );
        if (result != 0) {
            std.debug.print("CUDA-GL keypoint detection failed with error: {}\n", .{result});
            return error.KeypointDetectionFailed;
        }

        std.debug.print("Found {} keypoints\n", .{num_keypoints});
        return @intCast(num_keypoints);
    }

    pub fn disableGLInterop(self: *Self) void {
        if (self.gl_interop_enabled) {
            cuda_unregister_gl_buffers();
            self.gl_interop_enabled = false;
        }
    }

    pub fn deinit(self: *Self) void {
        if (self.gl_interop_enabled) {
            self.disableGLInterop();
        }
        cuda_cleanup_detector();
    }
};
