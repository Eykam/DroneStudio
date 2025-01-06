const std = @import("std");
// const kernels = @cImport({
//     @cInclude("keypoint_detector.h");
// });

pub const KeyPoint = extern struct {
    x: f32,
    y: f32,
};

pub const ImageParams = extern struct {
    y_plane: [*c]u8,
    width: c_int,
    height: c_int,
    y_linesize: c_int,
    num_keypoints: *c_int,
    image_width: f32,
    image_height: f32,
};

// External CUDA function declarations
extern "cuda_keypoint_detector" fn cuda_create_detector(max_keypoints: c_int) c_int;
extern "cuda_keypoint_detector" fn cuda_register_gl_buffers(
    detector_id: c_int,
    position_buffer: c_uint,
    color_buffer: c_uint,
    max_keypoints: c_int,
) c_int;
extern "cuda_keypoint_detector" fn cuda_detect_keypoints(
    detector_id: c_int,
    threshold: u8,
    image: *ImageParams,
) f32;
extern "cuda_keypoint_detector" fn cuda_match_keypoints(
    detector_id_left: c_int,
    detector_id_right: c_int,
    detector_id_combined: c_int,
    baseline: f32,
    focal_length: f32,
    num_matches: *c_int,
    threshold: u8,
    left: *ImageParams,
    right: *ImageParams,
) c_int;
extern "cuda_keypoint_detector" fn cuda_unregister_gl_buffers(detector_id: c_int) void;
extern "cuda_keypoint_detector" fn cuda_cleanup_detector(detector_id: c_int) void;
extern "cuda_keypoint_detector" fn cuda_map_gl_resources(detector_id: c_int) c_int;
extern "cuda_keypoint_detector" fn cuda_unmap_gl_resources(detector_id: c_int) void;

pub const CudaKeypointDetector = struct {
    const Self = @This();

    // Track whether we're using GL interop
    detector_id: c_int,
    gl_interop_enabled: bool,

    pub fn init(max_keypoints: u32) !Self {
        const id = cuda_create_detector(@intCast(max_keypoints));
        if (id < 0) {
            return error.CudaInitFailed;
        }

        return Self{
            .detector_id = id,
            .gl_interop_enabled = false,
        };
    }

    pub fn enableGLInterop(self: *Self, position_buffer: u32, color_buffer: u32, max_keypoints: u32) !void {
        if (cuda_register_gl_buffers(
            self.detector_id,
            position_buffer,
            color_buffer,
            @intCast(max_keypoints),
        ) != 0) {
            return error.CudaGLRegistrationFailed;
        }
        self.gl_interop_enabled = true;
    }

    pub fn disableGLInterop(self: *Self) void {
        if (self.gl_interop_enabled) {
            cuda_unregister_gl_buffers(self.detector_id);
            self.gl_interop_enabled = false;
        }
    }

    pub fn deinit(self: *Self) void {
        self.disableGLInterop();
        cuda_cleanup_detector(self.detector_id);
    }

    pub fn detect_keypoints(
        self: *Self,
        threshold: u8,
        image: *ImageParams,
    ) !void {
        std.debug.print(
            "\n\nCalling CUDA detector with dims: {}x{}, linestride: {}\n",
            .{ image.width, image.height, image.y_linesize },
        );

        if (cuda_map_gl_resources(self.detector_id) < 0) {
            return;
        }
        defer cuda_unmap_gl_resources(self.detector_id);

        // Use GL interop path
        const result = cuda_detect_keypoints(
            self.detector_id,
            threshold,
            image,
        );

        if (result < 0) {
            std.debug.print("CUDA-GL keypoint detection failed with error: {}\n", .{result});
            return error.KeypointDetectionFailed;
        }

        std.debug.print("Keypoint Detection Exection TIMe: {d:.5}\n", .{result});
        std.debug.print("Found {} keypoints\n", .{image.num_keypoints.*});
    }

    pub fn match_keypoints(
        detector_id_left: c_int,
        detector_id_right: c_int,
        detector_id_combined: c_int,
        baseline: f32,
        focal_length: f32,
        num_matches: *c_int,
        threshold: u8,
        left: *ImageParams,
        right: *ImageParams,
    ) !void {
        const result = cuda_match_keypoints(
            detector_id_left,
            detector_id_right,
            detector_id_combined,
            baseline,
            focal_length,
            num_matches,
            threshold,
            left,
            right,
        );

        if (result < 0) {
            std.debug.print("CUDA-GL keypoint matcher failed with error: {}\n", .{result});
            return error.KeypointMatchFailed;
        }

        std.debug.print("Result of cuda_match_keypoints => {}\n", .{result});
    }
};
