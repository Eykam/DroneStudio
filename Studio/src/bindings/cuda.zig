const std = @import("std");
const Node = @import("../Node.zig").InstanceData;
const ORB = @import("../ORB.zig");
const gl = @import("gl.zig");
const glad = gl.glad;

pub const KeyPoint = extern struct {
    x: f32,
    y: f32,
};

pub const ImageParams = extern struct {
    y_plane: [*c]u8,
    uv_plane: [*c]u8,
    width: c_int,
    height: c_int,
    y_linesize: c_int,
    uv_linesize: c_int,
    num_keypoints: *c_int,
    image_width: f32,
    image_height: f32,
};

// External CUDA function declarations
extern "cuda_keypoint_detector" fn cuda_create_detector(
    max_keypoints: c_int,
    gl_ytexture: c_uint,
    gl_uvtexture: c_uint,
    gl_depthtexture: c_uint,
) c_int;
extern "cuda_keypoint_detector" fn cuda_cleanup_detector(detector_id: c_int) void;

extern "cuda_keypoint_detector" fn cuda_register_gl_texture(detector_id: c_int) c_int;
extern "cuda_keypoint_detector" fn cuda_unregister_gl_texture() void;
extern "cuda_keypoint_detector" fn cuda_register_buffers(
    detector_id: c_int,
    keypoint_position_buffer: c_uint,
    keypoint_color_buffer: c_uint,
    left_position_buffer: [*c]c_uint,
    left_color_buffer: [*c]c_uint,
    right_position_buffer: [*c]c_uint,
    right_color_buffer: [*c]c_uint,
    buffer_size: c_uint,
) c_int;
extern "cuda_keypoint_detector" fn cuda_unregister_buffers(detector_id: c_int) void;

extern "cuda_keypoint_detector" fn cuda_map_resources(detector_id: c_int) c_int;
extern "cuda_keypoint_detector" fn cuda_unmap_resources(detector_id: c_int) void;

extern "cuda_keypoint_detector" fn cuda_map_transformation(detector_id: c_int, transformation: *const [16]f32) c_int;

extern "cuda_keypoint_detector" fn cuda_detect_keypoints(
    detector_id: c_int,
    threshold: u8,
    image: *ImageParams,
    sigma: f32,
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

pub const CudaKeypointDetector = struct {
    const Self = @This();

    detector_id: c_int,
    gl_interop_enabled: bool,

    pub fn init(
        max_keypoints: u32,
        gl_ytexture: glad.GLuint,
        gl_uvtexture: glad.GLuint,
        gl_depthtexture: glad.GLuint,
    ) !Self {
        const id = cuda_create_detector(
            @intCast(max_keypoints),
            gl_ytexture,
            gl_uvtexture,
            gl_depthtexture,
        );

        if (id < 0) {
            return error.CudaInitFailed;
        }

        return Self{
            .detector_id = id,
            .gl_interop_enabled = false,
        };
    }

    pub fn enableGlInterop(
        self: *Self,
        keypoint_position_buffer: u32,
        keypoint_color_buffer: u32,
        left_position_buffer: ?u32,
        left_color_buffer: ?u32,
        right_position_buffer: ?u32,
        right_color_buffer: ?u32,
        buffer_size: u32,
    ) !void {
        // Create null pointers by default
        var left_pos: c_uint = undefined;
        var left_col: c_uint = undefined;
        var right_pos: c_uint = undefined;
        var right_col: c_uint = undefined;

        // Only pass non-null pointers if both position and color buffers are provided
        const left_pos_ptr: [*c]c_uint = if (left_position_buffer != null and left_color_buffer != null) blk: {
            left_pos = @intCast(left_position_buffer.?);
            break :blk &left_pos;
        } else null;

        const left_col_ptr: [*c]c_uint = if (left_position_buffer != null and left_color_buffer != null) blk: {
            left_col = @intCast(left_color_buffer.?);
            break :blk &left_col;
        } else null;

        const right_pos_ptr: [*c]c_uint = if (right_position_buffer != null and right_color_buffer != null) blk: {
            right_pos = @intCast(right_position_buffer.?);
            break :blk &right_pos;
        } else null;

        const right_col_ptr: [*c]c_uint = if (right_position_buffer != null and right_color_buffer != null) blk: {
            right_col = @intCast(right_color_buffer.?);
            break :blk &right_col;
        } else null;

        if (cuda_register_buffers(
            self.detector_id,
            keypoint_position_buffer,
            keypoint_color_buffer,
            left_pos_ptr,
            left_col_ptr,
            right_pos_ptr,
            right_col_ptr,
            @intCast(buffer_size),
        ) != 0) {
            std.debug.print("Failed to Enable GL Interop!\n", .{});
            return error.CudaGLBufferRegistrationFailed;
        }

        std.debug.print("Setting interop to true!\n", .{});
        self.gl_interop_enabled = true;
    }

    pub fn updateDetectorTransformation(self: *Self, transformation: [16]f32) !void {
        // std.debug.print("World Transform: {any}\n", .{transformation});
        const err = cuda_map_transformation(self.detector_id, &transformation);

        if (err < 0) {
            std.debug.print("Failed to update Detectors Transformation matrix => {d}\n", .{self.detector_id});
            return error.DetectorTransformationUpdateFailed;
        }

        return;
    }

    pub fn deinit(self: *Self) void {
        self.gl_interop_enabled = false;
        cuda_cleanup_detector(self.detector_id);
    }

    pub fn detect_keypoints(
        self: *Self,
        threshold: u8,
        image: *ImageParams,
    ) !void {
        if (!self.gl_interop_enabled) @panic("GL Resources are not registered!\n");

        std.debug.print(
            "Calling CUDA detector with dims: {}x{}, linestride: {}\n",
            .{ image.width, image.height, image.y_linesize },
        );

        if (cuda_map_resources(self.detector_id) < 0) {
            return;
        }
        defer cuda_unmap_resources(self.detector_id);

        // Use GL interop path
        const result = cuda_detect_keypoints(
            self.detector_id,
            threshold,
            image,
            1.25,
        );

        if (result < 0) {
            std.debug.print("CUDA-GL keypoint detection failed with error: {}\n", .{result});
            return error.KeypointDetectionFailed;
        }

        std.debug.print("Keypoint Detection Exection Time: {d:.5}\n", .{result});
        std.debug.print("Found {} keypoints\n", .{image.num_keypoints.*});
    }

    pub fn match_keypoints(
        detector_id_left: c_int,
        detector_id_right: c_int,
        detector_id_combined: c_int,
        params: *ORB.MatchingParameters,
        num_matches: *c_int,
        left: *ImageParams,
        right: *ImageParams,
    ) !void {
        const result = cuda_match_keypoints(
            detector_id_left,
            detector_id_right,
            detector_id_combined,
            params.baseline_mm,
            params.focal_length_mm,
            num_matches,
            params.intensity_threshold,
            left,
            right,
        );

        if (result < 0) {
            std.debug.print("CUDA-GL keypoint matcher failed with error: {}\n", .{result});
            return error.KeypointMatchFailed;
        }
    }
};
