const std = @import("std");

pub const KeyPoint = extern struct {
    x: f32,
    y: f32,
};

extern "cuda_keypoint_detector" fn cuda_init_detector(max_width: c_int, max_height: c_int) c_int;
extern "cuda_keypoint_detector" fn cuda_detect_keypoints(
    y_plane: [*]const u8,
    uv_plane: [*]const u8,
    width: c_int,
    height: c_int,
    y_linesize: c_int,
    uv_linesize: c_int,
    threshold: u8,
    keypoints: [*]KeyPoint,
    max_keypoints: c_int,
    num_keypoints: *c_int,
) c_int;
extern "cuda_keypoint_detector" fn cuda_cleanup_detector() void;

pub const CudaKeypointDetector = struct {
    const Self = @This();

    pub fn init(max_width: c_int, max_height: c_int) !void {
        if (cuda_init_detector(max_width, max_height) != 0) {
            return error.CudaInitFailed;
        }
    }

    pub fn detectKeypoints(
        y_plane: [*c]u8,
        uv_plane: [*c]u8,
        width: u32,
        height: u32,
        y_linesize: u32,
        uv_linesize: u32,
        threshold: u8,
        keypoints: []KeyPoint,
    ) !usize {
        var num_keypoints: c_int = 0;

        std.debug.print(
            "\n\nCalling CUDA detector with dims: {}x{}, linestride: {}\n",
            .{ width, height, y_linesize },
        );

        const result = cuda_detect_keypoints(
            y_plane,
            uv_plane,
            @intCast(width),
            @intCast(height),
            @intCast(y_linesize),
            @intCast(uv_linesize),
            threshold,
            keypoints.ptr,
            @intCast(keypoints.len),
            &num_keypoints,
        );

        if (result != 0) {
            std.debug.print("CUDA keypoint detection failed with error: {}\n", .{result});
            return error.KeypointDetectionFailed;
        }

        std.debug.print("Found {} keypoints\n", .{num_keypoints});
        return @intCast(num_keypoints);
    }

    pub fn deinit() void {
        cuda_cleanup_detector();
    }
};
