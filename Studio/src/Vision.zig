const std = @import("std");
const Node = @import("Node.zig");
const Shape = @import("Shape.zig");
const libav = @import("bindings/libav.zig");
const gl = @import("bindings/gl.zig");
const CudaGL = @import("CudaGL.zig");
const Math = @import("Math.zig");
const cuda = CudaGL.cuda;
const video = libav.video;
const glad = gl.glad;

const KeypointDebugger = Shape.InstancedKeypointDebugger;
const InstancedLine = Shape.InstancedLine;

const DetectionResources = CudaGL.DetectionResources;
const StereoParams = CudaGL.StereoParams;
const ImageParams = CudaGL.ImageParams;

pub const CameraPose = struct {
    rotation: [9]f32, // 3x3 rotation matrix in row-major order
    translation: [3]f32, // 3D translation vector
    pub fn init() CameraPose {
        return .{
            .rotation = .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 },
            .translation = .{ 0, 0, 0 },
        };
    }
    pub fn fromCType(c_pose: cuda.CameraPose) CameraPose {
        var pose = CameraPose.init();

        const conv = [9]f32{
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, -1.0,
        };

        // Convert rotation: R_opengl = conv * R_camera * conv^T
        // First multiply conv * R_camera
        var temp: [9]f32 = undefined;

        for (0..3) |i| {
            for (0..3) |j| {
                var sum: f32 = 0;
                for (0..3) |k| {
                    sum += conv[i * 3 + k] * c_pose.rotation[k * 3 + j];
                }
                temp[i * 3 + j] = sum;
            }
        }

        // Then multiply by conv^T
        for (0..3) |i| {
            for (0..3) |j| {
                var sum: f32 = 0;
                for (0..3) |k| {
                    sum += temp[i * 3 + k] * conv[j * 3 + k]; // Note: conv^T means swapped indices
                }
                pose.rotation[i * 3 + j] = sum;
            }
        }

        // Convert translation
        pose.translation[0] = c_pose.translation[0];
        pose.translation[1] = c_pose.translation[1];
        pose.translation[2] = -c_pose.translation[2]; // Flip Z for OpenGL

        return pose;
    }

    pub fn multiply(self: CameraPose, other: CameraPose) CameraPose {
        var result = CameraPose.init();
        // Multiply rotation matrices
        for (0..3) |i| {
            for (0..3) |j| {
                result.rotation[i * 3 + j] = 0;
                for (0..3) |k| {
                    result.rotation[i * 3 + j] +=
                        self.rotation[i * 3 + k] * other.rotation[k * 3 + j];
                }
            }
        }
        // Transform and add translations
        for (0..3) |i| {
            result.translation[i] = self.translation[i];
            for (0..3) |j| {
                result.translation[i] +=
                    self.rotation[i * 3 + j] * other.translation[j];
            }
        }
        return result;
    }
    // Convert rotation matrix to quaternion
    pub fn toQuaternion(self: CameraPose) Math.Quaternion {
        const m = self.rotation;
        const trace = m[0] + m[4] + m[8];
        if (trace > 0) {
            const s = 0.5 / @sqrt(trace + 1.0);
            return Math.Quaternion{
                .w = 0.25 / s,
                .x = (m[7] - m[5]) * s,
                .y = (m[2] - m[6]) * s,
                .z = (m[3] - m[1]) * s,
            };
        } else {
            if (m[0] > m[4] and m[0] > m[8]) {
                const s = 2.0 * @sqrt(1.0 + m[0] - m[4] - m[8]);
                return Math.Quaternion{
                    .w = (m[7] - m[5]) / s,
                    .x = 0.25 * s,
                    .y = (m[1] + m[3]) / s,
                    .z = (m[2] + m[6]) / s,
                };
            } else if (m[4] > m[8]) {
                const s = 2.0 * @sqrt(1.0 + m[4] - m[0] - m[8]);
                return Math.Quaternion{
                    .w = (m[2] - m[6]) / s,
                    .x = (m[1] + m[3]) / s,
                    .y = 0.25 * s,
                    .z = (m[5] + m[7]) / s,
                };
            } else {
                const s = 2.0 * @sqrt(1.0 + m[8] - m[0] - m[4]);
                return Math.Quaternion{
                    .w = (m[3] - m[1]) / s,
                    .x = (m[2] + m[6]) / s,
                    .y = (m[5] + m[7]) / s,
                    .z = 0.25 * s,
                };
            }
        }
    }
    // Convert translation array to Vec3
    pub fn toVec3(self: CameraPose) Math.Vec3 {
        return Math.Vec3{
            .x = self.translation[0],
            .y = self.translation[1],
            .z = self.translation[2],
        };
    }
};

pub fn TemporalParams() cuda.TemporalParams {
    return cuda.TemporalParams{
        .max_distance = 1.0, // Maximum distance for temporal matching
        .max_pixel_distance = 50.0, // Maximum pixel distance
        .min_confidence = 0.7, // Minimum confidence threshold
        .min_matches = 10, // Minimum required matches
        .ransac_threshold = 0.05, // RANSAC inlier threshold
        .ransac_iterations = 256, // Number of RANSAC iterations
        .spatial_weight = 0.4, // Weight for spatial distance term
        .hamming_weight = 0.4, // Weight for descriptor distance
        .img_weight = 0.2, // Weight for image space distance
        .max_hamming_dist = 1.0,
        .cost_threshold = 0.7,
        .lowes_ratio = 0.8,
    };
}

pub const StereoVO = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    params: *StereoParams,
    params_changed: bool,

    left: *DetectionResourceManager,
    right: *DetectionResourceManager,
    combined: *DetectionResourceManager,
    temporal: *DetectionResourceManager,

    num_matches: *c_uint,
    d_matches: ?*cuda.MatchedKeypoint = null,
    prev_num_matches: c_uint = 0,
    d_prev_matches: ?*cuda.MatchedKeypoint = null,
    temporal_params: *cuda.TemporalParams,

    pub fn init(
        allocator: std.mem.Allocator,
        left_node: *Node,
        right_node: *Node,
        combined_node: *Node,
        temporal_node: *Node,
        params: ?StereoParams,
    ) !*Self {
        var matcher = try allocator.create(Self);

        matcher.allocator = allocator;

        const stereo_params = try allocator.create(StereoParams);
        const sensor_width_mm = 6.45;
        const focal_length_mm = 2.75;
        stereo_params.* = params orelse StereoParams{
            .image_width = left_node.width.?,
            .image_height = left_node.height.?,
            .baseline_mm = 76.3,
            .focal_length_mm = focal_length_mm,
            .focal_length_px = focal_length_mm * (@as(f32, @floatFromInt(left_node.width.?)) / sensor_width_mm),
            .sensor_width_mm = sensor_width_mm,
            .intensity_threshold = 15,
            .circle_radius = 3,
            .arc_length = 9,
            .max_keypoints = 50000,
            .sigma = 1.0,
            .max_disparity = 100,
            .epipolar_threshold = 15,
            .max_hamming_dist = 1.0,
            .cost_threshold = 0.7,
            .lowes_ratio = 0.8,
            .epipolar_weight = 0.3,
            .disparity_weight = 0.2,
            .hamming_dist_weight = 0.5,
            .show_connections = true,
            .disable_matching = false,
            .disable_depth = false,
            .disable_spatial_tracking = true,
        };

        const temporal_params = try allocator.create(cuda.TemporalParams);
        temporal_params.* = TemporalParams();

        matcher.* = Self{
            .allocator = allocator,
            .params = stereo_params,
            .params_changed = false,
            .num_matches = try allocator.create(c_uint),
            .left = undefined,
            .right = undefined,
            .combined = undefined,
            .temporal = undefined,
            .temporal_params = temporal_params,
        };
        matcher.num_matches.* = 0;

        const left_keypoints = try KeypointDebugger.init(
            allocator,
            .{ 0.0, 0.0, 1.0 },
            stereo_params.max_keypoints,
        );
        const right_keypoints = try KeypointDebugger.init(
            allocator,
            .{ 0.0, 0.0, 1.0 },
            stereo_params.max_keypoints,
        );
        const combined_keypoints = try KeypointDebugger.init(
            allocator,
            .{ 0.0, 0.0, 1.0 },
            stereo_params.max_keypoints,
        );

        const temporal_keypoints = try KeypointDebugger.init(
            allocator,
            .{ 1.0, 1.0, 0.0 }, // Yellow color for temporal keypoints
            stereo_params.max_keypoints,
        );

        try left_node.addChild(left_keypoints);
        try right_node.addChild(right_keypoints);
        try combined_node.addChild(combined_keypoints);
        try temporal_node.addChild(temporal_keypoints);

        // Initialize connection lines
        const left_to_combined = try InstancedLine.init(
            matcher.allocator,
            [_]f32{ 1.0, 0.0, 0.0 },
            stereo_params.max_keypoints,
        );
        const right_to_combined = try InstancedLine.init(
            matcher.allocator,
            [_]f32{ 0.0, 0.0, 1.0 },
            stereo_params.max_keypoints,
        );
        // Initialize connection lines between temporal matches
        const temporal_to_combined = try InstancedLine.init(
            matcher.allocator,
            [_]f32{ 0.0, 1.0, 1.0 }, // Cyan color for temporal match lines
            stereo_params.max_keypoints,
        );

        try combined_node.addChild(left_to_combined);
        try combined_node.addChild(right_to_combined);
        try temporal_node.addChild(temporal_to_combined);

        const left_instance = left_keypoints.instance_data.?;
        const right_instance = right_keypoints.instance_data.?;
        const combined_instance = combined_keypoints.instance_data.?;
        const temporal_instance = temporal_keypoints.instance_data.?;

        const left_line = left_to_combined.instance_data.?;
        const right_line = right_to_combined.instance_data.?;
        const temporal_line = temporal_to_combined.instance_data.?;

        matcher.left = try DetectionResourceManager.init(
            allocator,
            left_node,
            0,
            stereo_params.max_keypoints,
            .{
                .position = left_instance.position_buffer,
                .color = left_instance.color_buffer,
                .size = stereo_params.max_keypoints,
            },
            null,
            null,
        );

        matcher.right = try DetectionResourceManager.init(
            allocator,
            right_node,
            1,
            stereo_params.max_keypoints,
            .{
                .position = right_instance.position_buffer,
                .color = right_instance.color_buffer,
                .size = stereo_params.max_keypoints,
            },
            null,
            null,
        );

        matcher.combined = try DetectionResourceManager.init(
            allocator,
            combined_node,
            2,
            stereo_params.max_keypoints,
            .{
                .position = combined_instance.position_buffer,
                .color = combined_instance.color_buffer,
                .size = stereo_params.max_keypoints,
            },
            .{
                .position = left_line.position_buffer,
                .color = left_line.color_buffer,
                .size = stereo_params.max_keypoints,
            },
            .{
                .position = right_line.position_buffer,
                .color = right_line.color_buffer,
                .size = stereo_params.max_keypoints,
            },
        );

        matcher.temporal = try DetectionResourceManager.init(
            allocator,
            temporal_node,
            3,
            stereo_params.max_keypoints,
            .{
                .position = temporal_instance.position_buffer,
                .color = temporal_instance.color_buffer,
                .size = stereo_params.max_keypoints,
            },
            .{
                .position = temporal_line.position_buffer,
                .color = temporal_line.color_buffer,
                .size = stereo_params.max_keypoints,
            },
            null,
        );

        return matcher;
    }

    pub fn deinit(self: *Self) void {
        if (self.d_matches) |matches| {
            _ = cuda.cudaFree(matches);
        }
        if (self.d_prev_matches) |matches| {
            _ = cuda.cudaFree(matches);
        }

        self.left.deinit();
        self.right.deinit();
        self.combined.deinit();
        self.allocator.destroy(self.params);
        self.allocator.destroy(self.num_matches);
        self.allocator.destroy(self);
    }

    fn detectKeypoints(
        resource_manager: *DetectionResourceManager,
        params: *StereoParams,
    ) !c_uint {
        if (resource_manager.frame) |frame| {
            try resource_manager.updateTransformation();

            const block = cuda.dim3{
                .x = 16,
                .y = 16,
                .z = 1,
            };

            const grid = cuda.dim3{
                .x = (@as(c_uint, @intCast(frame.width)) + block.x - 1) / block.x,
                .y = (@as(c_uint, @intCast(frame.height)) + block.y - 1) / block.y,
                .z = 1,
            };

            cuda.init_gaussian_kernel(params.sigma);

            var d_temp1: [*]u8 = undefined;
            var d_temp2: [*]u8 = undefined;

            const buffer_size = @as(usize, @intCast(frame.linesize[0])) * @as(usize, @intCast(frame.height));

            const err1 = cuda.cudaMalloc(@ptrCast(&d_temp1), buffer_size);
            const err2 = cuda.cudaMalloc(@ptrCast(&d_temp2), buffer_size);

            if (err1 != cuda.cudaSuccess or err2 != cuda.cudaSuccess) {
                return error.CudaAllocationFailed;
            }

            defer _ = cuda.cudaFree(d_temp1);
            defer _ = cuda.cudaFree(d_temp2);

            try CudaGL.track("Gaussian_Blur", void, cuda.launch_gaussian_blur, .{
                frame.data[0],
                d_temp1,
                d_temp2,
                frame.width,
                frame.height,
                frame.linesize[0],
                grid,
                block,
            });

            try CudaGL.track("Keypoint_Detection", void, cuda.launch_keypoint_detection, .{
                d_temp2,
                frame.width,
                frame.height,
                frame.linesize[0],
                params.intensity_threshold,
                params.arc_length,
                resource_manager.resources.keypoint_resources.?.keypoints.positions.d_buffer,
                resource_manager.resources.keypoint_resources.?.keypoints.colors.d_buffer,
                resource_manager.resources.d_descriptors,
                resource_manager.resources.d_keypoint_count,
                @as(c_int, @intCast(params.max_keypoints)),
                grid,
                block,
            });

            const err = cuda.cudaMemcpy(
                resource_manager.num_keypoints,
                resource_manager.resources.d_keypoint_count,
                @sizeOf(c_uint),
                cuda.cudaMemcpyDeviceToHost,
            );

            if (err != cuda.cudaSuccess) {
                std.debug.print("Copying num keypoints device => host failed: {s}\n", .{cuda.cudaGetErrorString(err)});
                return error.CudaMemcpyFailed;
            }

            try CudaGL.track("Texture_Update", void, cuda.launch_texture_update, .{
                resource_manager.resources.gl_textures.y.?.surface,
                resource_manager.resources.gl_textures.uv.?.surface,
                resource_manager.resources.gl_textures.depth.?.surface,
                frame.data[0],
                frame.data[1],
                frame.width,
                frame.height,
                frame.linesize[0],
                frame.linesize[1],
                grid,
                block,
            });

            resource_manager.target_node.texture_updated = true;
            for (resource_manager.target_node.children.items) |child| {
                child.instance_data.?.count = @intCast(resource_manager.num_keypoints.*);
            }

            return resource_manager.num_keypoints.*;
        }

        return 0;
    }

    pub fn match(self: *Self) !void {
        std.debug.print("\n\n\nStarting Detection...\n", .{});
        self.left.mutex.lock();
        self.right.mutex.lock();
        self.combined.mutex.lock();

        defer {
            self.left.mutex.unlock();
            self.right.mutex.unlock();
            self.combined.mutex.unlock();
        }

        const num_keypoints_left = try detectKeypoints(self.left, self.params);
        const num_keypoints_right = try detectKeypoints(self.right, self.params);

        std.debug.print("Detections =>  Left : {d} | Right: {d}\n", .{ num_keypoints_left, num_keypoints_right });

        const max_matches = @min(num_keypoints_left, num_keypoints_right);
        if (max_matches == 0 or self.params.disable_matching) {
            std.debug.print("Left / Right Frame not found or Matching Disabled. Skipping Matching...\n", .{});
            self.num_matches.* = 0;
            return;
        }

        std.debug.print("\nStarting Matching...\n", .{});

        var d_matches_left: ?*cuda.BestMatch = null;
        var d_matches_right: ?*cuda.BestMatch = null;

        var err = cuda.cudaMalloc(@ptrCast(&d_matches_left), num_keypoints_left * @sizeOf(cuda.BestMatch));
        err |= cuda.cudaMalloc(@ptrCast(&d_matches_right), num_keypoints_right * @sizeOf(cuda.BestMatch));

        defer {
            if (d_matches_left) |ptr| {
                _ = cuda.cudaFree(ptr);
            }
            if (d_matches_right) |ptr| {
                _ = cuda.cudaFree(ptr);
            }
        }

        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to allocate memory for best matches!\n", .{});
            return error.CudaMallocFailed;
        }

        try CudaGL.track("Stereo_Matching", void, cuda.launch_stereo_matching, .{
            self.left.resources.keypoint_resources.?.keypoints.positions.d_buffer,
            self.left.resources.d_descriptors,
            num_keypoints_left,
            self.right.resources.keypoint_resources.?.keypoints.positions.d_buffer,
            self.right.resources.d_descriptors,
            num_keypoints_right,
            self.params.*,
            d_matches_left,
            d_matches_right,
        });

        var d_match_count: ?*c_uint = null;

        err = cuda.cudaMalloc(@ptrCast(&self.d_matches), max_matches * @sizeOf(cuda.MatchedKeypoint));
        err |= cuda.cudaMalloc(@ptrCast(&d_match_count), @sizeOf(c_uint));

        if (d_match_count) |ptr| {
            _ = cuda.cudaMemset(ptr, 0, @sizeOf(c_uint));
        }

        defer {
            if (d_match_count) |ptr| {
                _ = cuda.cudaFree(ptr);
            }
        }

        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to allocate memory for matches and match count! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
            return error.CudaMallocFailed;
        }

        var block = cuda.dim3{
            .x = 512,
            .y = 1,
            .z = 1,
        };

        var grid = cuda.dim3{
            .x = (num_keypoints_left + block.x - 1) / block.x,
            .y = 1,
            .z = 1,
        };

        try CudaGL.track("Cross_Check_Matches", void, cuda.launch_cross_check_matches, .{
            d_matches_left,
            d_matches_right,
            self.left.resources.keypoint_resources.?.keypoints.positions.d_buffer,
            self.right.resources.keypoint_resources.?.keypoints.positions.d_buffer,
            num_keypoints_left,
            num_keypoints_right,
            self.params.*,
            self.d_matches.?,
            d_match_count.?,
            grid,
            block,
        });

        err = cuda.cudaMemcpy(
            self.num_matches,
            d_match_count.?,
            @sizeOf(c_uint),
            cuda.cudaMemcpyDeviceToHost,
        );

        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to copy num matches from device => host! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
            return error.CudaMallocFailed;
        }

        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to copy matches from device => host! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
            return error.CudaMallocFailed;
        }

        defer std.debug.print("Matches Found: {d}\n", .{self.num_matches.*});
        if (self.num_matches.* == 0) {
            return;
        }

        block = cuda.dim3{
            .x = 512,
            .y = 1,
            .z = 1,
        };

        grid = cuda.dim3{
            .x = @divTrunc(self.num_matches.* + block.x - 1, block.x),
            .y = 1,
            .z = 1,
        };

        const combined_resources = self.combined.resources;
        const keypoint_resources = combined_resources.keypoint_resources.?.keypoints;
        const connection_resources = combined_resources.keypoint_resources.?.connections;

        try CudaGL.track("Visualize_Triangulation", void, cuda.launch_visualization, .{
            self.d_matches,
            self.num_matches.*,
            keypoint_resources.positions.d_buffer,
            keypoint_resources.colors.d_buffer,
            connection_resources.left.?.positions.d_buffer,
            connection_resources.left.?.colors.d_buffer,
            connection_resources.right.?.positions.d_buffer,
            connection_resources.right.?.colors.d_buffer,
            self.left.resources.d_world_transform,
            self.right.resources.d_world_transform,
            self.params.show_connections,
            self.params.disable_depth,
            grid,
            block,
        });

        const frame = self.left.frame.?;
        block = cuda.dim3{
            .x = 16,
            .y = 16,
            .z = 1,
        };
        grid = cuda.dim3{
            .x = @divTrunc(@as(c_uint, @intCast(frame.width)) + block.x - 1, block.x),
            .y = @divTrunc(@as(c_uint, @intCast(frame.height)) + block.y - 1, block.y),
            .z = 1,
        };

        // Copy current frame textures to temporal view
        try CudaGL.track("Copy_Temporal_Texture", void, cuda.launch_surface_copy, .{
            self.combined.resources.gl_textures.y.?.surface,
            self.combined.resources.gl_textures.uv.?.surface,
            self.combined.resources.gl_textures.depth.?.surface,
            self.temporal.resources.gl_textures.y.?.surface,
            self.temporal.resources.gl_textures.uv.?.surface,
            self.temporal.resources.gl_textures.depth.?.surface,
            frame.width,
            frame.height,
            grid,
            block,
        });

        const src_textures = self.left.resources.gl_textures;
        const dest_textures = self.combined.resources.gl_textures;

        try CudaGL.track("Surface_Copy", void, cuda.launch_surface_copy, .{
            src_textures.y.?.surface,
            src_textures.uv.?.surface,
            src_textures.depth.?.surface,
            dest_textures.y.?.surface,
            dest_textures.uv.?.surface,
            dest_textures.depth.?.surface,
            frame.width,
            frame.height,
            grid,
            block,
        });

        if (!self.params.disable_depth) {
            try CudaGL.track("Depth_Interpolation", void, cuda.launch_depth_texture_update, .{
                combined_resources.gl_textures.depth.?.surface,
                self.d_matches,
                self.num_matches.*,
                frame.width,
                frame.height,
                grid,
                block,
            });
        }

        // Update visualization
        self.temporal.target_node.texture_updated = true;
        self.combined.target_node.texture_updated = true;
        for (self.combined.target_node.children.items) |child| {
            child.instance_data.?.count = @intCast(self.num_matches.*);
        }

        _ = cuda.cudaDeviceSynchronize();
    }

    pub fn estimate_pose(self: *Self) !void {
        if (self.num_matches.* == 0) {
            std.debug.print("No matches found. Skipping Pose Estimation...\n", .{});
            return;
        }

        if (self.d_prev_matches == null) {
            std.debug.print("No Previous Matches Found. Starting Pose Estimation on next frame...\n", .{});

            // Store current matches as previous
            try self.store_current_as_previous();

            return;
        }

        std.debug.print("\nStarting Pose Estimation...\n", .{});

        // Allocate device memory for temporal matching
        var d_temporal_matches: ?*cuda.TemporalMatch = null;
        var d_temporal_match_count: ?*c_uint = null;
        var d_best_pose: ?*cuda.CameraPose = null;
        var d_inlier_count: ?*c_uint = null;
        var d_curr_to_prev_matches: ?*cuda.BestMatch = null;
        var d_prev_to_curr_matches: ?*cuda.BestMatch = null;

        // Allocate all required device memory
        var err = cuda.cudaMalloc(@ptrCast(&d_curr_to_prev_matches), self.num_matches.* * @sizeOf(cuda.BestMatch));
        err |= cuda.cudaMalloc(@ptrCast(&d_prev_to_curr_matches), self.prev_num_matches * @sizeOf(cuda.BestMatch));
        err |= cuda.cudaMalloc(@ptrCast(&d_temporal_matches), @min(self.num_matches.*, self.prev_num_matches) * @sizeOf(cuda.TemporalMatch));
        err |= cuda.cudaMalloc(@ptrCast(&d_temporal_match_count), @sizeOf(c_uint));
        err |= cuda.cudaMalloc(@ptrCast(&d_best_pose), @sizeOf(cuda.CameraPose));
        err |= cuda.cudaMalloc(@ptrCast(&d_inlier_count), @sizeOf(c_uint));

        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to allocate memory for Temporal Matching! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
            return error.CudaAllocationFailed;
        }

        defer {
            if (d_temporal_matches) |ptr| _ = cuda.cudaFree(ptr);
            if (d_temporal_match_count) |ptr| _ = cuda.cudaFree(ptr);
            if (d_best_pose) |ptr| _ = cuda.cudaFree(ptr);
            if (d_inlier_count) |ptr| _ = cuda.cudaFree(ptr);
            if (d_curr_to_prev_matches) |ptr| _ = cuda.cudaFree(ptr);
            if (d_prev_to_curr_matches) |ptr| _ = cuda.cudaFree(ptr);
        }

        // Reset counters
        err = cuda.cudaMemset(d_temporal_match_count, 0, @sizeOf(c_uint));
        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to reset temporal match count: {s}\n", .{cuda.cudaGetErrorString(err)});
            return error.CudaMemsetFailed;
        }

        err = cuda.cudaMemset(d_inlier_count, 0, @sizeOf(c_uint));
        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to reset inlier count: {s}\n", .{cuda.cudaGetErrorString(err)});
            return error.CudaMemsetFailed;
        }

        const block = cuda.dim3{ .x = 128, .y = 1, .z = 1 };
        const curr_grid = cuda.dim3{
            .x = (self.num_matches.* + block.x - 1) / block.x,
            .y = 1,
            .z = 1,
        };
        const prev_grid = cuda.dim3{
            .x = (self.prev_num_matches + block.x - 1) / block.x,
            .y = 1,
            .z = 1,
        };

        // Match current to previous
        try CudaGL.track("Temporal_Match_Current_To_Prev", void, cuda.launch_temporal_match_current_to_prev, .{
            self.d_matches.?,
            self.num_matches.*,
            self.d_prev_matches.?,
            self.prev_num_matches,
            d_curr_to_prev_matches.?,
            self.temporal_params.*,
            curr_grid,
            block,
        });

        // Match previous to current
        try CudaGL.track("Temporal_Match_Prev_To_Current", void, cuda.launch_temporal_match_prev_to_current, .{
            self.d_prev_matches.?,
            self.prev_num_matches,
            self.d_matches.?,
            self.num_matches.*,
            d_prev_to_curr_matches.?,
            self.temporal_params.*,
            prev_grid,
            block,
        });

        // Cross check matches
        try CudaGL.track("Temporal_Cross_Check", void, cuda.launch_temporal_cross_check, .{
            d_curr_to_prev_matches.?,
            d_prev_to_curr_matches.?,
            self.d_matches.?,
            self.d_prev_matches.?,
            self.num_matches.*,
            self.prev_num_matches,
            d_temporal_matches.?,
            d_temporal_match_count.?,
            self.temporal_params.*,
            curr_grid,
            block,
        });

        var host_match_count: c_uint = undefined;
        err = cuda.cudaMemcpy(&host_match_count, d_temporal_match_count.?, @sizeOf(c_uint), cuda.cudaMemcpyDeviceToHost);

        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to copy match count: {s}\n", .{cuda.cudaGetErrorString(err)});
            return error.CudaMemcpyFailed;
        }

        std.debug.print("Found {d} temporal matches\n", .{host_match_count});

        if (host_match_count < self.temporal_params.min_matches) {
            std.debug.print("Not enough matches for RANSAC (need {d})\n", .{self.temporal_params.min_matches});

            // Update temporal visualization counts
            for (self.temporal.target_node.children.items, 0..) |child, ind| {
                switch (ind) {
                    0 => child.instance_data.?.count = @intCast(self.prev_num_matches),
                    1 => child.instance_data.?.count = @intCast(host_match_count),
                    else => {},
                }
            }

            try self.store_current_as_previous();
            return;
        }

        // Launch RANSAC kernel for motion estimation
        const ransac_block = cuda.dim3{
            .x = 128,
            .y = 1,
            .z = 1,
        };

        const ransac_grid = cuda.dim3{
            .x = self.temporal_params.ransac_iterations, // Use local value
            .y = 1,
            .z = 1,
        };

        try CudaGL.track("Motion_Estimation", void, cuda.launch_motion_estimation, .{
            d_temporal_matches.?,
            host_match_count,
            d_best_pose.?,
            d_inlier_count.?,
            self.temporal_params,
            ransac_grid,
            ransac_block,
        });

        std.debug.print("Motion Estimation complete. Copying Pose back to host...\n", .{});
        // Copy results back
        var c_best_pose: cuda.CameraPose = undefined;
        err = cuda.cudaMemcpy(&c_best_pose, d_best_pose, @sizeOf(cuda.CameraPose), cuda.cudaMemcpyDeviceToHost);

        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to copy pose from device! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
            return error.CudaMemcpyFailed;
        }

        std.debug.print("Best Pose:\nTranslation:{d}\nRotation:{d}\n", .{ c_best_pose.translation, c_best_pose.rotation });

        const best_pose = CameraPose.fromCType(c_best_pose);
        self.combined.target_node.scene.?.appState.pose = best_pose;

        if (!self.params.disable_spatial_tracking) {
            // Update current pose
            const rotation_quaternion = best_pose.toQuaternion();
            const translation_vec3 = best_pose.toVec3();

            // Apply the new transform to the combined target node
            self.combined.target_node.setRotation(rotation_quaternion);
            self.combined.target_node.setPosition(translation_vec3.x, translation_vec3.y, translation_vec3.z);
            try self.combined.updateTransformation();
        }

        const visualize_block = cuda.dim3{
            .x = 512,
            .y = 1,
            .z = 1,
        };
        const visualize_grid = cuda.dim3{
            .x = @divTrunc(host_match_count + block.x - 1, block.x),
            .y = 1,
            .z = 1,
        };

        try self.combined.updateTransformation();
        try CudaGL.track("Visualize_Temporal_Matches", void, cuda.launch_temporal_visualization, .{
            d_temporal_matches.?,
            host_match_count,
            self.temporal.resources.keypoint_resources.?.keypoints.positions.d_buffer,
            self.temporal.resources.keypoint_resources.?.keypoints.colors.d_buffer,
            self.temporal.resources.keypoint_resources.?.connections.left.?.positions.d_buffer,
            self.temporal.resources.keypoint_resources.?.connections.left.?.colors.d_buffer,
            self.temporal.resources.d_world_transform,
            self.combined.resources.d_world_transform,
            self.params.disable_depth,
            visualize_grid,
            visualize_block,
        });

        // Update temporal visualization counts
        for (self.temporal.target_node.children.items, 0..) |child, ind| {
            switch (ind) {
                0 => child.instance_data.?.count = @intCast(self.prev_num_matches),
                1 => child.instance_data.?.count = @intCast(host_match_count),
                else => {},
            }
        }

        try self.store_current_as_previous();
    }

    pub fn update(self: *Self) !void {
        try self.right.map_resources();
        try self.left.map_resources();
        try self.combined.map_resources();
        try self.temporal.map_resources();

        defer self.right.unmap_resources();
        defer self.left.unmap_resources();
        defer self.combined.unmap_resources();
        defer self.temporal.unmap_resources();

        try self.match();
        try self.estimate_pose();
    }

    fn store_current_as_previous(self: *Self) !void {
        // Free previous matches
        if (self.d_prev_matches) |matches| {
            _ = cuda.cudaFree(matches);
            self.d_prev_matches = null;
        }

        // Store current matches as previous
        self.prev_num_matches = self.num_matches.*;
        var err = cuda.cudaMalloc(
            @ptrCast(&self.d_prev_matches),
            self.num_matches.* * @sizeOf(cuda.MatchedKeypoint),
        );

        if (err != cuda.cudaSuccess) {
            std.debug.print("Error Allocating mem for prev matches! {s} => {s}", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
            return error.FailedToAllocatePrevMatches;
        }

        err = cuda.cudaMemcpy(
            self.d_prev_matches.?,
            self.d_matches.?,
            self.num_matches.* * @sizeOf(cuda.MatchedKeypoint),
            cuda.cudaMemcpyDeviceToDevice,
        );
        defer self.free_matches();

        if (err != cuda.cudaSuccess) {
            std.debug.print("Error Copying Matches to prev! {s} => {s}", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
            return error.FailedToCopyMatchedToPrev;
        }
    }

    pub fn free_matches(self: *Self) void {
        if (self.d_matches) |ptr| {
            _ = cuda.cudaFree(ptr);
            self.d_matches = null;
        }
    }
};

pub const DetectionResourceManager = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex,
    resources: *DetectionResources,
    target_node: *Node,
    frame: ?*video.AVFrame,
    num_keypoints: *c_uint,

    pub fn init(
        allocator: std.mem.Allocator,
        target_node: *Node,
        id: u32,
        max_keypoints: u32,
        buffer_ids: CudaGL.BufferParams,
        left_connection: ?CudaGL.BufferParams,
        right_connection: ?CudaGL.BufferParams,
    ) !*Self {
        var manager = try allocator.create(Self);
        manager.* = .{
            .allocator = allocator,
            .mutex = std.Thread.Mutex{},
            .target_node = target_node,
            .frame = null,
            .num_keypoints = try allocator.create(c_uint),
            .resources = undefined,
        };
        manager.num_keypoints.* = 0;

        const mesh = target_node.mesh.?;
        manager.resources = try DetectionResources.init(
            allocator,
            id,
            max_keypoints,
            mesh.textureID.y,
            mesh.textureID.uv,
            mesh.textureID.depth,
            buffer_ids,
            left_connection,
            right_connection,
        );

        return manager;
    }

    pub fn deinit(self: *Self) void {
        self.resources.deinit();
        self.allocator.destroy(self.num_keypoints);
        if (self.frame) |frame| {
            video.av_frame_free(@constCast(@ptrCast(&frame)));
        }
        self.allocator.destroy(self);
    }

    pub fn queueFrame(self: *Self, frame: *video.AVFrame) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.frame) |curr_frame| {
            video.av_frame_free(@constCast(@ptrCast(&curr_frame)));
        }

        self.frame = video.av_frame_clone(frame);
    }

    pub fn map_resources(self: *Self) !void {
        self.num_keypoints.* = 0;
        try self.resources.map_resources();
    }

    pub fn unmap_resources(self: *Self) void {
        const err = cuda.cudaMemset(self.resources.d_keypoint_count, 0, @sizeOf(c_uint));
        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to reset keypoint count: {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
        }

        self.resources.unmap_resources();
    }

    pub fn updateTransformation(self: *Self) !void {
        try self.resources.map_transformation(self.target_node.world_transform);
    }
};
