const std = @import("std");
const Node = @import("Node.zig");
const Shape = @import("Shape.zig");
const libav = @import("bindings/libav.zig");
const gl = @import("bindings/gl.zig");
const CudaGL = @import("CudaGL.zig");
const cuda = CudaGL.cuda;
const video = libav.video;
const glad = gl.glad;

const KeypointDebugger = Shape.InstancedKeypointDebugger;
const InstancedLine = Shape.InstancedLine;

const DetectionResources = CudaGL.DetectionResources;
const StereoParams = CudaGL.StereoParams;
const ImageParams = CudaGL.ImageParams;

pub const StereoMatcher = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    params: *StereoParams,
    params_changed: bool,
    num_matches: *c_int,

    left: *DetectionResourceManager,
    right: *DetectionResourceManager,
    combined: *DetectionResourceManager,

    left_to_combined: *Node,
    right_to_combined: *Node,

    pub fn init(
        allocator: std.mem.Allocator,
        left_node: *Node,
        right_node: *Node,
        combined_node: *Node,
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
        };

        matcher.* = .{
            .allocator = allocator,
            .params = stereo_params,
            .params_changed = false,
            .num_matches = try allocator.create(c_int),
            .left = undefined,
            .right = undefined,
            .combined = undefined,
            .left_to_combined = undefined,
            .right_to_combined = undefined,
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

        try left_node.addChild(left_keypoints);
        try right_node.addChild(right_keypoints);
        try combined_node.addChild(combined_keypoints);

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

        try combined_node.addChild(left_to_combined);
        try combined_node.addChild(right_to_combined);

        const left_instance = left_keypoints.instance_data.?;
        const right_instance = right_keypoints.instance_data.?;
        const combined_instance = combined_keypoints.instance_data.?;
        const left_line = left_to_combined.instance_data.?;
        const right_line = right_to_combined.instance_data.?;

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

        return matcher;
    }

    pub fn deinit(self: *Self) void {
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
    ) !void {
        if (resource_manager.frame) |frame| {
            try resource_manager.updateTransformation();

            try resource_manager.resources.map_resources();
            defer resource_manager.resources.unmap_resources();

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

            cuda.launch_gaussian_blur(
                frame.data[0],
                d_temp1,
                d_temp2,
                frame.width,
                frame.height,
                frame.linesize[0],
                grid,
                block,
            );

            var err = cuda.cudaGetLastError();
            if (err != cuda.cudaSuccess) {
                std.debug.print("Gaussian Blur horizontal Kernel failed: {s}\n", .{cuda.cudaGetErrorString(err)});
                return error.CudaKernelError;
            }

            cuda.launch_keypoint_detection(
                frame.data[0],
                frame.width,
                frame.height,
                frame.linesize[0],
                params.intensity_threshold,
                params.arc_length,
                resource_manager.resources.keypoint_resources.?.keypoints.positions.d_buffer,
                resource_manager.resources.keypoint_resources.?.keypoints.colors.d_buffer,
                resource_manager.resources.d_descriptors,
                @ptrCast(resource_manager.num_keypoints),
                @intCast(params.max_keypoints),
                grid,
                block,
            );

            err = cuda.cudaGetLastError();
            if (err != cuda.cudaSuccess) {
                std.debug.print("Gaussian Blur vertical Kernel failed: {s}\n", .{cuda.cudaGetErrorString(err)});
                return error.CudaKernelError;
            }

            err = cuda.cudaMemcpy(
                resource_manager.num_keypoints,
                resource_manager.resources.d_keypoint_count,
                @sizeOf(u32),
                cuda.cudaMemcpyDeviceToHost,
            );

            if (err != cuda.cudaSuccess) {
                return error.CudaMemcpyFailed;
            }

            std.debug.print("Keypoints detected Successfully!\n", .{});

            cuda.launch_texture_update(
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
            );

            err = cuda.cudaGetLastError();
            if (err != cuda.cudaSuccess) {
                std.debug.print("Texture update Kernel failed: {s}\n", .{cuda.cudaGetErrorString(err)});
                return error.CudaKernelError;
            }

            resource_manager.target_node.texture_updated = true;
            for (resource_manager.target_node.children.items) |child| {
                child.instance_data.?.count = @intCast(resource_manager.num_keypoints.*);
            }
        }
    }

    pub fn match(self: *Self) !void {
        self.left.mutex.lock();
        self.right.mutex.lock();
        self.combined.mutex.lock();

        defer {
            self.left.mutex.unlock();
            self.right.mutex.unlock();
            self.combined.mutex.unlock();
        }

        // Detect keypoints in both images
        try detectKeypoints(self.left, self.params);
        try detectKeypoints(self.right, self.params);

        if (self.left.frame == null or self.right.frame == null) {
            std.debug.print("Missing frames for stereo matching\n", .{});
            return;
        }

        // const block = cuda.dim3{
        //     .x = 256,
        //     .y = 1,
        //     .z = 1,
        // };

        // const grid = cuda.dim3{
        //     .x = (@as(c_uint, @intCast(self.left.num_keypoints)) + block.x - 1) / block.x,
        //     .y = (@as(c_uint, @intCast(self.right.num_keypoints)) + block.y - 1) / block.y,
        //     .z = 1,
        // };

        const matches_left: ?*cuda.BestMatch = null;
        const matches_right: ?*cuda.BestMatch = null;

        var err = cuda.cudaMalloc(@ptrCast(@constCast(&matches_left)), @sizeOf(cuda.BestMatch));
        err |= cuda.cudaMalloc(@ptrCast(@constCast(&matches_right)), @sizeOf(cuda.BestMatch));

        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to allocate memory for matches!\n", .{});
            return error.CudaMallocFailed;
        }

        cuda.launch_stereo_matching(
            self.left.resources.keypoint_resources.?.keypoints.positions.d_buffer,
            self.left.resources.d_descriptors,
            self.left.num_keypoints.*,
            self.right.resources.keypoint_resources.?.keypoints.positions.d_buffer,
            self.right.resources.d_descriptors,
            self.right.num_keypoints.*,
            self.params.*,
            matches_left,
            matches_right,
        );

        err = cuda.cudaGetLastError();
        if (err != cuda.cudaSuccess) {
            std.debug.print("Stereo Matching update Kernel failed: {s}\n", .{cuda.cudaGetErrorString(err)});
            return error.CudaKernelError;
        }

        // Update visualization
        self.left.target_node.texture_updated = true;
        self.right.target_node.texture_updated = true;
        self.combined.target_node.texture_updated = true;

        for (self.combined.target_node.children.items) |child| {
            child.instance_data.?.count = @intCast(self.num_matches.*);
        }
    }

    // fn createImageParams(manager: *DetectionResourceManager) !*ImageParams {
    //     const frame = manager.frame.?;
    //     const image_params = try manager.allocator.create(ImageParams);
    //     image_params.* = .{
    //         .y_plane = frame.data[0],
    //         .uv_plane = frame.data[1],
    //         .width = frame.width,
    //         .height = frame.height,
    //         .y_linesize = frame.linesize[0],
    //         .uv_linesize = frame.linesize[1],
    //         .image_width = @floatFromInt(frame.width),
    //         .image_height = @floatFromInt(frame.height),
    //         .num_keypoints = manager.num_keypoints,
    //     };
    //     return image_params;
    // }
};

pub const DetectionResourceManager = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex,
    resources: *DetectionResources,
    target_node: *Node,
    frame: ?*video.AVFrame,
    num_keypoints: *c_int,

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
            .num_keypoints = try allocator.create(c_int),
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

    pub fn registerBuffers(
        self: *Self,
        own_buffers: CudaGL.BufferParams,
        left_buffers: ?CudaGL.BufferParams,
        right_buffers: ?CudaGL.BufferParams,
    ) !void {
        try self.resources.register_buffers(own_buffers, left_buffers, right_buffers);
    }

    pub fn updateTransformation(self: *Self) !void {
        try self.resources.map_transformation(self.target_node.world_transform);
    }
};
