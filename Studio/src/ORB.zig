const std = @import("std");
const Node = @import("Node.zig");
const KeypointDebugger = @import("Shape.zig").InstancedKeypointDebugger;
const InstancedLine = @import("Shape.zig").InstancedLine;
const CudaBinds = @import("bindings/cuda.zig");
const StereoParams = CudaBinds.StereoParams;
const libav = @import("bindings/libav.zig");
const gl = @import("bindings/gl.zig");
const video = libav.video;
const glad = gl.glad;

pub const StereoMatcher = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    params: *StereoParams,
    params_changed: bool,
    num_matches: *c_int,

    left: *KeypointManager,
    right: *KeypointManager,

    left_to_combined: *Node,
    right_to_combined: *Node,

    combined: *KeypointManager,

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

        matcher.params = stereo_params;
        matcher.params_changed = false;
        matcher.num_matches = try allocator.create(c_int);
        matcher.num_matches.* = 0;

        matcher.left = try KeypointManager.init(
            allocator,
            .{ 0.0, 0.0, 1.0 },
            left_node,
            matcher.params.max_keypoints,
        );
        matcher.right = try KeypointManager.init(
            allocator,
            .{ 0.0, 0.0, 1.0 },
            right_node,
            matcher.params.max_keypoints,
        );
        matcher.combined = try KeypointManager.init(
            allocator,
            .{ 0.0, 0.0, 1.0 },
            combined_node,
            matcher.params.max_keypoints,
        );

        const left_to_combined = try InstancedLine.init(
            matcher.allocator,
            [_]f32{ 1.0, 0.0, 0.0 },
            matcher.params.max_keypoints,
        );
        const right_to_combined = try InstancedLine.init(
            matcher.allocator,
            [_]f32{ 0.0, 0.0, 1.0 },
            matcher.params.max_keypoints,
        );

        try matcher.combined.target_node.addChild(left_to_combined);
        try matcher.combined.target_node.addChild(right_to_combined);

        matcher.left_to_combined = left_to_combined;
        matcher.right_to_combined = right_to_combined;

        try checkBufferValidAndEnable(matcher.left, null, matcher.params.max_keypoints);
        try checkBufferValidAndEnable(matcher.right, null, matcher.params.max_keypoints);
        try checkBufferValidAndEnable(
            matcher.combined,
            .{
                .left = left_to_combined,
                .right = right_to_combined,
            },
            matcher.params.max_keypoints,
        );

        return matcher;
    }

    // TODO:Implement
    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn match(self: *Self) !void {
        self.left.mutex.lock();
        self.right.mutex.lock();
        self.combined.mutex.lock();

        defer self.left.mutex.unlock();
        defer self.right.mutex.unlock();
        defer self.combined.mutex.unlock();

        std.debug.print("\n\n\nStarting Frame matching!\n", .{});

        var left_image_params: ?*CudaBinds.ImageParams = null;
        var right_image_params: ?*CudaBinds.ImageParams = null;

        defer {
            if (left_image_params) |params| self.allocator.destroy(params);
            if (right_image_params) |params| self.allocator.destroy(params);
        }

        if (self.left.frame) |frame| {
            const frame_width = frame.width;
            const frame_height = frame.height;

            left_image_params = try self.allocator.create(CudaBinds.ImageParams);

            left_image_params.?.* = CudaBinds.ImageParams{
                .y_plane = frame.data[0],
                .uv_plane = frame.data[1],
                .width = frame_width,
                .height = frame_height,
                .y_linesize = frame.linesize[0],
                .uv_linesize = frame.linesize[1],
                .image_width = @floatFromInt(frame_width),
                .image_height = @floatFromInt(frame_height),
                .num_keypoints = self.left.num_keypoints,
            };

            try self.left.keypoint_detector.?.updateDetectorTransformation(self.left.target_node.world_transform);
        }

        if (self.right.frame) |frame| {
            const frame_width = frame.width;
            const frame_height = frame.height;

            right_image_params = try self.allocator.create(CudaBinds.ImageParams);

            right_image_params.?.* = CudaBinds.ImageParams{
                .y_plane = frame.data[0],
                .uv_plane = frame.data[1],
                .width = frame_width,
                .height = frame_height,
                .y_linesize = frame.linesize[0],
                .uv_linesize = frame.linesize[1],
                .image_width = @floatFromInt(frame_width),
                .image_height = @floatFromInt(frame_height),
                .num_keypoints = self.right.num_keypoints,
            };

            try self.right.keypoint_detector.?.updateDetectorTransformation(self.right.target_node.world_transform);
        }

        if (right_image_params == null and left_image_params == null) {
            std.debug.print("Unable to create Image params for both left and right...\n", .{});
            return;
        } else if (right_image_params != null and left_image_params == null) {
            std.debug.print("Unable to create Image params for left, Continuing to detect and render Right image...\n", .{});
            try self.right.keypoint_detector.?.detect_keypoints(
                self.params,
                right_image_params.?,
            );

            self.right.target_node.texture_updated = true;
            for (self.right.target_node.children.items) |child| {
                child.instance_data.?.count = @intCast(self.right.num_keypoints.*);
            }
            return;
        } else if (left_image_params != null and right_image_params == null) {
            std.debug.print("Unable to create Image params for right, Continuing to detect and render Left image...\n", .{});
            try self.left.keypoint_detector.?.detect_keypoints(
                self.params,
                left_image_params.?,
            );

            self.left.target_node.texture_updated = true;
            for (self.left.target_node.children.items) |child| {
                child.instance_data.?.count = @intCast(self.left.num_keypoints.*);
            }
            return;
        }

        try self.combined.keypoint_detector.?.updateDetectorTransformation(self.combined.target_node.world_transform);

        try CudaBinds.CudaKeypointDetector.match_keypoints(
            self.left.keypoint_detector.?.detector_id,
            self.right.keypoint_detector.?.detector_id,
            self.combined.keypoint_detector.?.detector_id,
            self.params,
            self.num_matches,
            left_image_params.?,
            right_image_params.?,
        );

        self.left.target_node.texture_updated = true;
        self.right.target_node.texture_updated = true;
        self.combined.target_node.texture_updated = true;

        for (self.left.target_node.children.items) |child| {
            child.instance_data.?.count = @intCast(self.left.num_keypoints.*);
        }

        for (self.right.target_node.children.items) |child| {
            child.instance_data.?.count = @intCast(self.right.num_keypoints.*);
        }

        for (self.combined.target_node.children.items) |child| {
            child.instance_data.?.count = @intCast(self.num_matches.*);
        }
    }
};

pub const KeypointManager = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex,
    gpa: std.heap.GeneralPurposeAllocator(.{}),
    arena: std.heap.ArenaAllocator,

    target_node: *Node,
    keypoints: *Node,

    keypoint_detector: ?*CudaBinds.CudaKeypointDetector,
    frame_width: u32,
    frame_height: u32,
    num_keypoints: *c_int,

    frame: ?*video.AVFrame,

    pub fn init(
        allocator: std.mem.Allocator,
        color: ?[3]f32,
        target_node: *Node,
        max_keypoints: u32,
    ) !*Self {
        var self = try allocator.create(KeypointManager);

        self.allocator = std.heap.c_allocator;
        self.arena = std.heap.ArenaAllocator.init(self.allocator);

        self.mutex = std.Thread.Mutex{};

        self.num_keypoints = try allocator.create(c_int);
        self.num_keypoints.* = 0;

        self.target_node = target_node;
        self.frame = null;

        const node = try KeypointDebugger.init(
            self.arena.allocator(),
            color,
            max_keypoints,
        );
        try self.target_node.addChild(node);

        const mesh = target_node.mesh.?;

        self.keypoint_detector.? = try allocator.create(CudaBinds.CudaKeypointDetector);
        self.keypoint_detector.?.* = try CudaBinds.CudaKeypointDetector.init(
            max_keypoints,
            mesh.textureID.y,
            mesh.textureID.uv,
            mesh.textureID.depth,
        );

        self.keypoints = node;

        std.debug.print("Node instance_data {any}\n", .{node.instance_data});
        if (glad.glGetError() != glad.GL_NO_ERROR) {
            std.debug.print("GL error before registration\n", .{});
            return error.GLError;
        }

        std.debug.print("Current openGL rendering device => {s} {s}\n", .{
            std.mem.span(glad.glGetString(glad.GL_VENDOR)),
            std.mem.span(glad.glGetString(glad.GL_RENDERER)),
        });

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.keypoint_detector) |detector| {
            detector.deinit();
            self.allocator.destroy(detector);
        }
        self.arena.deinit();
    }

    // Called from video thread
    pub fn queueFrame(self: *Self, frame: *video.AVFrame) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.frame) |curr_frame| {
            video.av_frame_free(@constCast(@ptrCast(&curr_frame)));
        }

        self.frame = video.av_frame_clone(frame);
    }
};

fn checkBufferValidAndEnable(
    manager: *KeypointManager,
    connections: ?struct { left: *Node, right: *Node },
    max_keypoints: u32,
) !void {
    const instance_data = manager.keypoints.instance_data.?;

    if (glad.glIsBuffer(instance_data.position_buffer) != 1 or glad.glIsBuffer(instance_data.color_buffer) != 1) {
        return error.InvalidKeypointBuffer;
    }

    var keypoint_detector = manager.keypoint_detector.?;
    if (connections) |line_nodes| {
        // Combined node with line connections
        const left_instance_data = line_nodes.left.instance_data.?;
        const right_instance_data = line_nodes.right.instance_data.?;

        if (glad.glIsBuffer(left_instance_data.position_buffer) != 1 or glad.glIsBuffer(left_instance_data.color_buffer) != 1) {
            return error.InvalidLeftConnectionBuffer;
        }

        if (glad.glIsBuffer(right_instance_data.position_buffer) != 1 or glad.glIsBuffer(right_instance_data.color_buffer) != 1) {
            return error.InvalidRightConnectionBuffer;
        }

        try keypoint_detector.enableGlInterop(
            instance_data.position_buffer,
            instance_data.color_buffer,
            left_instance_data.position_buffer,
            left_instance_data.color_buffer,
            right_instance_data.position_buffer,
            right_instance_data.color_buffer,
            max_keypoints,
        );
    } else {
        // Left or right node without connections
        try keypoint_detector.enableGlInterop(
            instance_data.position_buffer,
            instance_data.color_buffer,
            null,
            null,
            null,
            null,
            max_keypoints,
        );
    }

    std.debug.print("Enabled Gl Interop => {any}!\n", .{manager.keypoint_detector.?.gl_interop_enabled});
    return;
}
