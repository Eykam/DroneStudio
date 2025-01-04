const std = @import("std");
const Node = @import("Node.zig");
const KeypointDebugger = @import("Shape.zig").InstancedKeypointDebugger;
const CudaBinds = @import("bindings/cuda.zig");
const libav = @import("bindings/libav.zig");
const gl = @import("bindings/gl.zig");
const KeyPoint = CudaBinds.KeyPoint;
const video = libav.video;
const glad = gl.glad;

pub const StereoMatcher = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    baseline: f32, // in millimeters
    focal_length: f32, // in pixels
    max_disparity: f32,
    epipolar_threshold: f32,
    num_matches: *c_int,

    left: *KeypointManager,
    right: *KeypointManager,

    combined: *KeypointManager,

    pub fn init(allocator: std.mem.Allocator, left: *KeypointManager, right: *KeypointManager, combined: *KeypointManager) !*Self {
        var matcher = try allocator.create(Self);

        matcher.allocator = allocator;

        matcher.baseline = 76.3;
        matcher.focal_length = 2.75;
        matcher.max_disparity = 100.0;
        matcher.epipolar_threshold = 5.0;
        matcher.num_matches = try allocator.create(c_int);
        matcher.num_matches.* = 0;

        matcher.left = left;
        matcher.right = right;

        matcher.combined = combined;

        return matcher;
    }

    // TODO:Implement
    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn match(self: *Self) !void {
        self.left.mutex.lock();
        self.right.mutex.lock();

        defer self.left.mutex.unlock();
        defer self.right.mutex.unlock();

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
                .width = frame_width,
                .height = frame_height,
                .y_linesize = frame.linesize[0],
                .image_width = @floatFromInt(frame_width),
                .image_height = @floatFromInt(frame_height),
                .num_keypoints = self.left.num_keypoints,
            };
        }

        if (self.right.frame) |frame| {
            const frame_width = frame.width;
            const frame_height = frame.height;

            right_image_params = try self.allocator.create(CudaBinds.ImageParams);

            right_image_params.?.* = CudaBinds.ImageParams{
                .y_plane = frame.data[0],
                .width = frame_width,
                .height = frame_height,
                .y_linesize = frame.linesize[0],
                .image_width = @floatFromInt(frame_width),
                .image_height = @floatFromInt(frame_height),
                .num_keypoints = self.right.num_keypoints,
            };
        }

        if (right_image_params == null and left_image_params == null) {
            std.debug.print("Unable to create Image params for both left and right...\n", .{});
            return;
        } else if (right_image_params != null and left_image_params == null) {
            std.debug.print("Unable to create Image params for left, Continuing to detect and render Right image...\n", .{});
            try self.right.keypoint_detector.?.detect_keypoints(
                self.right.threshold,
                right_image_params.?,
            );

            for (self.right.target_node.children.items) |child| {
                child.instance_data.?.count = @intCast(self.right.num_keypoints.*);
            }
            return;
        } else if (left_image_params != null and right_image_params == null) {
            std.debug.print("Unable to create Image params for right, Continuing to detect and render Left image...\n", .{});
            try self.left.keypoint_detector.?.detect_keypoints(
                self.left.threshold,
                left_image_params.?,
            );

            for (self.left.target_node.children.items) |child| {
                child.instance_data.?.count = @intCast(self.left.num_keypoints.*);
            }
            return;
        }

        try CudaBinds.CudaKeypointDetector.match_keypoints(
            self.left.keypoint_detector.?.detector_id,
            self.right.keypoint_detector.?.detector_id,
            self.combined.keypoint_detector.?.detector_id,
            self.baseline,
            self.focal_length,
            self.num_matches,
            self.left.threshold,
            left_image_params.?,
            right_image_params.?,
        );

        // Just update the count
        // std.debug.print("Left keypoints: {}\n", .{self.left.num_keypoints.*});
        // std.debug.print("Right keypoints: {}\n", .{self.right.num_keypoints.*});
        // std.debug.print("Detected Matches: {}\n", .{self.num_matches.*});

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

    keypoint_detector: ?CudaBinds.CudaKeypointDetector,

    threshold: u8,
    max_keypoints: u32,
    frame_width: u32,
    frame_height: u32,
    num_keypoints: *c_int,

    frame: ?*video.AVFrame,

    pub fn init(allocator: std.mem.Allocator, target_node: *Node) !*Self {
        var self = try allocator.create(KeypointManager);

        self.allocator = std.heap.c_allocator;
        self.arena = std.heap.ArenaAllocator.init(self.allocator);

        self.mutex = std.Thread.Mutex{};

        self.num_keypoints = try allocator.create(c_int);
        self.num_keypoints.* = 0;
        self.max_keypoints = 500000;
        self.threshold = 15;

        self.target_node = target_node;
        self.frame = null;

        const node = try KeypointDebugger.init(
            self.arena.allocator(),
            self.max_keypoints,
        );
        try self.target_node.addChild(node);

        var detector = try CudaBinds.CudaKeypointDetector.init();

        std.debug.print("Node instance_data {any}\n", .{node.instance_data});
        if (glad.glGetError() != glad.GL_NO_ERROR) {
            std.debug.print("GL error before registration\n", .{});
            return error.GLError;
        }
        std.debug.print("Current openGL rendering device => {s} {s}\n", .{ std.mem.span(glad.glGetString(glad.GL_VENDOR)), std.mem.span(glad.glGetString(glad.GL_RENDERER)) });

        // Ensure buffers are valid
        if (glad.glIsBuffer(node.instance_data.?.position_buffer) != 1 or glad.glIsBuffer(node.instance_data.?.color_buffer) != 1) {
            return error.InvalidBuffer;
        }

        try detector.enableGLInterop(
            node.instance_data.?.position_buffer,
            node.instance_data.?.color_buffer,
            self.max_keypoints,
        );

        self.keypoint_detector = detector;

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.keypoint_detector.?.deinit();
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
