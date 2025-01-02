const std = @import("std");
const Node = @import("Node.zig");
const KeypointDebugger = @import("Shape.zig").InstancedKeypointDebugger;
const CudaBinds = @import("bindings/cuda.zig");
const libav = @import("bindings/libav.zig");
const gl = @import("bindings/gl.zig");
const KeyPoint = CudaBinds.KeyPoint;
const video = libav.video;
const glad = gl.glad;

// FAST Keypoint Detection
const fast_offsets = [_]i32{
    3,  0,
    3,  1,
    2,  2,
    1,  3,
    0,  3,
    -1, 3,
    -2, 2,
    -3, 1,
    -3, 0,
    -3, -1,
    -2, -2,
    -1, -3,
    0,  -3,
    1,  -3,
    2,  -2,
    3,  -1,
};

pub fn detectFastKeypoints(
    allocator: std.mem.Allocator,
    image: []const u8,
    width: usize,
    height: usize,
    linesize: c_int,
    threshold: u8,
    min_consecutive: u8,
) !std.ArrayListAligned(KeyPoint, null) {
    var keypoints = std.ArrayList(KeyPoint).init(allocator);

    const y_start = 3;
    for (y_start..(height - 3)) |y| {
        const x_start = 3;
        for (x_start..(width - 3)) |x| {
            const I_c = image[y * @as(usize, @intCast(linesize)) + x];
            var neighborhood = [_]u8{0} ** 16;
            for (0..(fast_offsets.len / 2)) |i| {
                const dx = fast_offsets[2 * i];
                const dy = fast_offsets[2 * i + 1];
                const nx = @as(usize, @intCast(@as(i32, @intCast(x)) + dx));
                const ny = @as(usize, @intCast(@as(i32, @intCast(y)) + dy));
                neighborhood[i] = image[ny * @as(usize, @intCast(linesize)) + nx];
            }
            if (hasConsecutivePixels(neighborhood, I_c, threshold, min_consecutive)) {
                try keypoints.append(.{ .x = @floatFromInt(x), .y = @floatFromInt(y) });
            }
        }
    }

    return keypoints;
}

fn hasConsecutivePixels(
    neighborhood: [16]u8,
    I_c: u8,
    threshold: u8,
    min_consecutive: u8,
) bool {
    const threshold_high: u8 = @intCast(@min(@as(u16, @intCast(I_c)) + @as(u16, @intCast(threshold)), 255));
    const threshold_low: u8 = @intCast(@max(@as(i16, @intCast(I_c)) - @as(i16, @intCast(threshold)), 0));

    var doubled_neighborhood = neighborhood ++ neighborhood;
    var count_bright: u8 = 0;
    var count_dark: u8 = 0;

    for (doubled_neighborhood[0..16]) |pixel| {
        if (pixel > threshold_high) {
            count_bright += 1;
            count_dark = 0;
            if (count_bright >= min_consecutive) {
                return true;
            }
        } else if (pixel < threshold_low) {
            count_dark += 1;
            count_bright = 0;
            if (count_dark >= min_consecutive) {
                return true;
            }
        } else {
            count_bright = 0;
            count_dark = 0;
        }
    }
    return false;
}

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

    frame: ?*video.AVFrame,

    pub fn init(allocator: std.mem.Allocator, target_node: *Node) !*Self {
        var self = try allocator.create(KeypointManager);

        self.allocator = std.heap.c_allocator;
        self.arena = std.heap.ArenaAllocator.init(self.allocator);

        self.mutex = std.Thread.Mutex{};

        self.max_keypoints = 5000000;
        self.threshold = 10;

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

        self.frame = video.av_frame_clone(frame);
    }

    // Called from render thread
    pub fn processFrame(self: *Self) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.frame) |frame| {
            const frame_width: u32 = @intCast(frame.width);
            const frame_height: u32 = @intCast(frame.height);

            const num_keypoints = try self.keypoint_detector.?.detectKeypoints(
                frame.data[0],
                frame.data[1],
                frame_width,
                frame_height,
                @as(u32, @intCast(frame.linesize[0])),
                @as(u32, @intCast(frame.linesize[1])),
                self.threshold,
            );

            // Just update the count
            std.debug.print("Detected Keypoint => {}\n", .{num_keypoints});

            for (self.target_node.children.items) |child| {
                child.instance_data.?.count = num_keypoints;
            }

            defer video.av_frame_free(@ptrCast(&self.frame));
        }
    }

    pub fn convertImageToWorldCoords(x: f32, y: f32, imageWidth: f32, imageHeight: f32) [3]f32 {
        // Convert to normalized coordinates (-1 to 1)
        const normalizedX = (x / imageWidth) * 2.0 - 1.0;
        const normalizedY = -((y / imageHeight) * 2.0 - 1.0); // Flip Y axis

        // Scale to world coordinates (TODO: use node dimensions in future instead)
        const worldX = normalizedX * 6.4; // Half of canvas width (12.8/2)
        const worldY = normalizedY * 3.6; // Half of canvas height (7.2/2)

        return [_]f32{ worldX, -0.01, worldY }; // Slight z-offset to avoid z-fighting
    }
};

// Orientation Assignment
// fn assignOrientations(image: []u8, keypoints: []KeyPoint) !void {
//     // Implementation of orientation assignment
// }

// // BRIEF Descriptor Generation
// fn generateBriefDescriptors(image: []u8, keypoints: []KeyPoint) ![]Descriptor {
//     // Implementation of BRIEF descriptor generation
// }

// // Step 4: Feature Matching
// fn matchDescriptors(descriptorsLeft: []Descriptor, descriptorsRight: []Descriptor) ![]Match {
//     // Implementation of feature matching using Hamming distance
// }

// // Step 5: Depth Estimation
// fn computeDepth(matchedPairs: []Match, focalLength: f32, baseline: f32) ![]f32 {
//     // Implementation of depth computation
// }
