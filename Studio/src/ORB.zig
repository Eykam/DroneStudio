const std = @import("std");
const Node = @import("Node.zig");
const KeypointDebugger = @import("Shape.zig").InstancedKeypointDebugger;
const CudaBinds = @import("bindings/cuda.zig");
const KeyPoint = CudaBinds.KeyPoint;

const Frame = struct {
    timestamp: u64,
    width: usize,
    height: usize,
    y: []const u8,
    linesize: usize,
};

const FramePair = struct {
    left: Frame,
    right: Frame,
};

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

    gpa: std.heap.GeneralPurposeAllocator(.{}),
    arena: std.heap.ArenaAllocator,
    allocator: std.mem.Allocator,
    pending_keypoints: ?[]KeypointDebugger.Instance,
    mutex: std.Thread.Mutex,
    target_node: *Node,

    radius: f32 = 1.0,
    resolution: u32 = 1,

    pub fn init(allocator: std.mem.Allocator, target_node: *Node) !*Self {
        var self = try allocator.create(KeypointManager);

        self.gpa = std.heap.GeneralPurposeAllocator(.{}).init;
        self.allocator = self.gpa.allocator();
        self.arena = std.heap.ArenaAllocator.init(self.allocator);
        self.pending_keypoints = null;
        self.mutex = std.Thread.Mutex{};

        self.radius = 0.025;
        self.resolution = 1;

        self.target_node = target_node;

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.pending_keypoints) |keypoints| {
            self.allocator.free(keypoints);
        }

        self.arena.deinit();
    }

    // Called from video thread
    pub fn queueKeypoints(self: *Self, frame_width: usize, frame_height: usize, keypoints: []KeyPoint) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Free any existing pending keypoints
        if (self.pending_keypoints) |old_keypoints| {
            std.debug.print("Old Keypoints detected, clearing array!\n", .{});
            self.allocator.free(old_keypoints);
        }

        // Allocate and fill new keypoints
        var new_keypoints = try self.allocator.alloc(KeypointDebugger.Instance, keypoints.len);
        for (keypoints, 0..) |kp, i| {
            const worldPos = convertImageToWorldCoords(kp.x, kp.y, @floatFromInt(frame_width), @floatFromInt(frame_height));
            new_keypoints[i] = KeypointDebugger.Instance{
                .position = worldPos,
                .color = .{ 1.0, 0.0, 0.0 }, // Default to red
            };
        }

        self.pending_keypoints = new_keypoints;
    }

    // Called from render thread
    pub fn update(self: *Self) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.pending_keypoints) |keypoints| {
            std.debug.print("Pending keypoints: {d} \n", .{keypoints.len});
            for (self.target_node.children.items) |child| {
                child.deinit();
            }

            self.target_node.children.clearAndFree();
            std.debug.print("Children length after cleanup: {d} \n", .{self.target_node.children.items.len});

            _ = self.arena.reset(.free_all);
            const temp_alloc = self.arena.allocator();

            const instancedKeypointNode = try KeypointDebugger.init(
                temp_alloc,
                self.radius,
                self.resolution,
                keypoints,
            );

            try self.target_node.addChild(instancedKeypointNode);
            std.debug.print("Children length: {d} \n", .{self.target_node.children.items.len});

            // Free the processed keypoints
            self.allocator.free(keypoints);
            self.pending_keypoints = null;
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
