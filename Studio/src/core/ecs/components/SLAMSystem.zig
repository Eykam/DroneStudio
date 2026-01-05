const std = @import("std");
const Math = @import("../../Math.zig");
const Vec3 = Math.Vec3;
const Mat4 = Math.Mat4;
const Quaternion = Math.Quaternion;
const Core = @import("../Core.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const Globals = @import("Globals.zig");
const CudaGL = @import("../graphics/CudaGL.zig");

// ============================================================================
// SE3: Rigid body transformation (rotation + translation)
// ============================================================================

/// SE3 represents a rigid body transformation in 3D space.
/// Uses quaternion + translation representation for compactness and to
/// leverage existing Quaternion helpers.
pub const SE3 = struct {
    /// Rotation as unit quaternion
    rotation: Quaternion,
    /// Translation vector
    translation: Vec3,

    const Self = @This();

    /// Identity transformation
    pub const identity = Self{
        .rotation = Quaternion.identity(),
        .translation = Vec3.zero(),
    };

    /// Create SE3 from quaternion and translation
    pub fn init(rotation: Quaternion, translation: Vec3) Self {
        return .{ .rotation = rotation.normalize(), .translation = translation };
    }

    /// Create SE3 from 4x4 transformation matrix
    pub fn fromMat4(m: Mat4) Self {
        return .{
            .rotation = Quaternion.from_mat3(m.to_mat3()),
            .translation = Vec3.init(m.data[12], m.data[13], m.data[14]),
        };
    }

    /// Create SE3 from TransformComponent-style data
    pub fn fromTransform(position: [3]f32, rotation: Quaternion) Self {
        return .{
            .rotation = rotation.normalize(),
            .translation = Vec3.init(position[0], position[1], position[2]),
        };
    }

    /// Convert to 4x4 transformation matrix
    pub fn toMat4(self: Self) Mat4 {
        var m = self.rotation.to_mat4();
        m.data[12] = self.translation.x();
        m.data[13] = self.translation.y();
        m.data[14] = self.translation.z();
        return m;
    }

    /// Compose two SE3 transforms: result = self * other
    /// This applies 'other' first, then 'self'
    pub fn multiply(self: Self, other: Self) Self {
        return .{
            .rotation = self.rotation.multiply(other.rotation).normalize(),
            .translation = self.transformPoint(other.translation),
        };
    }

    /// Compute inverse transformation
    pub fn inverse(self: Self) Self {
        const r_inv = self.rotation.conjugate();
        return .{
            .rotation = r_inv,
            .translation = self.translation.rotate_by_quaternion(r_inv).scale(-1),
        };
    }

    /// Transform a 3D point: R * p + t
    pub fn transformPoint(self: Self, p: Vec3) Vec3 {
        return p.rotate_by_quaternion(self.rotation).add(self.translation);
    }

    /// Rotate a vector (without translation)
    pub fn rotateVector(self: Self, v: Vec3) Vec3 {
        return v.rotate_by_quaternion(self.rotation);
    }

    /// Compute relative transformation: result = self.inverse() * other
    /// Returns the transformation from self's frame to other's frame
    pub fn relativeTo(self: Self, other: Self) Self {
        return self.inverse().multiply(other);
    }

    /// Interpolate between two SE3 transforms (slerp for rotation, lerp for translation)
    pub fn lerp(a: Self, b: Self, t: f32) Self {
        return .{
            .rotation = Quaternion.slerp(a.rotation, b.rotation, t),
            .translation = Vec3.lerp(a.translation, b.translation, t),
        };
    }

    /// Get translation distance between two poses
    pub fn translationDistance(self: Self, other: Self) f32 {
        return self.translation.sub(other.translation).length();
    }

    /// Get rotation angle difference between two poses (in radians)
    pub fn rotationDistance(self: Self, other: Self) f32 {
        const q_diff = self.rotation.multiply(other.rotation.conjugate());
        return 2.0 * std.math.acos(Math.clamp(@abs(q_diff.w()), 0.0, 1.0));
    }
};

// ============================================================================
// SLAMConfig: Configuration parameters
// ============================================================================

pub const SLAMConfig = struct {
    // Feature detection
    fast_threshold: u8 = 20,
    max_features_per_frame: u32 = 2000,
    grid_cells_x: u32 = 8,
    grid_cells_y: u32 = 6,
    min_features_per_cell: u32 = 5,

    // Stereo matching
    stereo_match_threshold: u32 = 50, // Hamming distance threshold
    min_disparity: f32 = 1.0,
    max_disparity: f32 = 128.0,
    epipolar_threshold: f32 = 1.5, // pixels

    // Temporal matching (frame-to-frame)
    temporal_match_threshold: u32 = 64,
    temporal_search_radius: u32 = 50, // pixels

    // Motion estimation (RANSAC)
    ransac_iterations: u32 = 200,
    ransac_threshold: f32 = 2.0, // reprojection error in pixels
    min_inliers: u32 = 20,

    // Keyframe selection
    keyframe_translation_threshold: f32 = 0.3, // meters
    keyframe_rotation_threshold: f32 = 10.0, // degrees
    keyframe_min_features: u32 = 100,
    keyframe_min_tracked_ratio: f32 = 0.5,

    // Landmark management
    min_observations_for_landmark: u32 = 3,
    landmark_max_reprojection_error: f32 = 3.0, // pixels
    landmark_cull_age: u32 = 5, // keyframes

    // Loop closure
    loop_closure_enabled: bool = true,
    loop_closure_min_keyframe_gap: u32 = 20,
    loop_closure_distance_threshold: f32 = 2.0, // meters
    loop_closure_min_matches: u32 = 30,

    // Bundle adjustment
    local_ba_enabled: bool = true,
    local_ba_window_size: u32 = 10, // keyframes
    full_ba_on_loop_closure: bool = true,
    ba_iterations: u32 = 10,

    // Camera intrinsics (will be populated from SensorCamera)
    fx: f32 = 619.5, // focal length x in pixels (default for 1280x720)
    fy: f32 = 619.5, // focal length y in pixels
    cx: f32 = 640.0, // principal point x
    cy: f32 = 360.0, // principal point y
    baseline: f32 = 0.075, // stereo baseline in meters (75mm)
    image_width: u32 = 1280,
    image_height: u32 = 720,

    const Self = @This();

    /// Initialize config with camera intrinsics
    /// fx, fy: focal lengths in pixels (from CameraModule.fx/fy)
    /// baseline_m: stereo baseline in meters
    pub fn initWithIntrinsics(
        width: u32,
        height: u32,
        fx_pixels: f32,
        fy_pixels: f32,
        baseline_m: f32,
    ) Self {
        return .{
            .image_width = width,
            .image_height = height,
            .fx = fx_pixels,
            .fy = fy_pixels,
            .cx = @as(f32, @floatFromInt(width)) / 2.0,
            .cy = @as(f32, @floatFromInt(height)) / 2.0,
            .baseline = baseline_m,
        };
    }

    /// Compute depth from disparity: Z = baseline * fx / disparity
    pub fn disparityToDepth(self: Self, disparity: f32) f32 {
        if (disparity < self.min_disparity) return std.math.inf(f32);
        return self.baseline * self.fx / disparity;
    }

    /// Compute disparity from depth
    pub fn depthToDisparity(self: Self, depth: f32) f32 {
        if (depth <= 0) return self.max_disparity;
        return self.baseline * self.fx / depth;
    }

    /// Project 3D point to image coordinates
    pub fn project(self: Self, p: Vec3) ?[2]f32 {
        if (p.z() <= 0) return null;
        const u = self.fx * p.x() / p.z() + self.cx;
        const v = self.fy * p.y() / p.z() + self.cy;
        if (u < 0 or u >= @as(f32, @floatFromInt(self.image_width)) or
            v < 0 or v >= @as(f32, @floatFromInt(self.image_height)))
        {
            return null;
        }
        return .{ u, v };
    }

    /// Unproject image point to 3D ray (normalized)
    pub fn unproject(self: Self, u: f32, v: f32) Vec3 {
        return Vec3.init(
            (u - self.cx) / self.fx,
            (v - self.cy) / self.fy,
            1.0,
        ).normalize();
    }

    /// Triangulate 3D point from stereo correspondence
    pub fn triangulate(self: Self, u_left: f32, v_left: f32, disparity: f32) ?Vec3 {
        const depth = self.disparityToDepth(disparity);
        if (depth == std.math.inf(f32) or depth > 50.0) return null; // Max range 50m

        return Vec3.init(
            (u_left - self.cx) * depth / self.fx,
            (v_left - self.cy) * depth / self.fy,
            depth,
        );
    }
};

// ============================================================================
// Feature: Single feature observation
// ============================================================================

pub const Feature = struct {
    x: f32, // image x coordinate
    y: f32, // image y coordinate
    response: f32, // corner response/score
    octave: u8, // scale octave
    descriptor: [32]u8, // BRIEF descriptor (256 bits)
};

// ============================================================================
// Observation: Link between feature and landmark
// ============================================================================

pub const Observation = struct {
    keyframe_id: u64,
    feature_idx: u32,
    landmark_id: u64,
};

// ============================================================================
// Keyframe: Camera view with features
// ============================================================================

pub const Keyframe = struct {
    id: u64,
    timestamp: i64, // microseconds
    pose: SE3, // Camera pose in world frame

    // Features (CPU copy - GPU resources managed separately)
    features_left: std.ArrayList(Feature),
    features_right: std.ArrayList(Feature),

    // Stereo matches: index i in features_left matches stereo_matches[i] in features_right (-1 if no match)
    stereo_matches: std.ArrayList(i32),
    disparities: std.ArrayList(f32),

    // Landmark observations: which landmarks are observed by which features
    observations: std.ArrayList(Observation),

    // Covisibility: other keyframes that share landmarks with this one
    covisible_keyframes: std.AutoHashMap(u64, u32), // keyframe_id -> shared landmark count

    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, id: u64, timestamp: i64, pose: SE3) !*Self {
        const kf = try allocator.create(Self);
        kf.* = .{
            .id = id,
            .timestamp = timestamp,
            .pose = pose,
            .features_left = std.ArrayList(Feature).init(allocator),
            .features_right = std.ArrayList(Feature).init(allocator),
            .stereo_matches = std.ArrayList(i32).init(allocator),
            .disparities = std.ArrayList(f32).init(allocator),
            .observations = std.ArrayList(Observation).init(allocator),
            .covisible_keyframes = std.AutoHashMap(u64, u32).init(allocator),
            .allocator = allocator,
        };
        return kf;
    }

    pub fn deinit(self: *Self) void {
        self.features_left.deinit();
        self.features_right.deinit();
        self.stereo_matches.deinit();
        self.disparities.deinit();
        self.observations.deinit();
        self.covisible_keyframes.deinit();
        self.allocator.destroy(self);
    }

    /// Get 3D position of a feature using stereo disparity
    pub fn getFeaturePosition(self: *const Self, feature_idx: u32, config: SLAMConfig) ?Vec3 {
        if (feature_idx >= self.features_left.items.len) return null;
        if (feature_idx >= self.stereo_matches.items.len) return null;

        const match_idx = self.stereo_matches.items[feature_idx];
        if (match_idx < 0) return null;

        const disparity = self.disparities.items[feature_idx];
        const feat = self.features_left.items[feature_idx];

        // Triangulate in camera frame
        const p_cam = config.triangulate(feat.x, feat.y, disparity) orelse return null;

        // Transform to world frame
        return self.pose.transformPoint(p_cam);
    }

    /// Add observation linking feature to landmark
    pub fn addObservation(self: *Self, feature_idx: u32, landmark_id: u64) !void {
        try self.observations.append(.{
            .keyframe_id = self.id,
            .feature_idx = feature_idx,
            .landmark_id = landmark_id,
        });
    }

    /// Update covisibility with another keyframe
    pub fn updateCovisibility(self: *Self, other_id: u64, shared_count: u32) !void {
        try self.covisible_keyframes.put(other_id, shared_count);
    }
};

// ============================================================================
// Landmark: 3D map point
// ============================================================================

pub const Landmark = struct {
    id: u64,
    position: Vec3, // 3D position in world frame
    normal: Vec3, // Mean viewing direction
    descriptor: [32]u8, // Representative descriptor

    // Observations
    observations: std.ArrayList(Observation),

    // Statistics
    created_keyframe_id: u64,
    last_observed_keyframe_id: u64,
    observation_count: u32,
    inlier_count: u32,
    outlier_count: u32,

    // State
    is_bad: bool,

    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, id: u64, position: Vec3, descriptor: [32]u8, keyframe_id: u64) !*Self {
        const lm = try allocator.create(Self);
        lm.* = .{
            .id = id,
            .position = position,
            .normal = Vec3.init(0, 0, -1), // Default viewing direction
            .descriptor = descriptor,
            .observations = std.ArrayList(Observation).init(allocator),
            .created_keyframe_id = keyframe_id,
            .last_observed_keyframe_id = keyframe_id,
            .observation_count = 0,
            .inlier_count = 0,
            .outlier_count = 0,
            .is_bad = false,
            .allocator = allocator,
        };
        return lm;
    }

    pub fn deinit(self: *Self) void {
        self.observations.deinit();
        self.allocator.destroy(self);
    }

    /// Add observation from a keyframe
    pub fn addObservation(self: *Self, keyframe_id: u64, feature_idx: u32) !void {
        try self.observations.append(.{
            .keyframe_id = keyframe_id,
            .feature_idx = feature_idx,
            .landmark_id = self.id,
        });
        self.observation_count += 1;
        self.last_observed_keyframe_id = keyframe_id;
    }

    /// Update mean viewing direction
    pub fn updateNormal(self: *Self, camera_position: Vec3) void {
        const view_dir = self.position.sub(camera_position).normalize();
        // Running average
        const n = @as(f32, @floatFromInt(self.observation_count));
        self.normal = self.normal.scale((n - 1.0) / n).add(view_dir.scale(1.0 / n)).normalize();
    }

    /// Mark as bad (should be culled)
    pub fn markBad(self: *Self) void {
        self.is_bad = true;
    }

    /// Check if landmark is good quality
    pub fn isGood(self: *const Self, min_observations: u32) bool {
        if (self.is_bad) return false;
        if (self.observation_count < min_observations) return false;
        // Check inlier ratio
        const total = self.inlier_count + self.outlier_count;
        if (total > 0) {
            const ratio = @as(f32, @floatFromInt(self.inlier_count)) / @as(f32, @floatFromInt(total));
            if (ratio < 0.25) return false;
        }
        return true;
    }
};

// ============================================================================
// PoseGraphEdge: Constraint between keyframes
// ============================================================================

pub const PoseGraphEdge = struct {
    from_id: u64,
    to_id: u64,
    relative_pose: SE3,
    information: [6]f32, // Diagonal of information matrix (6 DOF)
    is_loop_closure: bool,
};

// ============================================================================
// SLAMMap: Complete map storage
// ============================================================================

pub const SLAMMap = struct {
    pub const KeyframePosition = struct { id: u64, pos: Vec3 };

    keyframes: std.AutoHashMap(u64, *Keyframe),
    landmarks: std.AutoHashMap(u64, *Landmark),
    pose_graph_edges: std.ArrayList(PoseGraphEdge),

    next_keyframe_id: u64,
    next_landmark_id: u64,

    // Spatial index for loop closure (simple grid-based)
    keyframe_positions: std.ArrayList(KeyframePosition),

    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .keyframes = std.AutoHashMap(u64, *Keyframe).init(allocator),
            .landmarks = std.AutoHashMap(u64, *Landmark).init(allocator),
            .pose_graph_edges = std.ArrayList(PoseGraphEdge).init(allocator),
            .next_keyframe_id = 0,
            .next_landmark_id = 0,
            .keyframe_positions = std.ArrayList(KeyframePosition).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        // Free all keyframes
        var kf_iter = self.keyframes.valueIterator();
        while (kf_iter.next()) |kf| {
            kf.*.deinit();
        }
        self.keyframes.deinit();

        // Free all landmarks
        var lm_iter = self.landmarks.valueIterator();
        while (lm_iter.next()) |lm| {
            lm.*.deinit();
        }
        self.landmarks.deinit();

        self.pose_graph_edges.deinit();
        self.keyframe_positions.deinit();
    }

    /// Create a new keyframe
    pub fn createKeyframe(self: *Self, timestamp: i64, pose: SE3) !*Keyframe {
        const id = self.next_keyframe_id;
        self.next_keyframe_id += 1;

        const kf = try Keyframe.init(self.allocator, id, timestamp, pose);
        try self.keyframes.put(id, kf);
        try self.keyframe_positions.append(.{ .id = id, .pos = pose.translation });

        return kf;
    }

    /// Create a new landmark
    pub fn createLandmark(self: *Self, position: Vec3, descriptor: [32]u8, keyframe_id: u64) !*Landmark {
        const id = self.next_landmark_id;
        self.next_landmark_id += 1;

        const lm = try Landmark.init(self.allocator, id, position, descriptor, keyframe_id);
        try self.landmarks.put(id, lm);

        return lm;
    }

    /// Add pose graph edge
    pub fn addEdge(self: *Self, from_id: u64, to_id: u64, relative_pose: SE3, is_loop_closure: bool) !void {
        try self.pose_graph_edges.append(.{
            .from_id = from_id,
            .to_id = to_id,
            .relative_pose = relative_pose,
            .information = .{ 1, 1, 1, 1, 1, 1 }, // Default uniform information
            .is_loop_closure = is_loop_closure,
        });
    }

    /// Get keyframe by ID
    pub fn getKeyframe(self: *Self, id: u64) ?*Keyframe {
        return self.keyframes.get(id);
    }

    /// Get landmark by ID
    pub fn getLandmark(self: *Self, id: u64) ?*Landmark {
        return self.landmarks.get(id);
    }

    /// Find keyframes within distance of a position (for loop closure)
    pub fn findNearbyKeyframes(self: *Self, position: Vec3, max_distance: f32, exclude_recent: u32) !std.ArrayList(u64) {
        var result = std.ArrayList(u64).init(self.allocator);
        const max_dist_sq = max_distance * max_distance;
        const min_id = if (self.next_keyframe_id > exclude_recent) self.next_keyframe_id - exclude_recent else 0;

        for (self.keyframe_positions.items) |entry| {
            if (entry.id >= min_id) continue; // Skip recent keyframes
            const dist_sq = position.sub(entry.pos).length_squared();
            if (dist_sq <= max_dist_sq) {
                try result.append(entry.id);
            }
        }

        return result;
    }

    /// Get statistics
    pub fn getStats(self: *const Self) struct { keyframes: usize, landmarks: usize, edges: usize } {
        return .{
            .keyframes = self.keyframes.count(),
            .landmarks = self.landmarks.count(),
            .edges = self.pose_graph_edges.items.len,
        };
    }

    /// Cull bad landmarks
    pub fn cullBadLandmarks(self: *Self) void {
        var to_remove = std.ArrayList(u64).init(self.allocator);
        defer to_remove.deinit();

        var iter = self.landmarks.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.*.is_bad) {
                to_remove.append(entry.key_ptr.*) catch continue;
            }
        }

        for (to_remove.items) |id| {
            if (self.landmarks.get(id)) |lm| {
                lm.deinit();
            }
            _ = self.landmarks.remove(id);
        }
    }
};

// ============================================================================
// TrackingState: Current tracking status
// ============================================================================

pub const TrackingState = enum {
    not_initialized,
    tracking,
    lost,
    relocalization,
};

// ============================================================================
// SLAMComponent: ECS component for SLAM state
// ============================================================================

const Viewports = @import("Viewports.zig");
const ViewportComponent = Viewports.ViewportComponent;
const Transform = @import("Transform.zig");
const TransformComponent = Transform.TransformComponent;
const LandmarkCloud = @import("../prefabs/LandmarkCloud.zig");

pub const SLAMComponent = struct {
    entity_id: ?Core.EntityID = null,
    allocator: std.mem.Allocator,

    // Owned processor (runs async SLAM pipeline)
    processor: *SLAMProcessor,

    // Viewport references for stereo frame acquisition
    left_viewport: *ViewportComponent,
    right_viewport: *ViewportComponent,

    // Optional ground truth source
    ground_truth_transform: ?*TransformComponent = null,

    // Landmark visualization
    landmark_mesh_name: [:0]const u8 = "slam_landmarks",
    landmarks_initialized: bool = false,

    // State
    enabled: bool = false,
    frame_counter: u64 = 0,

    // Cached output (updated from processor)
    tracking_state: TrackingState = .not_initialized,
    current_pose: SE3 = SE3.identity,
    stats: SLAMStats = .{},

    // Ground truth comparison
    ground_truth_available: bool = false,
    pose_error_translation: f32 = 0,
    pose_error_rotation: f32 = 0,

    const Self = @This();

    pub const SLAMStats = struct {
        tracked_features: u32 = 0,
        stereo_matches: u32 = 0,
        temporal_matches: u32 = 0,
        inliers: u32 = 0,
        processing_time_us: u64 = 0,
        keyframes: u32 = 0,
        landmarks: u32 = 0,
    };

    pub const InitOptions = struct {
        config: SLAMConfig = .{},
        left_viewport: *ViewportComponent,
        right_viewport: *ViewportComponent,
        ground_truth_transform: ?*TransformComponent = null,
    };

    pub fn init(allocator: std.mem.Allocator, options: InitOptions) !Self {
        const processor = try SLAMProcessor.init(allocator, options.config);
        return .{
            .allocator = allocator,
            .processor = processor,
            .left_viewport = options.left_viewport,
            .right_viewport = options.right_viewport,
            .ground_truth_transform = options.ground_truth_transform,
        };
    }

    pub fn deinit(self: *Self) void {
        self.processor.deinit();
    }

    pub fn attach(self: *Self, ecs: anytype, eid: Core.EntityID) !void {
        self.entity_id = eid;
        try ecs.slam_components.add(eid, self.*);
    }

    /// Enable/disable SLAM processing
    pub fn setEnabled(self: *Self, enabled: bool) !void {
        self.enabled = enabled;
        if (enabled) {
            try self.processor.start();
        } else {
            self.processor.stop();
        }
    }

    /// Capture frame and submit to processor (called by SLAMSystem on main thread)
    pub fn captureFrame(self: *Self, frame_count: u32) void {
        if (!self.enabled) return;
        if (!cuda_config.CUDA_ENABLED) return;

        // Ensure sharing is enabled on viewports
        if (self.left_viewport.shared_info == null) self.left_viewport.enableSharing();
        if (self.right_viewport.shared_info == null) self.right_viewport.enableSharing();

        // Copy GL textures to CUDA buffers (must be on main/render thread)
        const left_ptr = if (self.left_viewport.shared_info) |*info| info else return;
        const right_ptr = if (self.right_viewport.shared_info) |*info| info else return;
        _ = left_ptr.copyFromGL(frame_count);
        _ = right_ptr.copyFromGL(frame_count);

        // Get ground truth pose from transform if available
        var ground_truth: ?SE3 = null;
        if (self.ground_truth_transform) |tf| {
            ground_truth = SE3.fromTransform(tf.position, tf.rotation);
        }

        // Acquire buffers for processing - this swaps the double buffers so SLAM gets
        // a stable buffer while main thread continues writing to the other buffer
        const left_buffer = left_ptr.acquireForProcessing();
        const right_buffer = right_ptr.acquireForProcessing();

        // Submit frame to processor with acquired buffer pointers
        self.processor.submitFrame(.{
            .timestamp = std.time.milliTimestamp(),
            .left_rgba_ptr = left_buffer,
            .right_rgba_ptr = right_buffer,
            .width = @intCast(left_ptr.width),
            .height = @intCast(left_ptr.height),
            .rgba_pitch = @intCast(left_ptr.pitch),
            .ground_truth_pose = ground_truth,
        });

        self.frame_counter += 1;

        // Update cached output from processor
        self.syncFromProcessor();
    }

    /// Sync cached state from processor output
    pub fn syncFromProcessor(self: *Self) void {
        const output = self.processor.getOutput();
        self.current_pose = output.pose;
        self.stats = .{
            .tracked_features = output.stats.tracked_features,
            .stereo_matches = output.stats.stereo_matches,
            .temporal_matches = output.stats.temporal_matches,
            .inliers = output.stats.inliers,
            .processing_time_us = output.stats.processing_time_us,
            .keyframes = output.stats.keyframes,
            .landmarks = output.stats.landmarks,
        };
        self.tracking_state = self.processor.getTrackingState();

        // Update ground truth error if available
        if (self.ground_truth_transform) |tf| {
            const gt = SE3.fromTransform(tf.position, tf.rotation);
            self.updateError(gt);
        }
    }

    /// Update pose error from ground truth
    fn updateError(self: *Self, ground_truth: SE3) void {
        self.ground_truth_available = true;
        self.pose_error_translation = self.current_pose.translation.sub(ground_truth.translation).length();
        const q_diff = self.current_pose.rotation.multiply(ground_truth.rotation.conjugate());
        self.pose_error_rotation = Math.degrees(2.0 * std.math.acos(Math.clamp(@abs(q_diff.w()), -1.0, 1.0)));
    }

    pub fn reset(self: *Self) void {
        self.tracking_state = .not_initialized;
        self.current_pose = SE3.identity;
        self.stats = .{};
        self.frame_counter = 0;
        self.pose_error_translation = 0;
        self.pose_error_rotation = 0;
        self.ground_truth_available = false;
    }
};

// ============================================================================
// SLAMSystem: ECS system that drives SLAM frame capture on main thread
// ============================================================================

const ECSManager = @import("../ECSManager.zig");

pub const SLAMSystem = struct {
    pub const MAX_LANDMARKS = 2500000;

    allocator: std.mem.Allocator,
    slam_components: *SparseSet(SLAMComponent),
    globals: *Globals.GlobalsComponent,
    ecs: *ECSManager,

    // Pre-allocated buffers for landmark visualization
    landmark_positions: []align(16) [4]f32,
    landmark_colors: []align(16) [4]f32,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        slam_components: *SparseSet(SLAMComponent),
        globals: *Globals.GlobalsComponent,
        ecs: *ECSManager,
    ) !Self {
        return .{
            .allocator = allocator,
            .slam_components = slam_components,
            .globals = globals,
            .ecs = ecs,
            .landmark_positions = try allocator.alignedAlloc([4]f32, 16, MAX_LANDMARKS),
            .landmark_colors = try allocator.alignedAlloc([4]f32, 16, MAX_LANDMARKS),
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.landmark_positions);
        self.allocator.free(self.landmark_colors);

        // Cleanup all SLAM components
        var it = self.slam_components.iterator();
        while (it.next()) |entry| {
            entry.component.deinit();
        }
    }

    /// Called each frame after rendering to capture and process SLAM frames
    pub fn update(self: *Self) void {
        var it = self.slam_components.iterator();
        while (it.next()) |entry| {
            const component = entry.component;

            // Initialize landmark visualization if not done
            if (component.enabled and !component.landmarks_initialized) {
                self.initLandmarks(component);
            }

            // Capture and process frame
            component.captureFrame(self.globals.frame_count);

            // Update landmark visualization
            if (component.enabled and component.landmarks_initialized) {
                self.updateLandmarks(component);
            }
        }
    }

    /// Initialize the landmark cloud mesh
    fn initLandmarks(self: *Self, component: *SLAMComponent) void {
        _ = LandmarkCloud.spawn(
            self.allocator,
            self.ecs,
            component.landmark_mesh_name,
            MAX_LANDMARKS,
        ) catch |err| {
            std.debug.print("Failed to spawn landmark cloud: {}\n", .{err});
            return;
        };

        component.landmarks_initialized = true;
    }

    /// Update landmark positions from the SLAM map
    fn updateLandmarks(self: *Self, component: *SLAMComponent) void {
        const mesh_resource = self.ecs.world.resource_manager.meshes.getPtr(component.landmark_mesh_name) orelse return;

        // Lock processor for thread-safe access to map
        component.processor.output_mutex.lock();
        defer component.processor.output_mutex.unlock();

        const landmarks = &component.processor.map.landmarks;
        const count = @min(landmarks.count(), MAX_LANDMARKS);

        if (count == 0) return;

        var idx: usize = 0;
        var lm_iter = landmarks.valueIterator();
        while (lm_iter.next()) |lm_ptr| {
            if (idx >= count) break;
            const lm = lm_ptr.*;

            if (lm.is_bad) continue;

            // Position (xyz) + point size (w)
            self.landmark_positions[idx] = .{
                lm.position.x(),
                lm.position.y(),
                lm.position.z(),
                3.0,
            };

            // Color based on observation count
            const brightness = @min(1.0, @as(f32, @floatFromInt(lm.observation_count)) / 10.0);
            self.landmark_colors[idx] = .{ 0.2, 0.5 + brightness * 0.5, 0.2, 1.0 };

            idx += 1;
        }

        if (idx > 0) {
            mesh_resource.updateInstancePositions(self.landmark_positions[0..idx]);
            mesh_resource.updateInstanceColors(self.landmark_colors[0..idx]);
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "SE3 identity" {
    const se3 = SE3.identity;
    const p = Vec3.init(1, 2, 3);
    const result = se3.transformPoint(p);
    try std.testing.expectApproxEqAbs(result.x(), 1.0, 0.001);
    try std.testing.expectApproxEqAbs(result.y(), 2.0, 0.001);
    try std.testing.expectApproxEqAbs(result.z(), 3.0, 0.001);
}

test "SE3 inverse" {
    const t = Vec3.init(1, 2, 3);
    const se3 = SE3.init(Quaternion.identity(), t);
    const inv = se3.inverse();
    const composed = se3.multiply(inv);

    try std.testing.expectApproxEqAbs(composed.translation.x(), 0.0, 0.001);
    try std.testing.expectApproxEqAbs(composed.translation.y(), 0.0, 0.001);
    try std.testing.expectApproxEqAbs(composed.translation.z(), 0.0, 0.001);
}

test "SLAMConfig projection" {
    const config = SLAMConfig{};
    const p = Vec3.init(0, 0, 1); // 1 meter in front
    const uv = config.project(p);
    try std.testing.expect(uv != null);
    try std.testing.expectApproxEqAbs(uv.?[0], config.cx, 0.001);
    try std.testing.expectApproxEqAbs(uv.?[1], config.cy, 0.001);
}

// ============================================================================
// SLAMGPUResources: CUDA memory for SLAM processing
// ============================================================================

const cuda_config = @import("cuda_config");
const c = @import("../../bindings/c.zig");
const cuda = c.cuda;

pub const SLAMGPUResources = struct {
    // Frame buffers (grayscale)
    d_left_gray: ?[*]u8 = null,
    d_right_gray: ?[*]u8 = null,
    d_left_blurred: ?[*]u8 = null,
    d_right_blurred: ?[*]u8 = null,

    // Feature storage
    d_left_positions: ?[*]cuda.float4 = null,
    d_left_colors: ?[*]cuda.float4 = null,
    d_left_descriptors: ?[*]cuda.BRIEFDescriptor = null,
    d_left_count: ?*c_uint = null,
    d_right_positions: ?[*]cuda.float4 = null,
    d_right_colors: ?[*]cuda.float4 = null,
    d_right_descriptors: ?[*]cuda.BRIEFDescriptor = null,
    d_right_count: ?*c_uint = null,

    // Stereo matching
    d_stereo_matches_lr: ?[*]cuda.BestMatch = null,
    d_stereo_matches_rl: ?[*]cuda.BestMatch = null,
    d_matched_keypoints: ?[*]cuda.MatchedKeypoint = null,
    d_match_count: ?*c_uint = null,

    // Temporal matching
    d_prev_matched_keypoints: ?[*]cuda.MatchedKeypoint = null,
    d_temporal_curr_to_prev: ?[*]cuda.BestMatch = null,
    d_temporal_prev_to_curr: ?[*]cuda.BestMatch = null,
    d_temporal_matches: ?[*]cuda.TemporalMatch = null,
    d_temporal_match_count: ?*c_uint = null,

    // Motion estimation
    d_best_pose: ?*cuda.CameraPose = null,
    d_inlier_count: ?*c_uint = null,

    // Dimensions
    width: u32 = 0,
    height: u32 = 0,
    max_features: u32 = 0,

    // Previous frame data
    prev_match_count: u32 = 0,

    const Self = @This();

    fn checkCudaAlloc(err: c_uint, name: []const u8) void {
        if (err != cuda.cudaSuccess) {
            std.debug.print("SLAM GPU: cudaMalloc failed for {s}: {s}\n", .{
                name,
                cuda.cudaGetErrorString(err),
            });
            @panic("SLAM GPU allocation failed");
        }
    }

    pub fn init(width: u32, height: u32, max_features: u32) !Self {
        if (!cuda_config.CUDA_ENABLED) {
            return Self{ .width = width, .height = height, .max_features = max_features };
        }

        // Ensure CUDA context is initialized before any allocations
        const set_err = cuda.cudaSetDevice(0);
        if (set_err != cuda.cudaSuccess) {
            std.debug.print("SLAMGPUResources.init: cudaSetDevice failed: {s}\n", .{cuda.cudaGetErrorString(set_err)});
            return error.CudaInitFailed;
        }

        std.debug.print("SLAMGPUResources.init: width={d} height={d} max_features={d}\n", .{
            width,
            height,
            max_features,
        });

        var self = Self{
            .width = width,
            .height = height,
            .max_features = max_features,
        };

        const frame_size: usize = @as(usize, width) * @as(usize, height);
        const feature_size: usize = @as(usize, max_features);

        // Allocate frame buffers - panic immediately on any failure
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_left_gray), frame_size), "d_left_gray");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_right_gray), frame_size), "d_right_gray");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_left_blurred), frame_size), "d_left_blurred");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_right_blurred), frame_size), "d_right_blurred");

        // Allocate feature storage
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_left_positions), feature_size * @sizeOf(cuda.float4)), "d_left_positions");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_left_colors), feature_size * @sizeOf(cuda.float4)), "d_left_colors");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_left_descriptors), feature_size * @sizeOf(cuda.BRIEFDescriptor)), "d_left_descriptors");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_left_count), @sizeOf(c_uint)), "d_left_count");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_right_positions), feature_size * @sizeOf(cuda.float4)), "d_right_positions");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_right_colors), feature_size * @sizeOf(cuda.float4)), "d_right_colors");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_right_descriptors), feature_size * @sizeOf(cuda.BRIEFDescriptor)), "d_right_descriptors");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_right_count), @sizeOf(c_uint)), "d_right_count");

        // Allocate stereo matching buffers
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_stereo_matches_lr), feature_size * @sizeOf(cuda.BestMatch)), "d_stereo_matches_lr");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_stereo_matches_rl), feature_size * @sizeOf(cuda.BestMatch)), "d_stereo_matches_rl");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_matched_keypoints), feature_size * @sizeOf(cuda.MatchedKeypoint)), "d_matched_keypoints");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_match_count), @sizeOf(c_uint)), "d_match_count");

        // Allocate temporal matching buffers
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_prev_matched_keypoints), feature_size * @sizeOf(cuda.MatchedKeypoint)), "d_prev_matched_keypoints");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_temporal_curr_to_prev), feature_size * @sizeOf(cuda.BestMatch)), "d_temporal_curr_to_prev");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_temporal_prev_to_curr), feature_size * @sizeOf(cuda.BestMatch)), "d_temporal_prev_to_curr");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_temporal_matches), feature_size * @sizeOf(cuda.TemporalMatch)), "d_temporal_matches");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_temporal_match_count), @sizeOf(c_uint)), "d_temporal_match_count");

        // Allocate motion estimation buffers
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_best_pose), @sizeOf(cuda.CameraPose)), "d_best_pose");
        checkCudaAlloc(cuda.cudaMalloc(@ptrCast(&self.d_inlier_count), @sizeOf(c_uint)), "d_inlier_count");

        std.debug.print("SLAMGPUResources.init: all allocations succeeded\n", .{});

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (!cuda_config.CUDA_ENABLED) return;

        // Free frame buffers
        if (self.d_left_gray) |p| _ = cuda.cudaFree(p);
        if (self.d_right_gray) |p| _ = cuda.cudaFree(p);
        if (self.d_left_blurred) |p| _ = cuda.cudaFree(p);
        if (self.d_right_blurred) |p| _ = cuda.cudaFree(p);

        // Free feature storage
        if (self.d_left_positions) |p| _ = cuda.cudaFree(p);
        if (self.d_left_colors) |p| _ = cuda.cudaFree(p);
        if (self.d_left_descriptors) |p| _ = cuda.cudaFree(p);
        if (self.d_left_count) |p| _ = cuda.cudaFree(p);
        if (self.d_right_positions) |p| _ = cuda.cudaFree(p);
        if (self.d_right_colors) |p| _ = cuda.cudaFree(p);
        if (self.d_right_descriptors) |p| _ = cuda.cudaFree(p);
        if (self.d_right_count) |p| _ = cuda.cudaFree(p);

        // Free stereo matching
        if (self.d_stereo_matches_lr) |p| _ = cuda.cudaFree(p);
        if (self.d_stereo_matches_rl) |p| _ = cuda.cudaFree(p);
        if (self.d_matched_keypoints) |p| _ = cuda.cudaFree(p);
        if (self.d_match_count) |p| _ = cuda.cudaFree(p);

        // Free temporal matching
        if (self.d_prev_matched_keypoints) |p| _ = cuda.cudaFree(p);
        if (self.d_temporal_curr_to_prev) |p| _ = cuda.cudaFree(p);
        if (self.d_temporal_prev_to_curr) |p| _ = cuda.cudaFree(p);
        if (self.d_temporal_matches) |p| _ = cuda.cudaFree(p);
        if (self.d_temporal_match_count) |p| _ = cuda.cudaFree(p);

        // Free motion estimation
        if (self.d_best_pose) |p| _ = cuda.cudaFree(p);
        if (self.d_inlier_count) |p| _ = cuda.cudaFree(p);

        self.* = .{};
    }

    /// Swap current frame data to previous frame for next iteration
    pub fn swapFrames(self: *Self) void {
        if (!cuda_config.CUDA_ENABLED) return;

        // Copy current matched keypoints to previous buffer
        if (self.d_matched_keypoints != null and self.d_prev_matched_keypoints != null) {
            const err = cuda.cudaMemcpy(
                self.d_prev_matched_keypoints,
                self.d_matched_keypoints,
                self.max_features * @sizeOf(cuda.MatchedKeypoint),
                cuda.cudaMemcpyDeviceToDevice,
            );
            if (err != cuda.cudaSuccess) {
                std.debug.print("SLAM swapFrames cudaMemcpy failed: {s}\n", .{cuda.cudaGetErrorString(err)});
            }
        }
    }
};

// ============================================================================
// SLAMCPUResources: Pre-allocated CPU buffers for SLAM processing
// ============================================================================

pub const SLAMCPUResources = struct {
    /// CPU-side representation of a matched stereo keypoint
    pub const MatchedKeypointCPU = struct {
        left_pos: [2]f32,
        right_pos: [2]f32,
        world_pos: Vec3,
        descriptor: [32]u8,
        disparity: f32,
    };

    // Pre-allocated buffers (sized to max_features)
    curr_matched_keypoints: []MatchedKeypointCPU,
    prev_matched_keypoints: []MatchedKeypointCPU,
    curr_landmark_ids: []?u64,
    prev_landmark_ids: []?u64,

    // Temp buffers for GPU readback
    gpu_matched_keypoints: []cuda.MatchedKeypoint,
    gpu_temporal_matches: []cuda.TemporalMatch,
    gpu_best_matches: []cuda.BestMatch,

    // Counts
    curr_match_count: u32 = 0,
    prev_match_count: u32 = 0,
    max_features: u32,

    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, max_features: u32) !Self {
        return .{
            .allocator = allocator,
            .max_features = max_features,
            .curr_matched_keypoints = try allocator.alloc(MatchedKeypointCPU, max_features),
            .prev_matched_keypoints = try allocator.alloc(MatchedKeypointCPU, max_features),
            .curr_landmark_ids = try allocator.alloc(?u64, max_features),
            .prev_landmark_ids = try allocator.alloc(?u64, max_features),
            .gpu_matched_keypoints = try allocator.alloc(cuda.MatchedKeypoint, max_features),
            .gpu_temporal_matches = try allocator.alloc(cuda.TemporalMatch, max_features),
            .gpu_best_matches = try allocator.alloc(cuda.BestMatch, max_features),
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.curr_matched_keypoints);
        self.allocator.free(self.prev_matched_keypoints);
        self.allocator.free(self.curr_landmark_ids);
        self.allocator.free(self.prev_landmark_ids);
        self.allocator.free(self.gpu_matched_keypoints);
        self.allocator.free(self.gpu_temporal_matches);
        self.allocator.free(self.gpu_best_matches);
    }

    /// Clear landmark IDs for new frame
    pub fn clearCurrentLandmarkIds(self: *Self) void {
        for (self.curr_landmark_ids) |*id| {
            id.* = null;
        }
    }

    /// Swap current and previous buffers
    pub fn swapBuffers(self: *Self) void {
        std.mem.swap([]MatchedKeypointCPU, &self.curr_matched_keypoints, &self.prev_matched_keypoints);
        std.mem.swap([]?u64, &self.curr_landmark_ids, &self.prev_landmark_ids);
        self.prev_match_count = self.curr_match_count;
        self.curr_match_count = 0;
    }
};

// ============================================================================
// StereoFrame: Input frame pair for SLAM processing
// ============================================================================

pub const StereoFrame = struct {
    timestamp: i64,
    // GPU pointers to RGBA data (from viewport FBOs)
    left_rgba_ptr: [*]u8,
    right_rgba_ptr: [*]u8,
    width: u32,
    height: u32,
    rgba_pitch: u32,
    // Ground truth pose (for comparison)
    ground_truth_pose: ?SE3 = null,
};

// ============================================================================
// SLAMProcessor: The main SLAM processing pipeline
// ============================================================================

pub const SLAMProcessor = struct {
    allocator: std.mem.Allocator,
    config: SLAMConfig,
    map: SLAMMap,
    gpu: SLAMGPUResources,
    cpu: SLAMCPUResources,

    // Current state
    current_pose: SE3 = SE3.identity,
    tracking_state: TrackingState = .not_initialized,
    frame_count: u64 = 0,
    last_keyframe_id: ?u64 = null,

    // Processing thread
    thread: ?std.Thread = null,
    should_stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    is_processing: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    // Frame queue (single producer, single consumer)
    frame_mutex: std.Thread.Mutex = .{},
    pending_frame: ?StereoFrame = null,

    // Output (thread-safe read)
    output_mutex: std.Thread.Mutex = .{},
    last_estimated_pose: SE3 = SE3.identity,
    last_stats: SLAMStats = .{},

    const Self = @This();

    pub const SLAMStats = struct {
        tracked_features: u32 = 0,
        stereo_matches: u32 = 0,
        temporal_matches: u32 = 0,
        inliers: u32 = 0,
        processing_time_us: u64 = 0,
        keyframes: u32 = 0,
        landmarks: u32 = 0,
    };

    pub fn init(allocator: std.mem.Allocator, config: SLAMConfig) !*Self {
        const self = try allocator.create(Self);

        const gpu = try SLAMGPUResources.init(
            config.image_width,
            config.image_height,
            config.max_features_per_frame,
        );

        const cpu = try SLAMCPUResources.init(allocator, config.max_features_per_frame);

        self.* = .{
            .allocator = allocator,
            .config = config,
            .map = SLAMMap.init(allocator),
            .gpu = gpu,
            .cpu = cpu,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.stop();
        self.gpu.deinit();
        self.cpu.deinit();
        self.map.deinit();
        self.allocator.destroy(self);
    }

    /// Start the SLAM processing thread
    pub fn start(self: *Self) !void {
        if (self.thread != null) return;

        self.should_stop.store(false, .release);
        self.thread = try std.Thread.spawn(.{}, processingLoop, .{self});
    }

    /// Stop the SLAM processing thread
    pub fn stop(self: *Self) void {
        self.should_stop.store(true, .release);
        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }
    }

    /// Submit a stereo frame for processing
    pub fn submitFrame(self: *Self, frame: StereoFrame) void {
        self.frame_mutex.lock();
        defer self.frame_mutex.unlock();
        self.pending_frame = frame;
    }

    /// Get the latest estimated pose and stats
    pub fn getOutput(self: *Self) struct { pose: SE3, stats: SLAMStats } {
        self.output_mutex.lock();
        defer self.output_mutex.unlock();
        return .{ .pose = self.last_estimated_pose, .stats = self.last_stats };
    }

    /// Get just the stats
    pub fn getStats(self: *Self) SLAMStats {
        self.output_mutex.lock();
        defer self.output_mutex.unlock();
        return self.last_stats;
    }

    /// Get current tracking state
    pub fn getTrackingState(self: *Self) TrackingState {
        return self.tracking_state;
    }

    fn processingLoop(self: *Self) void {
        // Ensure this thread uses the same CUDA device/context
        if (cuda_config.CUDA_ENABLED) {
            const err = cuda.cudaSetDevice(0);
            if (err != cuda.cudaSuccess) {
                std.debug.print("SLAM thread: cudaSetDevice failed: {s}\n", .{cuda.cudaGetErrorString(err)});
            } else {
                std.debug.print("SLAM thread: cudaSetDevice(0) succeeded\n", .{});
            }
        }

        while (!self.should_stop.load(.acquire)) {
            // Try to get a frame
            const frame = blk: {
                self.frame_mutex.lock();
                defer self.frame_mutex.unlock();
                const f = self.pending_frame;
                self.pending_frame = null;
                break :blk f;
            };

            if (frame) |f| {
                self.is_processing.store(true, .release);
                self.processFrame(f);
                self.is_processing.store(false, .release);
            } else {
                // No frame available, sleep briefly
                std.time.sleep(1_000_000); // 1ms
            }
        }
    }

    fn processFrame(self: *Self, frame: StereoFrame) void {
        if (!cuda_config.CUDA_ENABLED) return;

        const start_time = std.time.microTimestamp();

        // 1. Convert RGBA to grayscale
        self.convertToGrayscale(frame);

        // 2. Gaussian blur
        self.applyGaussianBlur();

        // 3. Detect features (FAST + BRIEF)
        const left_count = self.detectFeatures(true);
        const right_count = self.detectFeatures(false);

        // 4. Stereo matching
        const stereo_matches = self.matchStereo(left_count, right_count);

        // 5. Copy matched keypoints from GPU to CPU for landmark tracking
        self.copyMatchedKeypointsToCPU(stereo_matches) catch {};

        var temporal_matches: u32 = 0;
        var inliers: u32 = 0;

        // 6. Temporal matching and motion estimation (if we have previous frame)
        if (self.frame_count > 0 and self.gpu.prev_match_count > 0) {
            temporal_matches = self.matchTemporal(stereo_matches);

            // Propagate landmark IDs from previous to current via temporal matches
            self.propagateLandmarkIds(temporal_matches) catch {};

            if (temporal_matches >= self.config.min_inliers) {
                // 7. Estimate motion with RANSAC
                const pose_result = self.estimateMotion(temporal_matches);
                inliers = pose_result.inliers;

                if (inliers >= self.config.min_inliers) {
                    // Update pose
                    self.current_pose = self.current_pose.multiply(pose_result.relative_pose);
                    self.tracking_state = .tracking;
                } else {
                    self.tracking_state = .lost;
                }
            } else {
                self.tracking_state = .lost;
            }
        } else if (self.frame_count == 0) {
            // First frame - initialize
            if (frame.ground_truth_pose) |gt| {
                self.current_pose = gt;
            }
            self.tracking_state = .tracking;
        }

        // 8. Keyframe decision - create keyframe with landmarks
        if (self.shouldCreateKeyframe()) {
            self.createKeyframe(frame.timestamp, stereo_matches) catch {};
        }

        // 9. Swap buffers for next frame
        self.gpu.prev_match_count = stereo_matches;
        self.gpu.swapFrames();
        self.cpu.swapBuffers();
        self.frame_count += 1;

        const end_time = std.time.microTimestamp();

        // Update output
        {
            self.output_mutex.lock();
            defer self.output_mutex.unlock();
            self.last_estimated_pose = self.current_pose;
            self.last_stats = .{
                .tracked_features = left_count,
                .stereo_matches = stereo_matches,
                .temporal_matches = temporal_matches,
                .inliers = inliers,
                .processing_time_us = @intCast(end_time - start_time),
                .keyframes = @intCast(self.map.keyframes.count()),
                .landmarks = @intCast(self.map.landmarks.count()),
            };
        }

        // Compare with ground truth if available
        if (frame.ground_truth_pose) |gt| {
            const trans_err = self.current_pose.translationDistance(gt);
            const rot_err = self.current_pose.rotationDistance(gt);
            _ = trans_err;
            _ = rot_err;
            // Could log or store these errors
        }
    }

    fn convertToGrayscale(self: *Self, frame: StereoFrame) void {
        const block = cuda.dim3{ .x = 16, .y = 16, .z = 1 };
        const grid = cuda.dim3{
            .x = (frame.width + 15) / 16,
            .y = (frame.height + 15) / 16,
            .z = 1,
        };

        // Convert left frame
        CudaGL.track("rgba_to_gray_left", void, cuda.launch_rgba_to_gray, .{
            frame.left_rgba_ptr,
            self.gpu.d_left_gray.?,
            @as(c_int, @intCast(frame.width)),
            @as(c_int, @intCast(frame.height)),
            @as(c_int, @intCast(frame.rgba_pitch)),
            @as(c_int, @intCast(frame.width)),
            grid,
            block,
        }) catch {};

        // Convert right frame
        CudaGL.track("rgba_to_gray_right", void, cuda.launch_rgba_to_gray, .{
            frame.right_rgba_ptr,
            self.gpu.d_right_gray.?,
            @as(c_int, @intCast(frame.width)),
            @as(c_int, @intCast(frame.height)),
            @as(c_int, @intCast(frame.rgba_pitch)),
            @as(c_int, @intCast(frame.width)),
            grid,
            block,
        }) catch {};
    }

    fn applyGaussianBlur(self: *Self) void {
        const block = cuda.dim3{ .x = 16, .y = 16, .z = 1 };
        const grid = cuda.dim3{
            .x = (self.gpu.width + 15) / 16,
            .y = (self.gpu.height + 15) / 16,
            .z = 1,
        };

        // Initialize gaussian kernel (once)
        cuda.init_gaussian_kernel(1.0); // sigma = 1.0

        // Blur left frame
        CudaGL.track("gaussian_blur_left", void, cuda.launch_gaussian_blur, .{
            self.gpu.d_left_gray.?,
            self.gpu.d_left_blurred.?,
            @as(c_int, @intCast(self.gpu.width)),
            @as(c_int, @intCast(self.gpu.height)),
            @as(c_int, @intCast(self.gpu.width)),
            grid,
            block,
        }) catch {};

        // Blur right frame
        CudaGL.track("gaussian_blur_right", void, cuda.launch_gaussian_blur, .{
            self.gpu.d_right_gray.?,
            self.gpu.d_right_blurred.?,
            @as(c_int, @intCast(self.gpu.width)),
            @as(c_int, @intCast(self.gpu.height)),
            @as(c_int, @intCast(self.gpu.width)),
            grid,
            block,
        }) catch {};
    }

    fn detectFeatures(self: *Self, is_left: bool) u32 {
        const block = cuda.dim3{ .x = 16, .y = 16, .z = 1 };
        const grid = cuda.dim3{
            .x = (self.gpu.width + 15) / 16,
            .y = (self.gpu.height + 15) / 16,
            .z = 1,
        };

        const input = if (is_left) self.gpu.d_left_blurred.? else self.gpu.d_right_blurred.?;
        const positions = if (is_left) self.gpu.d_left_positions.? else self.gpu.d_right_positions.?;
        const colors = if (is_left) self.gpu.d_left_colors.? else self.gpu.d_right_colors.?;
        const descriptors = if (is_left) self.gpu.d_left_descriptors.? else self.gpu.d_right_descriptors.?;
        const count_ptr = if (is_left) self.gpu.d_left_count.? else self.gpu.d_right_count.?;

        // Reset count
        _ = cuda.cudaMemset(count_ptr, 0, @sizeOf(c_uint));

        if (is_left) {
            CudaGL.track("keypoint_detect_left", void, cuda.launch_keypoint_detection, .{
                input,
                @as(c_int, @intCast(self.gpu.width)),
                @as(c_int, @intCast(self.gpu.height)),
                @as(c_int, @intCast(self.gpu.width)),
                self.config.fast_threshold,
                @as(u32, 9), // arc_length for FAST
                positions,
                colors,
                descriptors,
                count_ptr,
                @as(c_int, @intCast(self.gpu.max_features)),
                grid,
                block,
            }) catch {};
        } else {
            CudaGL.track("keypoint_detect_right", void, cuda.launch_keypoint_detection, .{
                input,
                @as(c_int, @intCast(self.gpu.width)),
                @as(c_int, @intCast(self.gpu.height)),
                @as(c_int, @intCast(self.gpu.width)),
                self.config.fast_threshold,
                @as(u32, 9), // arc_length for FAST
                positions,
                colors,
                descriptors,
                count_ptr,
                @as(c_int, @intCast(self.gpu.max_features)),
                grid,
                block,
            }) catch {};
        }

        // Read back count
        var count: c_uint = 0;
        _ = cuda.cudaMemcpy(&count, count_ptr, @sizeOf(c_uint), cuda.cudaMemcpyDeviceToHost);
        return count;
    }

    fn matchStereo(self: *Self, left_count: u32, right_count: u32) u32 {
        if (left_count == 0 or right_count == 0) return 0;

        const stereo_params = cuda.StereoParams{
            .image_width = @intCast(self.gpu.width),
            .image_height = @intCast(self.gpu.height),
            .baseline_mm = self.config.baseline * 1000.0,
            .focal_length_mm = 3.04, // from SensorCamera
            .focal_length_px = self.config.fx,
            .sensor_width_mm = 6.287,
            .intensity_threshold = self.config.fast_threshold,
            .circle_radius = 3,
            .arc_length = 9,
            .max_keypoints = self.gpu.max_features,
            .sigma = 1.0,
            .max_disparity = self.config.max_disparity,
            .epipolar_threshold = self.config.epipolar_threshold,
            .max_hamming_dist = @floatFromInt(self.config.stereo_match_threshold),
            .lowes_ratio = 0.8,
            .cost_threshold = 100.0,
            .epipolar_weight = 1.0,
            .disparity_weight = 1.0,
            .hamming_dist_weight = 1.0,
            .show_connections = false,
            .disable_matching = false,
            .disable_depth = false,
            .disable_spatial_tracking = false,
        };

        // Stereo matching
        CudaGL.track("stereo_matching", void, cuda.launch_stereo_matching, .{
            self.gpu.d_left_positions.?,
            self.gpu.d_left_descriptors.?,
            @as(c_uint, left_count),
            self.gpu.d_right_positions.?,
            self.gpu.d_right_descriptors.?,
            @as(c_uint, right_count),
            stereo_params,
            self.gpu.d_stereo_matches_lr.?,
            self.gpu.d_stereo_matches_rl.?,
        }) catch {};

        // Cross-check and triangulate
        const block = cuda.dim3{ .x = 256, .y = 1, .z = 1 };
        const grid = cuda.dim3{ .x = (left_count + 255) / 256, .y = 1, .z = 1 };

        _ = cuda.cudaMemset(self.gpu.d_match_count.?, 0, @sizeOf(c_uint));

        CudaGL.track("cross_check_matches", void, cuda.launch_cross_check_matches, .{
            self.gpu.d_stereo_matches_lr.?,
            self.gpu.d_stereo_matches_rl.?,
            self.gpu.d_left_positions.?,
            self.gpu.d_right_positions.?,
            @as(c_uint, left_count),
            @as(c_uint, right_count),
            stereo_params,
            self.gpu.d_matched_keypoints.?,
            self.gpu.d_match_count.?,
            grid,
            block,
        }) catch {};

        // Read back match count
        var count: c_uint = 0;
        _ = cuda.cudaMemcpy(&count, self.gpu.d_match_count.?, @sizeOf(c_uint), cuda.cudaMemcpyDeviceToHost);
        return count;
    }

    fn matchTemporal(self: *Self, curr_count: u32) u32 {
        if (curr_count == 0 or self.gpu.prev_match_count == 0) return 0;

        const temporal_params = cuda.TemporalParams{
            .max_distance = 5.0,
            .max_pixel_distance = @floatFromInt(self.config.temporal_search_radius),
            .min_confidence = 0.5,
            .min_matches = self.config.min_inliers,
            .ransac_threshold = self.config.ransac_threshold,
            .ransac_iterations = self.config.ransac_iterations,
            .ransac_points = 4,
            .spatial_weight = 1.0,
            .hamming_weight = 1.0,
            .img_weight = 1.0,
            .max_hamming_dist = @floatFromInt(self.config.temporal_match_threshold),
            .cost_threshold = 100.0,
            .lowes_ratio = 0.8,
        };

        const block = cuda.dim3{ .x = 256, .y = 1, .z = 1 };
        const grid_curr = cuda.dim3{ .x = (curr_count + 255) / 256, .y = 1, .z = 1 };
        const grid_prev = cuda.dim3{ .x = (self.gpu.prev_match_count + 255) / 256, .y = 1, .z = 1 };

        // Match current to previous
        CudaGL.track("temporal_curr_to_prev", void, cuda.launch_temporal_match_current_to_prev, .{
            self.gpu.d_matched_keypoints.?,
            @as(c_uint, curr_count),
            self.gpu.d_prev_matched_keypoints.?,
            @as(c_uint, self.gpu.prev_match_count),
            self.gpu.d_temporal_curr_to_prev.?,
            temporal_params,
            grid_curr,
            block,
        }) catch {};

        // Match previous to current
        CudaGL.track("temporal_prev_to_curr", void, cuda.launch_temporal_match_prev_to_current, .{
            self.gpu.d_prev_matched_keypoints.?,
            @as(c_uint, self.gpu.prev_match_count),
            self.gpu.d_matched_keypoints.?,
            @as(c_uint, curr_count),
            self.gpu.d_temporal_prev_to_curr.?,
            temporal_params,
            grid_prev,
            block,
        }) catch {};

        // Cross-check temporal matches
        _ = cuda.cudaMemset(self.gpu.d_temporal_match_count.?, 0, @sizeOf(c_uint));

        CudaGL.track("temporal_cross_check", void, cuda.launch_temporal_cross_check, .{
            self.gpu.d_temporal_curr_to_prev.?,
            self.gpu.d_temporal_prev_to_curr.?,
            self.gpu.d_matched_keypoints.?,
            self.gpu.d_prev_matched_keypoints.?,
            @as(c_uint, curr_count),
            @as(c_uint, self.gpu.prev_match_count),
            self.gpu.d_temporal_matches.?,
            self.gpu.d_temporal_match_count.?,
            temporal_params,
            grid_curr,
            block,
        }) catch {};

        // Read back count
        var count: c_uint = 0;
        _ = cuda.cudaMemcpy(&count, self.gpu.d_temporal_match_count.?, @sizeOf(c_uint), cuda.cudaMemcpyDeviceToHost);
        return count;
    }

    fn estimateMotion(self: *Self, match_count: u32) struct { relative_pose: SE3, inliers: u32 } {
        if (match_count < self.config.min_inliers) {
            return .{ .relative_pose = SE3.identity, .inliers = 0 };
        }

        var temporal_params = cuda.TemporalParams{
            .max_distance = 5.0,
            .max_pixel_distance = @floatFromInt(self.config.temporal_search_radius),
            .min_confidence = 0.5,
            .min_matches = self.config.min_inliers,
            .ransac_threshold = self.config.ransac_threshold,
            .ransac_iterations = self.config.ransac_iterations,
            .ransac_points = 4,
            .spatial_weight = 1.0,
            .hamming_weight = 1.0,
            .img_weight = 1.0,
            .max_hamming_dist = @floatFromInt(self.config.temporal_match_threshold),
            .cost_threshold = 100.0,
            .lowes_ratio = 0.8,
        };

        _ = cuda.cudaMemset(self.gpu.d_inlier_count.?, 0, @sizeOf(c_uint));

        const block = cuda.dim3{ .x = 256, .y = 1, .z = 1 };
        const grid = cuda.dim3{ .x = 1, .y = 1, .z = 1 };

        CudaGL.track("motion_estimation", void, cuda.launch_motion_estimation, .{
            self.gpu.d_temporal_matches.?,
            @as(c_uint, match_count),
            self.gpu.d_best_pose.?,
            self.gpu.d_inlier_count.?,
            &temporal_params,
            grid,
            block,
        }) catch {};

        // Read back results
        var pose: cuda.CameraPose = undefined;
        var inliers: c_uint = 0;
        _ = cuda.cudaMemcpy(&pose, self.gpu.d_best_pose.?, @sizeOf(cuda.CameraPose), cuda.cudaMemcpyDeviceToHost);
        _ = cuda.cudaMemcpy(&inliers, self.gpu.d_inlier_count.?, @sizeOf(c_uint), cuda.cudaMemcpyDeviceToHost);

        // Convert CameraPose to SE3
        const rotation = Quaternion.from_mat3(Math.Mat3{
            .base = .{ .data = .{
                pose.rotation[0], pose.rotation[1], pose.rotation[2],
                pose.rotation[3], pose.rotation[4], pose.rotation[5],
                pose.rotation[6], pose.rotation[7], pose.rotation[8],
            } },
        });
        const translation = Vec3.init(pose.translation[0], pose.translation[1], pose.translation[2]);

        return .{
            .relative_pose = SE3.init(rotation, translation),
            .inliers = inliers,
        };
    }

    fn shouldCreateKeyframe(self: *Self) bool {
        if (self.last_keyframe_id == null) return true;

        const kf = self.map.getKeyframe(self.last_keyframe_id.?) orelse return true;

        // Check translation threshold
        const trans_dist = self.current_pose.translationDistance(kf.pose);
        if (trans_dist > self.config.keyframe_translation_threshold) return true;

        // Check rotation threshold
        const rot_dist = Math.degrees(self.current_pose.rotationDistance(kf.pose));
        if (rot_dist > self.config.keyframe_rotation_threshold) return true;

        return false;
    }

    /// Copy matched keypoints from GPU to CPU for landmark tracking
    fn copyMatchedKeypointsToCPU(self: *Self, match_count: u32) !void {
        if (match_count == 0) {
            self.cpu.curr_match_count = 0;
            self.cpu.clearCurrentLandmarkIds();
            return;
        }

        const count = @min(match_count, self.cpu.max_features);

        // Copy from GPU to pre-allocated buffer
        _ = cuda.cudaMemcpy(
            self.cpu.gpu_matched_keypoints.ptr,
            self.gpu.d_matched_keypoints.?,
            count * @sizeOf(cuda.MatchedKeypoint),
            cuda.cudaMemcpyDeviceToHost,
        );

        // Convert to CPU format
        for (0..count) |i| {
            const gm = self.cpu.gpu_matched_keypoints[i];

            // Extract first 256 bits (32 bytes) of descriptor
            var desc: [32]u8 = undefined;
            const desc_bytes: [*]const u8 = @ptrCast(&gm.left_desc.descriptor);
            @memcpy(&desc, desc_bytes[0..32]);

            // Triangulate 3D point in camera frame using pixel coords and disparity
            const world_pos = self.config.triangulate(
                gm.world.image_coords.x,
                gm.world.image_coords.y,
                gm.world.disparity,
            ) orelse Vec3.init(0, 0, 0);

            self.cpu.curr_matched_keypoints[i] = .{
                .left_pos = .{ gm.left_pos.x, gm.left_pos.y },
                .right_pos = .{ gm.right_pos.x, gm.right_pos.y },
                .world_pos = world_pos,
                .descriptor = desc,
                .disparity = gm.world.disparity,
            };
            self.cpu.curr_landmark_ids[i] = null;
        }

        self.cpu.curr_match_count = count;
    }

    /// Propagate landmark IDs from previous frame to current via temporal matches
    fn propagateLandmarkIds(self: *Self, temporal_match_count: u32) !void {
        if (temporal_match_count == 0 or self.cpu.curr_match_count == 0) return;

        const count = @min(self.cpu.curr_match_count, self.cpu.max_features);

        // Copy temporal match indices from GPU
        _ = cuda.cudaMemcpy(
            self.cpu.gpu_best_matches.ptr,
            self.gpu.d_temporal_curr_to_prev.?,
            count * @sizeOf(cuda.BestMatch),
            cuda.cudaMemcpyDeviceToHost,
        );

        // For each current keypoint, propagate landmark ID from matched previous keypoint
        for (0..count) |curr_idx| {
            const match = self.cpu.gpu_best_matches[curr_idx];
            if (match.bestIdx >= 0 and match.bestCost < 100.0) {
                const prev_idx: usize = @intCast(match.bestIdx);
                if (prev_idx < self.cpu.prev_match_count) {
                    if (self.cpu.prev_landmark_ids[prev_idx]) |landmark_id| {
                        // Propagate landmark ID
                        self.cpu.curr_landmark_ids[curr_idx] = landmark_id;

                        // Update landmark position (running average)
                        if (self.map.getLandmark(landmark_id)) |lm| {
                            const cam_pos = self.cpu.curr_matched_keypoints[curr_idx].world_pos;
                            // Convert from camera frame to OpenGL frame
                            const cam_to_gl = Vec3.init(cam_pos.x(), -cam_pos.y(), -cam_pos.z());
                            const curr_world_pos = self.current_pose.transformPoint(cam_to_gl);
                            const n = @as(f32, @floatFromInt(lm.observation_count + 1));
                            lm.position = lm.position.scale((n - 1.0) / n).add(curr_world_pos.scale(1.0 / n));
                        }
                    }
                }
            }
        }
    }

    fn createKeyframe(self: *Self, timestamp: i64, match_count: u32) !void {
        const kf = try self.map.createKeyframe(timestamp, self.current_pose);
        self.last_keyframe_id = kf.id;

        // Add sequential edge to pose graph
        if (self.map.keyframes.count() > 1) {
            const prev_id = kf.id - 1;
            if (self.map.getKeyframe(prev_id)) |prev_kf| {
                const relative = prev_kf.pose.relativeTo(kf.pose);
                try self.map.addEdge(prev_id, kf.id, relative, false);
            }
        }

        // Create/update landmarks from stereo matches
        var covisibility_counts = std.AutoHashMap(u64, u32).init(self.allocator);
        defer covisibility_counts.deinit();

        const count = @min(match_count, self.cpu.curr_match_count);
        for (0..count) |idx| {
            const mkp = self.cpu.curr_matched_keypoints[idx];

            // Convert from camera frame (Z forward, Y down) to OpenGL frame (Z backward, Y up)
            // This is a 180° rotation around X axis: Y and Z are negated
            const cam_to_gl = Vec3.init(mkp.world_pos.x(), -mkp.world_pos.y(), -mkp.world_pos.z());

            // Transform to world coordinates
            const world_pos = self.current_pose.transformPoint(cam_to_gl);

            var landmark_id: u64 = undefined;

            if (self.cpu.curr_landmark_ids[idx]) |existing_id| {
                // Existing landmark - add observation
                landmark_id = existing_id;
                if (self.map.getLandmark(existing_id)) |lm| {
                    try lm.addObservation(kf.id, @intCast(idx));
                    lm.updateNormal(self.current_pose.translation);
                }
            } else {
                // Create new landmark
                const lm = try self.map.createLandmark(world_pos, mkp.descriptor, kf.id);
                landmark_id = lm.id;
                try lm.addObservation(kf.id, @intCast(idx));
                self.cpu.curr_landmark_ids[idx] = landmark_id;
            }

            // Add observation to keyframe
            try kf.observations.append(.{
                .keyframe_id = kf.id,
                .feature_idx = @intCast(idx),
                .landmark_id = landmark_id,
            });

            // Track covisibility with other keyframes
            if (self.map.getLandmark(landmark_id)) |lm| {
                for (lm.observations.items) |obs| {
                    if (obs.keyframe_id != kf.id) {
                        const entry = try covisibility_counts.getOrPut(obs.keyframe_id);
                        if (!entry.found_existing) {
                            entry.value_ptr.* = 0;
                        }
                        entry.value_ptr.* += 1;
                    }
                }
            }
        }

        // Update covisibility graph (bidirectional)
        var cov_it = covisibility_counts.iterator();
        while (cov_it.next()) |entry| {
            try kf.covisible_keyframes.put(entry.key_ptr.*, entry.value_ptr.*);
            if (self.map.getKeyframe(entry.key_ptr.*)) |other_kf| {
                try other_kf.covisible_keyframes.put(kf.id, entry.value_ptr.*);
            }
        }
    }
};
