//VisionV2.zig
const std = @import("std");
const c = @import("bindings/c.zig");
const gl = @import("bindings/gl.zig");
const CudaGL = @import("CudaGL.zig");
const Math = @import("Math.zig");
const libav = @import("bindings/libav.zig");
const Vision = @import("Vision.zig");
const Node = @import("Node.zig");
const Shape = @import("Shape.zig");
const Sensors = @import("Sensors.zig");

const cuda = c.cuda;
const glad = gl.glad;
const Vec3 = Math.Vec3;
const Mat3 = Math.Mat3;
const Mat4 = Math.Mat4;
const video = libav.video;
const InstancedKeypoints = Shape.InstancedKeypointDebugger;
const ImageParams = cuda.ImageParams;
const StereoParams = cuda.StereoParams;
const TemporalParams = cuda.TemporalParams;
const BufferResource = CudaGL.BufferResource;
const DetectionResources = CudaGL.DetectionResources;
const BRIEFDescriptor = cuda.BRIEFDescriptor;
const CameraPose = Vision.CameraPose;

const track_kernel = CudaGL.track;

const MAX_OBSERVATIONS_PER_LANDMARK = 32; // Maximum observations per landmark
const MAX_LANDMARKS_PER_KEYFRAME = 1024; // Maximum landmarks per keyframe

fn DefaultTemporalParams() TemporalParams {
    return TemporalParams{
        .max_distance = 1.0, // Maximum distance for temporal matching
        .max_pixel_distance = 50.0, // Maximum pixel distance
        .min_confidence = 0.7, // Minimum confidence threshold
        .min_matches = 30, // Minimum required matches
        .ransac_threshold = 0.01, // RANSAC inlier threshold
        .ransac_iterations = 256, // Number of RANSAC iterations
        .ransac_points = 8, // Number of randomly sampled temporal matches to estimate essential matrix with
        .spatial_weight = 0.4, // Weight for spatial distance term
        .hamming_weight = 0.4, // Weight for descriptor distance
        .img_weight = 0.2, // Weight for image space distance
        .max_hamming_dist = 1.0,
        .cost_threshold = 0.7,
        .lowes_ratio = 0.7,
    };
}

fn DefaultStereoParams(width: u32, height: u32) StereoParams {
    const sensor_width_mm = 6.45;
    const focal_length_mm = 2.75;

    return StereoParams{
        .image_width = @intCast(width),
        .image_height = @intCast(height),
        .baseline_mm = 76.3,
        .focal_length_mm = focal_length_mm,
        .focal_length_px = focal_length_mm * (@as(f32, @floatFromInt(width)) / sensor_width_mm),
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
}

/// KeyframeResources manages GPU resources for a keyframe's features
pub const KeyframeResources = struct {
    const Self = @This();

    // Basic info
    id: u64,
    allocator: std.mem.Allocator,
    max_features: u32,
    feature_count: u32 = 0,

    // OpenGL interop resources
    d_matches: *cuda.MatchedKeypoint,

    pub fn init(allocator: std.mem.Allocator, id: u64, max_features: u32) !*Self {
        const resources = try allocator.create(Self);

        var d_matches: *cuda.float4 = undefined;
        err = cuda.cudaMalloc(@ptrCast(&d_matches), max_features * @sizeOf(cuda.MatchedKeypoint));
        errdefer allocator.destroy(resources);
        if (err != cuda.cudaSuccess) {
            return error.CudaAllocationFailed;
        }

        resources.* = .{
            .id = id,
            .allocator = allocator,
            .max_features = max_features,
            .d_descriptors = d_descriptors,
            .d_matches = d_matches,
        };

        return resources;
    }

    pub fn deinit(self: *Self) void {
        _ = cuda.cudaFree(self.d_matches);
        self.allocator.destroy(self);
    }
};

/// FeatureStorage is a fixed-size array mapping landmark IDs to feature indices
pub const FeatureStorage = struct {
    landmark_ids: [MAX_LANDMARKS_PER_KEYFRAME]u64,
    feature_indices: [MAX_LANDMARKS_PER_KEYFRAME]u32,
    count: u32 = 0,

    pub fn init() FeatureStorage {
        var storage: FeatureStorage = undefined;
        storage.count = 0;
        return storage;
    }

    pub fn add(self: *FeatureStorage, landmark_id: u64, feature_idx: u32) bool {
        if (self.count >= MAX_LANDMARKS_PER_KEYFRAME) {
            return false;
        }

        self.landmark_ids[self.count] = landmark_id;
        self.feature_indices[self.count] = feature_idx;
        self.count += 1;
        return true;
    }

    pub fn getFeatureIndex(self: *const FeatureStorage, landmark_id: u64) ?u32 {
        for (0..self.count) |i| {
            if (self.landmark_ids[i] == landmark_id) {
                return self.feature_indices[i];
            }
        }
        return null;
    }

    pub fn getLandmarkIdsArray(self: *const FeatureStorage) []const u64 {
        return self.landmark_ids[0..self.count];
    }

    pub fn getFeatureIndicesArray(self: *const FeatureStorage) []const u32 {
        return self.feature_indices[0..self.count];
    }
};

/// Keyframe represents a camera view with detected features and their landmark observations
pub const Keyframe = struct {
    const Self = @This();

    // Basic information
    id: u64,
    timestamp: i64,
    pose: CameraPose,

    // GPU resources
    resources: *KeyframeResources,

    // Observations (optimized fixed-size storage)
    observations: FeatureStorage,

    // Original image (can be null to save memory)
    image: ?*video.AVFrame = null,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, id: u64, timestamp: i64, pose: CameraPose, resources: *KeyframeResources, image: ?*video.AVFrame) !*Self {
        const keyframe = try allocator.create(Self);

        var clone_image: ?*video.AVFrame = null;
        if (image) |img| {
            clone_image = try video.av_frame_clone(img);
        }

        keyframe.* = .{
            .id = id,
            .timestamp = timestamp,
            .pose = pose,
            .resources = resources,
            .observations = FeatureStorage.init(),
            .image = clone_image,
            .allocator = allocator,
        };

        return keyframe;
    }

    pub fn deinit(self: *Self) void {
        if (self.image) |img| {
            video.av_frame_free(@constCast(@ptrCast(&img)));
        }

        self.allocator.destroy(self);
    }

    // Add an observation of a landmark in this keyframe
    pub fn addObservation(self: *Self, landmark_id: u64, feature_idx: u32) bool {
        return self.observations.add(landmark_id, feature_idx);
    }

    // Get the feature index for a given landmark
    pub fn getFeatureIndex(self: *const Self, landmark_id: u64) ?u32 {
        return self.observations.getFeatureIndex(landmark_id);
    }

    // Copy feature data from source buffer
    pub fn copyFeatureData(self: *Self, source_positions: *cuda.MatchedKeypoint, feature_count: u32) !void {
        if (feature_count > self.resources.max_features) {
            return error.TooManyFeatures;
        }

        // Copy positions to OpenGL buffer
        _ = cuda.cudaMemcpy(
            self.resources.d_matches,
            source_positions,
            feature_count * @sizeOf(cuda.MatchedKeypoint),
            cuda.cudaMemcpyDeviceToDevice,
        );

        self.resources.feature_count = feature_count;
    }

    // Get the world position of a feature
    pub fn getFeatureWorldPosition(self: *Self, feature_idx: u32, image_width: f32, image_height: f32) !Vec3 {
        if (feature_idx >= self.resources.feature_count) {
            return error.InvalidFeatureIndex;
        }

        // Read feature position from GPU
        var feature: cuda.MatchedKeypoint = undefined;
        _ = cuda.cudaMemcpy(
            &feature_pos,
            &self.resources.d_matches[feature_idx],
            @sizeOf(cuda.float4),
            cuda.cudaMemcpyDeviceToHost,
        );

        const feature_pos = feature.left_pos;

        // Convert canvas coordinates to pixel coordinates
        const px_x = (feature_pos.x / 12.8 + 0.5) * image_width;
        const px_y = (-feature_pos.z / 7.2 + 0.5) * image_height;

        // Create world position vector
        // Note: For a real implementation, you'd need to unproject and transform by pose
        return Vec3.init(px_x, feature_pos.y, px_y);
    }
};

/// ObservationStorage is a fixed-size array for keyframe observations
pub const ObservationStorage = struct {
    keyframe_ids: [MAX_OBSERVATIONS_PER_LANDMARK]u64,
    feature_indices: [MAX_OBSERVATIONS_PER_LANDMARK]u32,
    count: u32 = 0,

    pub fn init() ObservationStorage {
        var storage: ObservationStorage = undefined;
        storage.count = 0;
        return storage;
    }

    pub fn add(self: *ObservationStorage, keyframe_id: u64, feature_idx: u32) bool {
        if (self.count >= MAX_OBSERVATIONS_PER_LANDMARK) {
            return false;
        }

        self.keyframe_ids[self.count] = keyframe_id;
        self.feature_indices[self.count] = feature_idx;
        self.count += 1;
        return true;
    }

    pub fn getFeatureIndex(self: *const ObservationStorage, keyframe_id: u64) ?u32 {
        for (0..self.count) |i| {
            if (self.keyframe_ids[i] == keyframe_id) {
                return self.feature_indices[i];
            }
        }
        return null;
    }

    pub fn getKeyframeIdsArray(self: *const ObservationStorage) []const u64 {
        return self.keyframe_ids[0..self.count];
    }

    pub fn getFeatureIndicesArray(self: *const ObservationStorage) []const u32 {
        return self.feature_indices[0..self.count];
    }
};

/// Landmark represents a 3D point observed across multiple keyframes
pub const Landmark = struct {
    const Self = @This();

    // Basic information
    id: u64,
    position: Vec3,
    color: [3]u8 = .{ 255, 0, 0 },

    // Source information
    source_keyframe_id: u64,
    source_feature_idx: u32,

    // Observations (optimized fixed storage)
    observations: ObservationStorage,

    // Optimization and tracking metadata
    is_valid: bool = true,
    consecutive_failures: u32 = 0,
    last_optimization_error: f32 = 0.0,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, id: u64, position: Vec3, source_keyframe_id: u64, source_feature_idx: u32) !*Self {
        var landmark = try allocator.create(Self);

        landmark.* = .{
            .id = id,
            .position = position,
            .source_keyframe_id = source_keyframe_id,
            .source_feature_idx = source_feature_idx,
            .observations = ObservationStorage.init(),
            .allocator = allocator,
        };

        // Add the source keyframe as the first observation
        _ = landmark.observations.add(source_keyframe_id, source_feature_idx);

        return landmark;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    // Add an observation from a keyframe
    pub fn addObservation(self: *Self, keyframe_id: u64, feature_idx: u32) bool {
        // Don't add duplicate observations
        if (self.observations.getFeatureIndex(keyframe_id) != null) {
            return false;
        }

        const success = self.observations.add(keyframe_id, feature_idx);
        if (success) {
            self.consecutive_failures = 0;
        }
        return success;
    }

    // Record a tracking failure
    pub fn recordFailure(self: *Self) void {
        self.consecutive_failures += 1;
    }

    // Update the 3D position (typically from bundle adjustment)
    pub fn updatePosition(self: *Self, new_position: Vec3) void {
        self.position = new_position;
    }

    // Check if this landmark is reliable
    pub fn isReliable(self: *const Self, min_observations: u32) bool {
        return self.is_valid and
            self.observations.count >= min_observations and
            self.consecutive_failures < 3;
    }

    // Get observation count
    pub fn getObservationCount(self: *const Self) u32 {
        return self.observations.count;
    }

    // Get track length in terms of keyframe span
    pub fn getTrackLength(self: *const Self) u32 {
        var min_id: u64 = std.math.maxInt(u64);
        var max_id: u64 = 0;

        for (0..self.observations.count) |i| {
            const keyframe_id = self.observations.keyframe_ids[i];
            min_id = @min(min_id, keyframe_id);
            max_id = @max(max_id, keyframe_id);
        }

        return if (min_id == std.math.maxInt(u64)) 0 else @intCast(max_id - min_id + 1);
    }
};

pub const SparseMapperConfig = struct {
    // Feature detection parameters
    max_features_per_frame: u32 = 10000,
    detection_threshold: u8 = 20,
    min_distance: i32 = 10,

    // Keyframe selection parameters
    min_translation: f32 = 0.1, // Minimum translation for new keyframe (meters)
    min_rotation: f32 = 15.0, // Minimum rotation for new keyframe (degrees)
    min_tracked_ratio: f32 = 0.3, // Minimum ratio of tracked features for new keyframe

    // Feature tracking parameters
    descriptor_distance_threshold: f32 = 50.0,
    max_descriptor_ratio: f32 = 0.8, // Lowe's ratio test

    // Landmark parameters
    min_observations: u32 = 3, // Minimum observations for valid landmark
    max_reprojection_error: f32 = 2.0, // Maximum reprojection error (pixels)

    // Stereo parameters
    baseline_mm: f32 = 76.3,
    focal_length_px: f32 = 1320.0, // Focal length in pixels

    // Camera intrinsics
    principal_x: f32 = 640.0, // Principal point x
    principal_y: f32 = 360.0, // Principal point y
};

/// TrackingData holds temporary data structures used during tracking
pub const TrackingData = struct {
    // Current feature tracking state
    current_keyframe_id: u64 = 0,
    current_feature_count: u32 = 0,

    // Feature to landmark mapping for the current frame
    // Mapping is by feature index -> landmark ID
    landmark_map: [MAX_LANDMARKS_PER_KEYFRAME]u64 = undefined,
    landmark_count: u32 = 0,

    // Buffers for matching and tracking
    d_matches_1: ?*cuda.BestMatch = null,
    d_matches_2: ?*cuda.BestMatch = null,
    d_match_count: ?*c_uint = null,

    // Temporary buffers for image processing
    d_right_blurred: ?*u8 = null,
    d_left_blurred: ?*u8 = null,

    // Stereo matching resources
    d_right_feature_count: ?*c_uint = null,
    d_right_positions: ?*cuda.float4 = null,
    d_right_colors: ?*cuda.float4 = null,
    d_right_descriptors: ?*cuda.BRIEFDescriptor = null,

    d_left_feature_count: ?*c_uint = null,
    d_left_positions: ?*cuda.float4 = null,
    d_left_colors: ?*cuda.float4 = null,
    d_left_descriptors: ?*cuda.BRIEFDescriptor = null,

    // Single-step match data
    d_matched_features: ?*cuda.MatchedKeypoint = null,

    // Temporal matches data
    d_temporal_matches: ?*cuda.TemporalMatch = null,

    // Feature buffer access for direct GPU-CPU transfers
    single_feature_buffer: cuda.float4 = .{ .x = 0, .y = 0, .z = 0, .w = 0 },

    pub fn init(max_features: u32) !TrackingData {
        var data = TrackingData{};

        // Allocate CUDA resources
        var err = cuda.cudaMalloc(@ptrCast(&data.d_matches_1), max_features * @sizeOf(cuda.BestMatch));
        err |= cuda.cudaMalloc(@ptrCast(&data.d_matches_2), max_features * @sizeOf(cuda.BestMatch));
        err |= cuda.cudaMalloc(@ptrCast(&data.d_match_count), @sizeOf(c_uint));

        // Allocate temporary buffers (assuming max resolution 1920x1080)
        const max_buffer_size = 1920 * 1080;
        err |= cuda.cudaMalloc(@ptrCast(&data.d_right_blurred), max_buffer_size);
        err |= cuda.cudaMalloc(@ptrCast(&data.d_left_blurred), max_buffer_size);

        // Allocate stereo matching resources
        err |= cuda.cudaMalloc(@ptrCast(&data.d_right_features_count), @sizeOf(c_uint));
        err |= cuda.cudaMalloc(@ptrCast(&data.d_right_positions), max_features * @sizeOf(cuda.float4));
        err |= cuda.cudaMalloc(@ptrCast(&data.d_right_colors), max_features * @sizeOf(cuda.float4));
        err |= cuda.cudaMalloc(@ptrCast(&data.d_right_descriptors), max_features * @sizeOf(cuda.BRIEFDescriptor));

        err |= cuda.cudaMalloc(@ptrCast(&data.d_left_features_count), @sizeOf(c_uint));
        err |= cuda.cudaMalloc(@ptrCast(&data.d_left_positions), max_features * @sizeOf(cuda.float4));
        err |= cuda.cudaMalloc(@ptrCast(&data.d_left_colors), max_features * @sizeOf(cuda.float4));
        err |= cuda.cudaMalloc(@ptrCast(&data.d_left_descriptors), max_features * @sizeOf(cuda.BRIEFDescriptor));

        // Allocate matched pairs buffer
        err |= cuda.cudaMalloc(@ptrCast(&data.d_matched_features), max_features * @sizeOf(cuda.MatchedKeypoint));

        // Allocate temporal matches buffer
        err |= cuda.cudaMalloc(@ptrCast(&data.d_temporal_matches), max_features * @sizeOf(cuda.TemporalMatch));

        if (err != cuda.cudaSuccess) {
            data.deinit();
            return error.CudaAllocationFailed;
        }

        return data;
    }

    pub fn deinit(self: *TrackingData) void {
        if (self.d_matches_1) |ptr| _ = cuda.cudaFree(ptr);
        if (self.d_matches_2) |ptr| _ = cuda.cudaFree(ptr);
        if (self.d_match_count) |ptr| _ = cuda.cudaFree(ptr);

        if (self.d_right_blurred) |ptr| _ = cuda.cudaFree(ptr);
        if (self.d_left_blurred) |ptr| _ = cuda.cudaFree(ptr);

        if (self.d_right_features_count) |ptr| _ = cuda.cudaFree(ptr);
        if (self.d_right_positions) |ptr| _ = cuda.cudaFree(ptr);
        if (self.d_right_colors) |ptr| _ = cuda.cudaFree(ptr);
        if (self.d_right_descriptors) |ptr| _ = cuda.cudaFree(ptr);

        if (self.d_left_features_count) |ptr| _ = cuda.cudaFree(ptr);
        if (self.d_left_positions) |ptr| _ = cuda.cudaFree(ptr);
        if (self.d_left_colors) |ptr| _ = cuda.cudaFree(ptr);
        if (self.d_left_descriptors) |ptr| _ = cuda.cudaFree(ptr);

        if (self.d_matched_features) |ptr| _ = cuda.cudaFree(ptr);
        if (self.d_temporal_matches) |ptr| _ = cuda.cudaFree(ptr);

        self.* = undefined;
    }

    pub fn resetTracking(self: *TrackingData) void {
        self.landmark_count = 0;

        // Initialize landmark map to invalid values
        @memset(self.landmark_map[0..MAX_LANDMARKS_PER_KEYFRAME], std.math.maxInt(u64));
    }

    pub fn addLandmarkMatch(self: *TrackingData, feature_idx: u32, landmark_id: u64) bool {
        if (feature_idx >= MAX_LANDMARKS_PER_KEYFRAME or self.landmark_count >= MAX_LANDMARKS_PER_KEYFRAME) {
            return false;
        }

        self.landmark_map[feature_idx] = landmark_id;
        self.landmark_count += 1;
        return true;
    }

    pub fn getLandmarkForFeature(self: *const TrackingData, feature_idx: u32) ?u64 {
        if (feature_idx >= MAX_LANDMARKS_PER_KEYFRAME) {
            return null;
        }

        const landmark_id = self.landmark_map[feature_idx];
        if (landmark_id == std.math.maxInt(u64)) {
            return null;
        }

        return landmark_id;
    }
};

/// SparseMapper implements visual SLAM functionality
pub const SparseMapper = struct {
    const Self = @This();

    // Configuration
    allocator: std.mem.Allocator,
    config: SparseMapperConfig,
    stereo_params: StereoParams,
    temporal_params: TemporalParams,

    // GL-CUDA interop for detection and visualization
    visualization_node: ?*Node,

    // Map data
    keyframes: std.AutoHashMap(u64, *Keyframe),
    landmarks: std.AutoHashMap(u64, *Landmark),

    // State tracking
    next_keyframe_id: u64 = 1,
    next_landmark_id: u64 = 1,
    last_keyframe_id: u64 = 0,
    tracking_data: TrackingData,
    current_pose: CameraPose,

    pub fn init(
        allocator: std.mem.Allocator,
        config: SparseMapperConfig,
        stereo_params: StereoParams,
        temporal_params: TemporalParams,
        environment_node: *Node,
    ) !*Self {
        const self = try allocator.create(Self);

        // Create visualization node
        const visualization_node = try Node.init(allocator, null, null, null);
        try environment_node.addChild(visualization_node);

        // Initialize tracking data
        const tracking_data = try TrackingData.init(config.max_features_per_frame);

        self.* = .{
            .allocator = allocator,
            .config = config,
            .stereo_params = stereo_params,
            .temporal_params = temporal_params,
            // .detection_resources = detection_resources,
            .visualization_node = visualization_node,
            .keyframes = std.AutoHashMap(u64, *Keyframe).init(allocator),
            .landmarks = std.AutoHashMap(u64, *Landmark).init(allocator),
            .tracking_data = tracking_data,
            .current_pose = .{
                .rotation = .{
                    1, 0, 0,
                    0, 1, 0,
                    0, 0, 1,
                },
                .translation = .{ 0, 0, 0 },
            },
        };

        cuda.init_gaussian_kernel(self.stereo_params.sigma);

        return self;
    }

    pub fn deinit(self: *Self) void {
        // Free all keyframes
        var keyframe_it = self.keyframes.valueIterator();
        while (keyframe_it.next()) |keyframe| {
            keyframe.*.deinit();
        }
        self.keyframes.deinit();

        // Free all landmarks
        var landmark_it = self.landmarks.valueIterator();
        while (landmark_it.next()) |landmark| {
            landmark.*.deinit();
        }
        self.landmarks.deinit();

        // Free tracking data
        self.tracking_data.deinit();

        self.allocator.destroy(self);
    }

    /// Process a new stereo frame pair
    pub fn processFrame(self: *Self, left_frame: *video.AVFrame, right_frame: *video.AVFrame, initial_pose: CameraPose) !void {
        std.debug.print("\n=== Processing new frame ===\n", .{});

        // 1. Triangulate Featutes
        self.triangulateFeatures(left_frame, right_frame) catch |err| {
            std.debug.print("Failed to triangulate Features. Err => {any}\n", .{err});
        };

        // 2. Track features from previous keyframe if available
        self.tracking_data.resetTracking();

        if (self.keyframes.count() > 0) {
            try self.trackFeatures();
        }

        // // 3. Determine if this should be a new keyframe
        const is_keyframe = self.keyframes.count() == 0 or try self.shouldCreateKeyframe(initial_pose);

        // // Start with initial pose and refine
        const estimated_pose = initial_pose;
        // if (self.keyframes.count() > 0) {
        //     if (self.keyframes.get(self.last_keyframe_id)) |prev_keyframe| {
        //         // Use previous keyframe pose as starting point
        //         estimated_pose.translation[0] = prev_keyframe.pose.translation[0];
        //         estimated_pose.translation[1] = prev_keyframe.pose.translation[1];
        //         estimated_pose.translation[2] = prev_keyframe.pose.translation[2];

        //         // TODO: Refine pose with motion model or feature tracking
        //     }
        // }

        if (is_keyframe) {
            std.debug.print("Creating new keyframe {}\n", .{self.next_keyframe_id});

            // 4. Create new keyframe
            try self.createKeyframe(left_frame, right_frame, estimated_pose);

            // 5. Periodically prune the map
            // if (self.keyframes.count() % 15 == 0) {
            //     try self.pruneMap();
            // }

            self.last_keyframe_id = self.next_keyframe_id - 1;
        } else {
            // Just update current pose without creating keyframe
            self.current_pose = estimated_pose;
        }

        // // 6. Update visualization
        // try self.updateVisualization();
    }

    /// Triangulate a feature using stereo matching
    fn triangulateFeatures(self: *Self, left_frame: *video.AVFrame, right_frame: *video.AVFrame) !void {
        _ = cuda.cudaDeviceSynchronize();

        const width = self.stereo_params.image_width;
        const height = self.stereo_params.image_height;

        // Configure blocks and grid for CUDA kernels
        const block = cuda.dim3{
            .x = 16,
            .y = 16,
            .z = 1,
        };

        const grid = cuda.dim3{
            .x = (@as(c_uint, @intCast(width)) + block.x - 1) / block.x,
            .y = (@as(c_uint, @intCast(height)) + block.y - 1) / block.y,
            .z = 1,
        };

        //Apply Gaussian Blur to left raw image
        try track_kernel("Gaussian_Blur", void, cuda.launch_gaussian_blur, .{
            left_frame.data[0],
            self.tracking_data.d_left_blurred.?,
            left_frame.width,
            left_frame.height,
            left_frame.linesize[0],
            grid,
            block,
        });

        // Reset left keypoint count
        _ = cuda.cudaMemset(self.tracking_data.d_left_feature_count, 0, @sizeOf(c_uint));

        // Detect FAST features and compute BRIEF descriptors in left image
        try track_kernel("Left_Keypoint_Detection", void, cuda.launch_keypoint_detection, .{
            self.tracking_data.d_left_blurred.?, // Blurred image
            width,
            height,
            left_frame.linesize[0],
            self.stereo_params.intensity_threshold, // FAST threshold
            self.stereo_params.arc_length, // FAST consecutive points
            self.tracking_data.d_left_positions.?,
            self.tracking_data.d_left_colors.?,
            self.tracking_data.d_left_descriptors.?,
            self.tracking_data.d_left_feature_count.?,
            @as(c_int, @intCast(self.config.max_features_per_frame)),
            grid,
            block,
        });

        // Get the number of features detected in left image
        var left_feature_count: c_uint = 0;
        _ = cuda.cudaMemcpy(&left_feature_count, self.tracking_data.d_left_feature_count.?, @sizeOf(c_uint), cuda.cudaMemcpyDeviceToHost);

        // If no features detected in left image, return empty result
        if (left_feature_count == 0) {
            return error.NoValidFeaturesLeft;
        }

        //Apply Gaussian Blur to right raw image
        try track_kernel("Gaussian_Blur", void, cuda.launch_gaussian_blur, .{
            right_frame.data[0],
            self.tracking_data.d_right_blurred.?,
            right_frame.width,
            right_frame.height,
            right_frame.linesize[0],
            grid,
            block,
        });

        // Reset right keypoint count
        _ = cuda.cudaMemset(self.tracking_data.d_right_feature_count, 0, @sizeOf(c_uint));

        // Detect FAST features and compute BRIEF descriptors in right image
        try track_kernel("Right_Keypoint_Detection", void, cuda.launch_keypoint_detection, .{
            self.tracking_data.d_right_blurred.?, // Blurred image
            width,
            height,
            right_frame.linesize[0],
            self.config.detection_threshold, // FAST threshold
            self.stereo_params.arc_length, // FAST consecutive points
            self.tracking_data.d_right_positions.?,
            self.tracking_data.d_right_colors.?,
            self.tracking_data.d_right_descriptors.?,
            self.tracking_data.d_right_feature_count.?,
            @as(c_int, @intCast(self.config.max_features_per_frame)),
            grid,
            block,
        });

        // Get the number of features detected in right image
        var right_feature_count: c_uint = 0;
        _ = cuda.cudaMemcpy(&right_feature_count, self.tracking_data.d_right_feature_count.?, @sizeOf(c_uint), cuda.cudaMemcpyDeviceToHost);

        // If no features detected in right image, return empty result
        if (right_feature_count == 0) {
            return error.NoValidFeaturesRight;
        }

        std.debug.print("Left Count: {d}    |   Right Count: {d}\n", .{ left_feature_count, right_feature_count });

        // Reset match count
        _ = cuda.cudaMemset(self.tracking_data.d_match_count, 0, @sizeOf(c_uint));

        // Run stereo matching between left and right images
        try track_kernel("Stereo_Matching", void, cuda.launch_stereo_matching, .{
            self.tracking_data.d_left_positions.?,
            self.tracking_data.d_left_descriptors.?,
            left_feature_count,
            self.tracking_data.d_right_positions.?,
            self.tracking_data.d_right_descriptors.?,
            right_feature_count,
            self.stereo_params,
            self.tracking_data.d_matches_1.?,
            self.tracking_data.d_matches_2.?,
        });

        // Configure blocks for cross-checking
        const cross_check_block = cuda.dim3{ .x = 512, .y = 1, .z = 1 };
        const cross_check_grid = cuda.dim3{
            .x = (left_feature_count + block.x - 1) / block.x,
            .y = 1,
            .z = 1,
        };

        // Run cross-check to find consistent matches
        try track_kernel("Cross_Check_Matches", void, cuda.launch_cross_check_matches, .{
            self.tracking_data.d_matches_1.?,
            self.tracking_data.d_matches_2.?,
            self.tracking_data.d_left_positions.?,
            self.tracking_data.d_right_positions.?,
            left_feature_count,
            right_feature_count,
            self.stereo_params,
            self.tracking_data.d_matched_pairs.?,
            self.tracking_data.d_match_count.?,
            cross_check_grid,
            cross_check_block,
        });

        // Get the number of successful matches
        _ = cuda.cudaMemcpy(&self.tracking_data.current_feature_count, self.tracking_data.d_match_count.?, @sizeOf(c_uint), cuda.cudaMemcpyDeviceToHost);
        std.debug.print("Num Matches: {d}\n", .{self.tracking_data.current_feature_count});

        // If no matches, return empty result
        if (self.tracking_data.current_feature_count == 0) {
            std.debug.print("No stereo matches found\n", .{});
            return error.NoValidMatches;
        }
    }

    /// Track features between current frame and the most recent keyframe
    fn trackFeatures(self: *Self) !void {
        if (self.keyframes.count() == 0 or self.last_keyframe_id == 0) {
            return; // Nothing to track
        }

        // Get the last keyframe
        const last_keyframe = self.keyframes.get(self.last_keyframe_id) orelse {
            std.debug.print("No keyframe available for tracking\n", .{});
            return;
        };

        // Check if we have features to track
        if (last_keyframe.observations.count == 0 or self.tracking_data.current_feature_count == 0) {
            std.debug.print(
                "No features to track (KF: {}, Current: {})\n",
                .{ last_keyframe.observations.count, self.tracking_data.current_feature_count },
            );
            return;
        }

        // Configure blocks and grid for CUDA kernels
        const block = cuda.dim3{ .x = 128, .y = 1, .z = 1 };

        const curr_grid = cuda.dim3{
            .x = (@as(c_uint, @intCast(self.tracking_data.current_feature_count)) + block.x - 1) / block.x,
            .y = 1,
            .z = 1,
        };

        const prev_grid = cuda.dim3{
            .x = (@as(c_uint, @intCast(last_keyframe.resources.feature_count)) + block.x - 1) / block.x,
            .y = 1,
            .z = 1,
        };

        // Current frame positions and descriptors
        const curr_features = self.tracking_data.d_matched_features.?;

        // Previous frame positions and descriptors
        const prev_features = last_keyframe.resources.d_matches;

        // Match current features to previous keyframe
        try track_kernel("Temporal_Match_Current_To_Prev", void, cuda.launch_temporal_match_current_to_prev, .{
            curr_features,
            @as(c_uint, @intCast(self.tracking_data.current_feature_count)),
            prev_features,
            @as(c_uint, @intCast(last_keyframe.resources.feature_count)),
            self.tracking_data.d_matches_1.?,
            self.temporal_params,
            curr_grid,
            block,
        });

        // Match previous keyframe features to current frame
        try track_kernel("Temporal_Match_Prev_To_Current", void, cuda.launch_temporal_match_prev_to_current, .{
            prev_features,
            @as(c_uint, @intCast(last_keyframe.resources.feature_count)),
            curr_features,
            @as(c_uint, @intCast(self.tracking_data.current_feature_count)),
            self.tracking_data.d_matches_2.?,
            self.temporal_params,
            prev_grid,
            block,
        });

        // Reset match count
        _ = cuda.cudaMemset(self.tracking_data.d_match_count, 0, @sizeOf(c_uint));

        // Run temporal cross-check to find consistent matches
        try track_kernel("Temporal_Cross_Check", void, cuda.launch_temporal_cross_check, .{
            self.tracking_data.d_matches_1.?,
            self.tracking_data.d_matches_2.?,
            curr_features,
            prev_features,
            @as(c_uint, @intCast(self.tracking_data.current_feature_count)),
            @as(c_uint, @intCast(last_keyframe.resources.feature_count)),
            self.tracking_data.d_temporal_matches.?,
            self.tracking_data.d_match_count.?,
            self.temporal_params,
            curr_grid,
            block,
        });

        // Get number of successful matches
        var match_count: c_uint = 0;
        _ = cuda.cudaMemcpy(&match_count, self.tracking_data.d_match_count, @sizeOf(c_uint), cuda.cudaMemcpyDeviceToHost);

        // If no matches found, exit early
        if (match_count == 0) {
            std.debug.print("No matches found during temporal tracking\n", .{});
            return;
        }

        // Fetch the matched pairs
        const temporal_matches = try self.allocator.alloc(cuda.TemporalMatch, match_count);
        defer self.allocator.free(temporal_matches);

        _ = cuda.cudaMemcpy(
            temporal_matches.ptr,
            self.tracking_data.d_temporal_matches,
            match_count * @sizeOf(cuda.TemporalMatch),
            cuda.cudaMemcpyDeviceToHost,
        );

        std.debug.print("Temporal Match Count: {d}\n", .{match_count});

        // // Process matches: connect current features with landmarks
        // var matches_found: u32 = 0;

        // // Build a mapping from feature index to landmark ID for the keyframe
        // var keyframe_feature_to_landmark = try self.allocator.alloc(?u64, last_keyframe.resources.feature_count);
        // defer self.allocator.free(keyframe_feature_to_landmark);

        // @memset(keyframe_feature_to_landmark, null);

        // // Populate the mapping from feature indices to landmark IDs
        // for (last_keyframe.observations.getLandmarkIdsArray(), last_keyframe.observations.getFeatureIndicesArray()) |landmark_id, feature_idx| {
        //     if (feature_idx < keyframe_feature_to_landmark.len) {
        //         keyframe_feature_to_landmark[feature_idx] = landmark_id;
        //     }
        // }

        // // Process temporal matches
        // for (temporal_matches) |match_data| {
        //     // Get feature indices
        //     const curr_idx = match_data.curr_idx;
        //     const prev_idx = match_data.prev_idx;

        //     if (prev_idx < keyframe_feature_to_landmark.len) {
        //         if (keyframe_feature_to_landmark[prev_idx]) |landmark_id| {
        //             // Found a valid match between current feature and existing landmark
        //             if (self.tracking_data.addLandmarkMatch(curr_idx, landmark_id)) {
        //                 // Update landmark stats to avoid pruning
        //                 if (self.landmarks.getPtr(landmark_id)) |landmark| {
        //                     landmark.*.consecutive_failures = 0;
        //                 }

        //                 matches_found += 1;
        //             }
        //         }
        //     }
        // }

        // const tracked_percentage = if (last_keyframe.observations.count > 0)
        //     @as(f32, @floatFromInt(matches_found)) * 100.0 / @as(f32, @floatFromInt(last_keyframe.observations.count))
        // else
        //     0.0;

        // std.debug.print(
        //     "Successfully tracked {}/{} landmarks ({:.1}%)\n",
        //     .{
        //         matches_found,
        //         last_keyframe.observations.count,
        //         tracked_percentage,
        //     },
        // );
    }

    /// Create a new keyframe from the current frame
    fn createKeyframe(self: *Self, left_frame: *video.AVFrame, right_frame: *video.AVFrame, pose: CameraPose) !void {
        // Now create the keyframe using these triangulated positions
        const keyframe_id = self.next_keyframe_id;
        self.next_keyframe_id += 1;

        // Create keyframe resources
        const keyframe_resources = try KeyframeResources.init(
            self.allocator,
            keyframe_id,
            self.config.max_features_per_frame,
        );

        // Create keyframe object
        var keyframe = try Keyframe.init(
            self.allocator,
            keyframe_id,
            left_frame.pts,
            pose,
            keyframe_resources,
            null, // Don't store original frame to save memory
        );

        // Copy feature data from detection resources to keyframe resources
        try keyframe.copyFeatureData(
            self.tracking_data.d_matched_features.?,
            self.tracking_data.current_feature_count,
        );

        // Add tracked landmarks as observations
        var tracked_landmarks_count: u32 = 0;

        // Process tracked landmarks from previous frames
        for (0..self.tracking_data.current_feature_count) |i| {
            const feature_idx: u32 = @intCast(i);
            if (self.tracking_data.getLandmarkForFeature(feature_idx)) |landmark_id| {
                if (self.landmarks.get(landmark_id)) |landmark| {
                    // Add observation to existing landmark
                    if (landmark.addObservation(keyframe_id, feature_idx)) {
                        // Add observation to keyframe
                        if (keyframe.addObservation(landmark_id, feature_idx)) {
                            tracked_landmarks_count += 1;
                        }
                    }
                }
            }
        }

        // Create new landmarks from untracked features
        var new_landmarks_count: u32 = 0;

        // For each untracked feature with a valid triangulated position
        for (0..self.tracking_data.current_feature_count) |i| {
            const feature_idx: u32 = @intCast(i);

            // Skip if feature is already tracked
            if (self.tracking_data.getLandmarkForFeature(feature_idx) != null) continue;

            // Skip if position is invalid (depth == 0)
            if (triangulated_positions[feature_idx].y <= 0.0) continue;

            // Create new landmark
            const landmark_id = self.next_landmark_id;
            self.next_landmark_id += 1;

            const landmark = try Landmark.init(
                self.allocator,
                landmark_id,
                triangulated_positions[feature_idx],
                keyframe_id,
                feature_idx,
            );

            try self.landmarks.put(landmark_id, landmark);

            // Add observation to keyframe
            if (keyframe.addObservation(landmark_id, feature_idx)) {
                // Add to tracking data for this frame
                _ = self.tracking_data.addLandmarkMatch(feature_idx, landmark_id);
                new_landmarks_count += 1;
            }
        }

        // Add keyframe to map
        try self.keyframes.put(keyframe_id, keyframe);

        std.debug.print("Created keyframe {}: {} landmarks tracked, {} new landmarks\n", .{ keyframe_id, tracked_landmarks_count, new_landmarks_count });
    }

    /// Check if a new keyframe should be created
    fn shouldCreateKeyframe(self: *Self, current_pose: CameraPose) !bool {
        if (self.keyframes.count() == 0) return true;

        const last_keyframe = self.keyframes.get(self.last_keyframe_id) orelse return true;

        // Check rotation angle
        const rotation = calculateRotationAngle(last_keyframe.pose, current_pose);
        if (rotation > self.config.min_rotation) {
            std.debug.print("Creating keyframe due to rotation: {d:.2} > {d:.2}\n", .{ rotation, self.config.min_rotation });
            return true;
        }

        // TODO: Check translation (would require proper scale)

        // Check feature tracking ratio
        const tracked_ratio = @as(f32, @floatFromInt(self.tracking_data.landmark_count)) /
            @as(f32, @floatFromInt(last_keyframe.observations.count));
        if (tracked_ratio < self.config.min_tracked_ratio) {
            std.debug.print("Creating keyframe due to low tracking ratio: {d:.2} < {d:.2}\n", .{ tracked_ratio, self.config.min_tracked_ratio });
            return true;
        }

        return false;
    }

    /// Calculate rotation angle between poses
    fn calculateRotationAngle(pose1: CameraPose, pose2: CameraPose) f32 {
        // Convert rotations to quaternions
        const q1 = CameraPose.toQuaternion(pose1);
        const q2 = CameraPose.toQuaternion(pose2);

        // Compute dot product between quaternions
        const dot = q1.dot(q2);

        // Clamp to valid range
        const abs_dot = @abs(dot);
        const clamped_dot = if (abs_dot > 1.0) 1.0 else abs_dot;

        // Convert to degrees
        return Math.degrees(2.0 * std.math.acos(clamped_dot));
    }

    /// Remove low-quality landmarks from the map
    // fn pruneMap(self: *Self) !void {
    //     var to_remove = std.ArrayList(u64).init(self.allocator);
    //     defer to_remove.deinit();

    //     var it = self.landmarks.iterator();
    //     while (it.next()) |entry| {
    //         const landmark = entry.value_ptr.*;

    //         // Remove landmarks with insufficient observations or many tracking failures
    //         if (!landmark.isReliable(self.config.min_observations)) {
    //             try to_remove.append(entry.key_ptr.*);
    //         }
    //     }

    //     // Remove the identified landmarks
    //     for (to_remove.items) |id| {
    //         if (self.landmarks.fetchRemove(id)) |kv| {
    //             kv.value.deinit();
    //         }
    //     }

    //     if (to_remove.items.len > 0) {
    //         std.debug.print("Pruned {} low-quality landmarks from map\n", .{to_remove.items.len});
    //     }
    // }

    /// Update the visualization
    fn updateVisualization(self: *Self) !void {
        _ = self;
        // In a real implementation, we would:
        // 1. Create/update a point cloud representation of landmarks
        // 2. Create/update a camera path visualization
        // 3. Apply colors/effects based on tracking quality

        // Since this is a placeholder, we don't implement the visualization details
    }
};

pub const FrameBuffer = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex,
    frame: ?*video.AVFrame,
    is_new: bool,

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .mutex = std.Thread.Mutex{},
            .frame = null,
            .is_new = false,
        };
    }

    pub fn deinit(self: *Self) void {
        self.clear();
    }

    // Update the frame (replacing any existing one)
    pub fn updateFrame(self: *Self, new_frame: *video.AVFrame) !void {
        const cloned_frame = video.av_frame_clone(new_frame);
        errdefer video.av_frame_free(@constCast(@ptrCast(&cloned_frame)));

        self.mutex.lock();
        defer self.mutex.unlock();

        // Free any existing frame
        if (self.frame) |old_frame| {
            video.av_frame_free(@constCast(@ptrCast(&old_frame)));
        }

        self.frame = cloned_frame;
        self.is_new = true;
    }

    // Get the current frame and mark it as used
    pub fn getFrameIfNew(self: *Self) ?*video.AVFrame {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.is_new or self.frame == null) return null;

        // Mark as used but leave the frame in place
        self.is_new = false;
        return self.frame;
    }

    // Clear the frame buffer
    pub fn clear(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.frame) |frame| {
            video.av_frame_free(@constCast(@ptrCast(&frame)));
            self.frame = null;
        }
        self.is_new = false;
    }
};

pub const SceneManager = struct {
    const Self = @This();

    pub const CameraID = enum {
        left,
        right,
    };

    allocator: std.mem.Allocator,

    // Core components
    pose_handler: *Sensors.PoseHandler,
    mapper: *SparseMapper,
    // bundle_adjuster: *BundleAdjuster,

    left_buffer: FrameBuffer,
    right_buffer: FrameBuffer,

    // Visualization
    point_cloud_node: *Node,
    camera_path_node: *Node,

    // Configuration
    optimize_every_n_keyframes: u32 = 3,

    pub fn init(allocator: std.mem.Allocator, environment: *Node, width: u32, height: u32, _stereo_params: ?StereoParams, _temporal_params: ?TemporalParams, pose_handler: *Sensors.PoseHandler) !*Self {
        const temporal_params = _temporal_params orelse DefaultTemporalParams();
        const stereo_params = _stereo_params orelse DefaultStereoParams(width, height);

        // Create visualization nodes
        const point_cloud_node = try InstancedKeypoints.init(allocator, null, stereo_params.max_keypoints * 10);
        try environment.addChild(point_cloud_node);

        const camera_path_node = try Node.init(allocator, null, null, null);
        try environment.addChild(camera_path_node);

        // const point_cloud_mesh = point_cloud_node.mesh.?;
        // const point_cloud_instance = point_cloud_node.instance_data.?;

        // const detection_resources = try DetectionResources.init(
        //     allocator,
        //     0,
        //     stereo_params.max_keypoints * 10,
        //     point_cloud_mesh.textureID.y,
        //     point_cloud_mesh.textureID.uv,
        //     point_cloud_mesh.textureID.depth,
        //     .{
        //         .position = point_cloud_instance.position_buffer,
        //         .color = point_cloud_instance.color_buffer,
        //         .size = stereo_params.max_keypoints * 10,
        //     },
        //     null,
        //     null,
        // );

        // Create mapper with visualization node
        var mapper = try SparseMapper.init(
            allocator,
            .{},
            // detection_resources,
            stereo_params,
            temporal_params,
            environment,
        );
        mapper.visualization_node = point_cloud_node;

        // Create bundle adjuster
        // var bundle_adjuster = try BundleAdjuster.init(allocator, mapper, .{});

        const left_buffer = FrameBuffer.init(allocator);
        const right_buffer = FrameBuffer.init(allocator);

        const manager = try allocator.create(Self);
        manager.* = .{
            .allocator = allocator,
            .pose_handler = pose_handler,
            .mapper = mapper,
            // .bundle_adjuster = bundle_adjuster,
            .left_buffer = left_buffer,
            .right_buffer = right_buffer,
            .point_cloud_node = point_cloud_node,
            .camera_path_node = camera_path_node,
        };
        return manager;
    }

    pub fn deinit(self: *Self) void {
        // self.bundle_adjuster.deinit();
        self.mapper.deinit();
        self.allocator.destroy(self);

        self.left_buffer.deinit();
        self.right_buffer.deinit();
    }

    // Update a camera's frame buffer with a new frame
    pub fn updateFrame(self: *Self, frame: *video.AVFrame, camera_id: CameraID) !void {
        switch (camera_id) {
            .left => try self.left_buffer.updateFrame(frame),
            .right => try self.right_buffer.updateFrame(frame),
        }
    }

    pub fn processFramePair(self: *Self) !void {
        const left_frame_opt = self.left_buffer.getFrameIfNew();
        const right_frame_opt = self.right_buffer.getFrameIfNew();

        // Only process if we have new frames from both cameras
        if (left_frame_opt == null or right_frame_opt == null) {
            // std.debug.print("New frame pair not ready!\n", .{});
            return;
        }

        const left_frame = left_frame_opt.?;
        const right_frame = right_frame_opt.?;

        // Get initial pose estimate from IMU
        const imu_pose = Mat3.from_quaternion(self.pose_handler.node.rotation);
        var pose = CameraPose.init();
        pose.rotation = imu_pose.to_array();

        // Process frame in mapper (feature detection, tracking, keyframe creation)
        try self.mapper.processFrame(left_frame, right_frame, pose);

        // Run bundle adjustment if needed
        // if (self.mapper.last_keyframe_id % self.optimize_every_n_keyframes == 0) {
        //     try self.bundle_adjuster.optimize();
        // }

        // Update visualization
        try self.updateVisualization();
    }

    fn updateVisualization(self: *Self) !void {
        _ = self;
        // Update point cloud visualization
        // try self.updatePointCloudVisualization();

        // Update camera path visualization
        // try self.updateCameraPathVisualization();
    }

    fn updatePointCloudVisualization(self: *Self) !void {
        // Clear existing point cloud
        self.point_cloud_node.clearChildren();

        // Create a new node for each landmark
        var landmark_it = self.mapper.landmarks.iterator();
        while (landmark_it.next()) |entry| {
            const landmark = entry.value_ptr.*;

            // Skip invalid landmarks
            if (!landmark.is_valid) continue;

            // Create a node for this landmark
            // TODO: create Vertex array
            const landmark_node = try Node.init(
                self.allocator,
                landmark.position,
                .{ landmark.color[0], landmark.color[1], landmark.color[2], 255 },
                null,
            );

            // Set size based on observation count for better visibility
            const size_factor = @min(landmark.getObservationCount(), 5) * 0.05 + 0.1;
            landmark_node.scale = .{ size_factor, size_factor, size_factor };

            // Add to point cloud node
            try self.point_cloud_node.addChild(landmark_node);
        }
    }

    fn updateCameraPathVisualization(self: *Self) !void {
        // Clear existing camera path
        self.camera_path_node.clearChildren();

        // Create a new node for each keyframe
        var keyframe_it = self.mapper.keyframes.iterator();
        while (keyframe_it.next()) |entry| {
            const keyframe = entry.value_ptr.*;

            // Convert SE(3) pose to transformation matrix
            const rotation = Mat3.from_array(keyframe.pose.rotation);
            const translation = keyframe.pose.translation;
            const transform = Mat4.from_mat3(rotation).translate(translation[0], translation[1], translation[2]);

            // Create a node for this camera pose
            const camera_node = try Node.init(self.allocator, transform, .{ 0, 255, 0, 255 }, // Green for camera poses
                null);

            // Scale appropriately for visualization
            camera_node.scale = .{ 0.2, 0.2, 0.2 };

            // Add to camera path node
            try self.camera_path_node.addChild(camera_node);
        }

        // Add camera connections to show trajectory
        if (self.mapper.keyframes.count() >= 2) {
            // const prev_keyframe_id: u64 = 0;
            var keyframes_by_id = std.ArrayList(*Keyframe).init(self.allocator);
            defer keyframes_by_id.deinit();

            // Sort keyframes by ID
            {
                var kf_it = self.mapper.keyframes.valueIterator();
                while (kf_it.next()) |keyframe| {
                    try keyframes_by_id.append(keyframe.*);
                }

                // Sort by ID
                std.sort.sort(*Keyframe, keyframes_by_id.items, {}, struct {
                    fn lessThan(_: void, a: *Keyframe, b: *Keyframe) bool {
                        return a.id < b.id;
                    }
                }.lessThan);
            }

            // Create connections between consecutive keyframes
            for (1..keyframes_by_id.items.len) |i| {
                const prev_keyframe = keyframes_by_id.items[i - 1];
                const curr_keyframe = keyframes_by_id.items[i];

                // Get positions
                const prev_pos = Vec3.init(prev_keyframe.pose.translation[0], prev_keyframe.pose.translation[1], prev_keyframe.pose.translation[2]);

                const curr_pos = Vec3.init(curr_keyframe.pose.translation[0], curr_keyframe.pose.translation[1], curr_keyframe.pose.translation[2]);

                // Create a line node between them
                const line_node = try Node.init(self.allocator, Mat4.identity(), .{ 255, 255, 0, 128 }, // Yellow, semi-transparent
                    null);

                // Set line properties
                line_node.line_start = prev_pos;
                line_node.line_end = curr_pos;
                line_node.is_line = true;

                // Add to camera path node
                try self.camera_path_node.addChild(line_node);
            }
        }
    }
};
