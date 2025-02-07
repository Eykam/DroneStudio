const std = @import("std");
const imgui = @import("bindings/c.zig").imgui;
const Scene = @import("Pipeline.zig").Scene;
const Vision = @import("Vision.zig");
const StereoVO = Vision.StereoVO;
const CameraPose = Vision.CameraPose;

const UIContext = struct {
    scene: *Scene,
    StereoVO: *StereoVO,
};

fn createWindowsStructType(Windows: []const type) type {
    var fields: [Windows.len]std.builtin.Type.StructField = undefined;

    for (Windows, 0..) |T, i| {
        if (!@hasDecl(T, "init")) {
            @compileError("Window type must init draw()");
        }

        if (!@hasDecl(T, "draw")) {
            @compileError("Window type must implement draw()");
        }

        if (!@hasDecl(T, "deinit")) {
            @compileError("Window type must deinit draw()");
        }

        fields[i] = .{
            .name = @typeName(T),
            .type = *T,
            .default_value = null,
            .is_comptime = false,
            .alignment = @alignOf(*T),
        };
    }

    return @Type(.{
        .@"struct" = .{
            .layout = .auto,
            .fields = &fields,
            .decls = &[_]std.builtin.Type.Declaration{},
            .is_tuple = false,
        },
    });
}

pub fn WindowManager(comptime Windows: []const type) type {
    const TWindows = createWindowsStructType(Windows);

    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        windows: TWindows,
        context: UIContext,

        pub fn init(allocator: std.mem.Allocator, context: UIContext) !Self {
            var windows: TWindows = undefined;
            inline for (Windows) |W| {
                @field(windows, @typeName(W)) = try W.init(allocator);
            }

            return Self{
                .allocator = allocator,
                .windows = windows,
                .context = context,
            };
        }

        pub fn deinit(self: *Self) void {
            inline for (Windows) |W| {
                @field(self.windows, @typeName(W)).deinit(self.allocator);
            }
        }

        pub fn drawAll(self: Self) void {
            imgui.ImGui_ImplOpenGL3_NewFrame();
            imgui.ImGui_ImplGlfw_NewFrame();
            imgui.igNewFrame();

            inline for (Windows) |W| {
                @field(self.windows, @typeName(W)).draw(&self.context);
            }
        }
    };
}

pub const OverlayWindow = struct {
    visible: bool,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .visible = true,
        };
        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.destroy(self);
    }

    pub fn draw(self: *Self, ctx: *const UIContext) void {
        if (!self.visible) return;

        // Window flags setup
        var window_flags = imgui.ImGuiWindowFlags_NoDecoration |
            imgui.ImGuiWindowFlags_NoDocking |
            imgui.ImGuiWindowFlags_AlwaysAutoResize |
            imgui.ImGuiWindowFlags_NoSavedSettings |
            imgui.ImGuiWindowFlags_NoFocusOnAppearing |
            imgui.ImGuiWindowFlags_NoNav;

        // Window positioning
        const PAD = 10.0;
        const viewport = imgui.igGetMainViewport();
        const work_pos = viewport.*.WorkPos;
        const window_pos = imgui.ImVec2{
            .x = work_pos.x + PAD,
            .y = work_pos.y + PAD,
        };
        const window_pos_pivot = imgui.ImVec2{
            .x = 0.0,
            .y = 0.0,
        };

        // Window setup
        imgui.igSetNextWindowPos(window_pos, imgui.ImGuiCond_Always, window_pos_pivot);
        imgui.igSetNextWindowViewport(viewport.*.ID);
        window_flags |= imgui.ImGuiWindowFlags_NoMove;
        imgui.igSetNextWindowBgAlpha(0.35);

        // Window content
        if (imgui.igBegin("FPS Counter", &self.visible, window_flags)) {
            imgui.igText("FPS Counter");
            imgui.igSeparator();
            imgui.igText("%.1f FPS\n%.3f Frame time (ms)", 1000.0 / ctx.scene.avg_frame_time, ctx.scene.avg_frame_time);
        }
        imgui.igEnd();
    }

    pub fn show(self: *Self) void {
        self.visible = true;
    }

    pub fn hide(self: *Self) void {
        self.visible = false;
    }

    pub fn toggle(self: *Self) void {
        self.visible = !self.visible;
    }
};

pub const StereoDebugWindow = struct {
    visible: bool,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .visible = true,
        };
        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.destroy(self);
    }

    pub fn draw(self: *Self, ctx: *const UIContext) void {
        if (!self.visible) return;

        // Make window tall enough to fit all sections without scrolling
        const window_flags = imgui.ImGuiWindowFlags_None;
        imgui.igSetNextWindowSize(.{ .x = 400, .y = 600 }, imgui.ImGuiCond_FirstUseEver);

        if (imgui.igBegin("StereoVO Debug", &self.visible, window_flags)) {
            // Statistics Section
            imgui.igText("Statistics");
            imgui.igSeparator();
            imgui.igText("Current Matches: %d", ctx.StereoVO.num_matches.*);
            imgui.igText("Left Keypoints: %d", ctx.StereoVO.left.num_keypoints.*);
            imgui.igText("Right Keypoints: %d", ctx.StereoVO.right.num_keypoints.*);
            if (ctx.StereoVO.left.frame) |frame| {
                imgui.igText("Frame dimensions: %dx%d", frame.width, frame.height);
            }
            imgui.igNewLine();

            var params = ctx.StereoVO.params;
            var params_changed = false;

            // Camera Parameters Section
            imgui.igText("Camera Parameters");
            imgui.igSeparator();
            imgui.igText("Baseline (mm): %.2f", params.baseline_mm);
            imgui.igText("Focal Length (mm): %.2f", params.focal_length_mm);
            imgui.igNewLine();

            const rad_to_deg = 180.0 / std.math.pi;

            imgui.igSeparator();
            if (imgui.igButton("Reset Pose", .{ .x = 0, .y = 0 })) {
                ctx.StereoVO.resetPose() catch {};
            }
            imgui.igText("Global Pose");

            const global_pose = ctx.StereoVO.global_pose;

            // Translation
            const global_pos = global_pose.translation;
            imgui.igText("Position:");
            imgui.igText("  X: %.3f", global_pos[0]);
            imgui.igText("  Y: %.3f", global_pos[1]);
            imgui.igText("  Z: %.3f", global_pos[2]);

            // Convert rotation matrix to Euler angles
            const global_rot = global_pose.rotation;

            const global_pitch = std.math.atan2(-global_rot[2], @sqrt(global_rot[6] * global_rot[6] + global_rot[8] * global_rot[8]));
            const global_yaw = std.math.atan2(global_rot[6], global_rot[8]);
            const global_roll = std.math.atan2(global_rot[1], global_rot[0]);

            // Convert radians to degrees
            imgui.igText("Rotation (degrees):");
            imgui.igText("  Roll:  %.2f", global_roll * rad_to_deg);
            imgui.igText("  Pitch: %.2f", global_pitch * rad_to_deg);
            imgui.igText("  Yaw:   %.2f", global_yaw * rad_to_deg);

            imgui.igSeparator();
            imgui.igText("Delta Pose");

            const delta_pose = ctx.StereoVO.delta_pose orelse CameraPose.init();

            // Translation
            const delta_pos = delta_pose.translation;
            imgui.igText("Position:");
            imgui.igText("  X: %.3f", delta_pos[0]);
            imgui.igText("  Y: %.3f", delta_pos[1]);
            imgui.igText("  Z: %.3f", delta_pos[2]);

            // Convert rotation matrix to Euler angles
            const delta_rot = delta_pose.rotation;

            const delta_pitch = std.math.atan2(-delta_rot[2], @sqrt(delta_rot[6] * delta_rot[6] + delta_rot[8] * delta_rot[8]));
            const delta_yaw = std.math.atan2(delta_rot[6], delta_rot[8]);
            const delta_roll = std.math.atan2(delta_rot[1], delta_rot[0]);

            // Convert radians to degrees
            imgui.igText("Rotation (degrees):");
            imgui.igText("  Roll:  %.2f", delta_roll * rad_to_deg);
            imgui.igText("  Pitch: %.2f", delta_pitch * rad_to_deg);
            imgui.igText("  Yaw:   %.2f", delta_yaw * rad_to_deg);

            var disable_spatial_tracking = params.disable_spatial_tracking;
            if (imgui.igCheckbox("Disable Spatial Tracking", &disable_spatial_tracking)) {
                params.disable_spatial_tracking = disable_spatial_tracking;
                params_changed = true;
            }
            imgui.igNewLine();

            // Keypoint Detection Parameters Section
            if (imgui.igCollapsingHeader_BoolPtr("Keypoint Detection Parameters", null, 0)) {
                var intensity = params.intensity_threshold;
                if (imgui.igSliderScalar(
                    "Intensity Threshold",
                    imgui.ImGuiDataType_U8,
                    &intensity,
                    &@as(u8, 1),
                    &@as(u8, 50),
                    "%u",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.intensity_threshold = intensity;
                    params_changed = true;
                }

                var radius = params.circle_radius;
                if (imgui.igSliderScalar(
                    "Circle Radius",
                    imgui.ImGuiDataType_U32,
                    &radius,
                    &@as(u32, 1),
                    &@as(u32, 10),
                    "%u",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.circle_radius = radius;
                    params_changed = true;
                }

                var arc_length = params.arc_length;
                if (imgui.igSliderScalar(
                    "Arc Length",
                    imgui.ImGuiDataType_U32,
                    &arc_length,
                    &@as(u32, 1),
                    &@as(u32, 16),
                    "%u",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.arc_length = arc_length;
                    params_changed = true;
                }

                var max_keypoints = params.max_keypoints;
                if (imgui.igSliderScalar(
                    "Max Keypoints",
                    imgui.ImGuiDataType_U32,
                    &max_keypoints,
                    &@as(u32, 1),
                    &@as(u32, 100000),
                    "%u",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.max_keypoints = max_keypoints;
                    params_changed = true;
                }

                var sigma = params.sigma;
                if (imgui.igSliderFloat(
                    "Sigma",
                    &sigma,
                    0.01,
                    2.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.sigma = sigma;
                    params_changed = true;
                }
                imgui.igNewLine();
            }

            // Matching Parameters Section
            if (imgui.igCollapsingHeader_BoolPtr("Matching Parameters", null, 0)) {
                var disable_matching = params.disable_matching;
                var show_connections = params.show_connections;
                var disable_depth = params.disable_depth;

                if (imgui.igCheckbox("Disable Matching", &disable_matching)) {
                    show_connections = false;
                    params_changed = true;
                    params.disable_matching = disable_matching;
                    params_changed = true;
                }

                if (imgui.igCheckbox("Show Connections", &show_connections)) {
                    params.show_connections = show_connections;
                    params_changed = true;
                }

                if (imgui.igCheckbox("Disable Depth", &disable_depth)) {
                    params.disable_depth = disable_depth;
                    params_changed = true;
                }
                imgui.igSeparator();

                var max_disparity = params.max_disparity;
                if (imgui.igSliderFloat(
                    "Max Disparity",
                    &max_disparity,
                    1.0,
                    300.0,
                    "%.1f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.max_disparity = max_disparity;
                    params_changed = true;
                }

                var epipolar = params.epipolar_threshold;
                if (imgui.igSliderFloat(
                    "Epipolar Threshold",
                    &epipolar,
                    1.0,
                    100.0,
                    "%.1f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.epipolar_threshold = epipolar;
                    params_changed = true;
                }

                var max_hamming_dist = params.max_hamming_dist;
                if (imgui.igSliderFloat(
                    "Hamming Distance Threshold",
                    &max_hamming_dist,
                    0.01,
                    1.0,
                    "%.1f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.max_hamming_dist = max_hamming_dist;
                    params_changed = true;
                }

                var lowe_ratio = params.lowes_ratio;
                if (imgui.igSliderFloat(
                    "Lowe's Ratio Threshold",
                    &lowe_ratio,
                    0.01,
                    1.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.lowes_ratio = lowe_ratio;
                    params_changed = true;
                }

                var cost_threshold = params.cost_threshold;
                if (imgui.igSliderFloat(
                    "Cost Threshold",
                    &cost_threshold,
                    0.01,
                    1.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.cost_threshold = cost_threshold;
                    params_changed = true;
                }

                imgui.igNewLine();
                imgui.igText("Cost Weights (should sum to 1.0)");

                var epipolar_weight = params.epipolar_weight;
                if (imgui.igSliderFloat(
                    "Epipolar Weight",
                    &epipolar_weight,
                    0.0,
                    1.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.epipolar_weight = epipolar_weight;
                    params_changed = true;
                }

                var disparity_weight = params.disparity_weight;
                if (imgui.igSliderFloat(
                    "Disparity Weight",
                    &disparity_weight,
                    0.0,
                    1.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.disparity_weight = disparity_weight;
                    params_changed = true;
                }

                var hamming_dist_weight = params.hamming_dist_weight;
                if (imgui.igSliderFloat(
                    "Hamming Distance Weight",
                    &hamming_dist_weight,
                    0.0,
                    1.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    params.hamming_dist_weight = hamming_dist_weight;
                    params_changed = true;
                }

                // Display total weight
                const total_weight = epipolar_weight + disparity_weight + hamming_dist_weight;
                imgui.igText("Total Weight: %.2f", total_weight);
                if (@abs(total_weight - 1.0) > 0.001) {
                    imgui.igTextColored(.{ .x = 1.0, .y = 0.0, .z = 0.0, .w = 1.0 }, "Warning: Weights should sum to 1.0");
                }
            }

            // Temporal Parameters Section

            if (imgui.igCollapsingHeader_BoolPtr("Temporal Parameters", null, 0)) {
                imgui.igText("Temporal Parameters");
                imgui.igSeparator();

                var temporal = ctx.StereoVO.temporal_params;

                var max_distance = temporal.max_distance;
                if (imgui.igSliderFloat(
                    "Max Distance",
                    &max_distance,
                    0.01,
                    5.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.max_distance = max_distance;
                    params_changed = true;
                }

                var max_pixel_distance = temporal.max_pixel_distance;
                if (imgui.igSliderFloat(
                    "Max Pixel Distance",
                    &max_pixel_distance,
                    1.0,
                    200.0,
                    "%.1f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.max_pixel_distance = max_pixel_distance;
                    params_changed = true;
                }

                var min_confidence = temporal.min_confidence;
                if (imgui.igSliderFloat(
                    "Min Confidence",
                    &min_confidence,
                    0.0,
                    1.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.min_confidence = min_confidence;
                    params_changed = true;
                }

                var min_matches = temporal.min_matches;
                if (imgui.igSliderScalar(
                    "Min Matches",
                    imgui.ImGuiDataType_U32,
                    &min_matches,
                    &@as(u32, 3),
                    &@as(u32, 1000),
                    "%u",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.min_matches = min_matches;
                    params_changed = true;
                }

                var ransac_threshold = temporal.ransac_threshold;
                if (imgui.igSliderFloat(
                    "RANSAC Threshold",
                    &ransac_threshold,
                    0.001,
                    0.1,
                    "%.3f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.ransac_threshold = ransac_threshold;
                    params_changed = true;
                }

                var ransac_iterations = temporal.ransac_iterations;
                if (imgui.igSliderScalar(
                    "RANSAC Iterations",
                    imgui.ImGuiDataType_U32,
                    &ransac_iterations,
                    &@as(u32, 10),
                    &@as(u32, 1000),
                    "%u",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.ransac_iterations = ransac_iterations;
                    params_changed = true;
                }

                var ransac_points = temporal.ransac_points;
                if (imgui.igSliderScalar(
                    "RANSAC Points",
                    imgui.ImGuiDataType_U32,
                    &ransac_points,
                    &@as(u32, 3),
                    &@as(u32, 16),
                    "%u",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.ransac_points = ransac_points;
                    params_changed = true;
                }

                imgui.igNewLine();
                imgui.igText("Cost Weights (should sum to 1.0)");

                var spatial_weight = temporal.spatial_weight;
                if (imgui.igSliderFloat(
                    "Spatial Weight",
                    &spatial_weight,
                    0.0,
                    1.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.spatial_weight = spatial_weight;
                    params_changed = true;
                }

                var hamming_weight = temporal.hamming_weight;
                if (imgui.igSliderFloat(
                    "Hamming Weight",
                    &hamming_weight,
                    0.0,
                    1.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.hamming_weight = hamming_weight;
                    params_changed = true;
                }

                var img_weight = temporal.img_weight;
                if (imgui.igSliderFloat(
                    "Image Weight",
                    &img_weight,
                    0.0,
                    1.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.img_weight = img_weight;
                    params_changed = true;
                }

                // Display total weight
                const total_temporal_weight = spatial_weight + hamming_weight + img_weight;
                imgui.igText("Total Weight: %.2f", total_temporal_weight);
                if (@abs(total_temporal_weight - 1.0) > 0.001) {
                    imgui.igTextColored(.{ .x = 1.0, .y = 0.0, .z = 0.0, .w = 1.0 }, "Warning: Weights should sum to 1.0");
                }

                imgui.igNewLine();
                imgui.igText("Other Parameters");
                imgui.igSeparator();

                var max_hamming_dist = temporal.max_hamming_dist;
                if (imgui.igSliderFloat(
                    "Max Hamming Distance",
                    &max_hamming_dist,
                    0.0,
                    1.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.max_hamming_dist = max_hamming_dist;
                    params_changed = true;
                }

                var cost_threshold = temporal.cost_threshold;
                if (imgui.igSliderFloat(
                    "Cost Threshold",
                    &cost_threshold,
                    0.0,
                    1.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.cost_threshold = cost_threshold;
                    params_changed = true;
                }

                var lowes_ratio = temporal.lowes_ratio;
                if (imgui.igSliderFloat(
                    "Lowe's Ratio",
                    &lowes_ratio,
                    0.0,
                    1.0,
                    "%.2f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    temporal.lowes_ratio = lowes_ratio;
                    params_changed = true;
                }
            }

            ctx.StereoVO.params_changed = params_changed;
        }
        imgui.igEnd();
    }

    pub fn show(self: *Self) void {
        self.visible = true;
    }

    pub fn hide(self: *Self) void {
        self.visible = false;
    }

    pub fn toggle(self: *Self) void {
        self.visible = !self.visible;
    }
};
