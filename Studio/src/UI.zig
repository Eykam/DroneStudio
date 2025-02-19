const std = @import("std");
const imgui = @import("bindings/c.zig").imgui;
const Scene = @import("Pipeline.zig").Scene;
const Vision = @import("Vision.zig");
const Drone = @import("Drone.zig");

const DroneConfig = Drone.DroneConfig;
const MotorController = Drone.MotorController;
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

const Styles = struct {
    const StyleState = struct {
        normal: imgui.ImVec4,
        hovered: imgui.ImVec4,
        active: imgui.ImVec4,
    };

    const selected: StyleState = StyleState{
        .normal = .{ .x = 0.2, .y = 0.6, .z = 0.2, .w = 1.0 },
        .hovered = .{ .x = 0.3, .y = 0.7, .z = 0.3, .w = 1.0 },
        .active = .{ .x = 0.4, .y = 0.8, .z = 0.4, .w = 1.0 },
    };
    const unselected = StyleState{
        .normal = .{ .x = 0.5, .y = 0.5, .z = 0.5, .w = 0.6 },
        .hovered = .{ .x = 0.6, .y = 0.6, .z = 0.6, .w = 0.7 },
        .active = .{ .x = 0.7, .y = 0.7, .z = 0.7, .w = 0.8 },
    };

    fn pushButtonColors(is_selected: bool) void {
        const style = if (is_selected) Styles.selected else Styles.unselected;
        imgui.igPushStyleColor_Vec4(imgui.ImGuiCol_Button, style.normal);
        imgui.igPushStyleColor_Vec4(imgui.ImGuiCol_ButtonHovered, style.hovered);
        imgui.igPushStyleColor_Vec4(imgui.ImGuiCol_ButtonActive, style.active);
    }

    fn popButtonColors() void {
        imgui.igPopStyleColor(3);
    }
};

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

pub const DroneConfigWindow = struct {
    const Self = @This();

    const max_ip_len = 16;
    const max_path_len = 256;
    const max_cwd_len = 4096;

    local_ip_buffer: [max_ip_len]u8 = undefined,
    controller_ip_buffer: [max_ip_len]u8 = undefined,
    config_path_buffer: [max_path_len]u8 = undefined,
    cwd_buffer: [max_cwd_len]u8 = undefined,

    visible: bool,
    show_save_modal: bool = false,
    show_load_modal: bool = false,

    selected_motor: MotorController.Motors,
    global_testing: bool = true,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .visible = true,
            .selected_motor = MotorController.Motors.Motor_1,
            .allocator = allocator,
        };

        @memset(&self.local_ip_buffer, 0);
        @memset(&self.controller_ip_buffer, 0);
        @memset(&self.config_path_buffer, 0);
        @memset(&self.cwd_buffer, 0);

        // const local_ip = DroneConfig.default_local_ip;
        // const controller_ip = DroneConfig.default_controller_ip;
        // const config_path = DroneConfig.default_config_path;

        // @memcpy(self.local_ip_buffer[0..local_ip.len], local_ip);
        // @memcpy(self.controller_ip_buffer[0..controller_ip.len], controller_ip);
        // @memcpy(self.config_path_buffer[0..config_path.len], config_path);
        // Copy IP to buffer
        // std.mem.copyForwards(u8, &self.ip_buffer, self.config.ip);

        // // Try to load saved config
        // self.config.loadFromFile() catch |err| {
        //     std.debug.print("Failed to load config: {}\n", .{err});
        // };

        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.allocator.free(self.local_ip_buffer);
        self.allocator.free(self.controller_ip_buffer);
        self.allocator.free(self.config_path_buffer);
        allocator.destroy(self);
    }

    pub fn draw(self: *Self, ctx: *const UIContext) void {
        if (!self.visible) return;

        const window_size = imgui.ImVec2{ .x = 400.0, .y = 600.0 };
        imgui.igSetNextWindowSize(window_size, imgui.ImGuiCond_FirstUseEver);

        const PAD = 250.0;
        const viewport = imgui.igGetMainViewport();
        const work_pos = viewport.*.WorkPos;
        const work_size = viewport.*.WorkSize;

        // Calculate position for top right
        const window_pos = imgui.ImVec2{
            .x = work_pos.x + work_size.x - (window_size.x + PAD),
            .y = work_pos.y + (work_size.y / 2) - (window_size.y / 2),
        };

        const window_pos_pivot = imgui.ImVec2{ .x = 0.0, .y = 0.0 };

        imgui.igSetNextWindowPos(window_pos, imgui.ImGuiCond_Always, window_pos_pivot);
        imgui.igSetNextWindowViewport(viewport.*.ID);

        if (imgui.igBegin("Drone Configuration", &self.visible, imgui.ImGuiWindowFlags_None)) {
            // Fixed Header Section
            self.drawFixedHeader(ctx);
            imgui.igSeparator();

            // Server Configuration Section
            if (imgui.igCollapsingHeader_BoolPtr("Server Configuration", null, imgui.ImGuiTreeNodeFlags_DefaultOpen)) {
                self.drawServerConfig(ctx);
            }

            imgui.igNewLine();
            imgui.igSeparator();

            // Protocol Configuration Section
            if (imgui.igCollapsingHeader_BoolPtr("Protocol Configuration", null, imgui.ImGuiTreeNodeFlags_DefaultOpen)) {
                self.drawProtocolConfig(ctx);
            }

            imgui.igNewLine();
            imgui.igSeparator();

            // Motor Configuration Section
            if (imgui.igCollapsingHeader_BoolPtr("Motor Configuration", null, imgui.ImGuiTreeNodeFlags_DefaultOpen)) {
                self.drawMotorConfig(ctx);
            }

            imgui.igNewLine();
            imgui.igSeparator();

            // Throttle Testing Section
            if (imgui.igCollapsingHeader_BoolPtr("Throttle Testing", null, imgui.ImGuiTreeNodeFlags_DefaultOpen)) {
                self.drawThrottleTesting(ctx);
            }

            // Modal Windows
            self.drawSaveModal(ctx);
            self.drawLoadModal(ctx);
        }
        imgui.igEnd();
    }

    inline fn drawFixedHeader(self: *Self, ctx: *const UIContext) void {
        // Config Path Display
        const curr_path = std.mem.sliceTo(&self.config_path_buffer, 0);
        imgui.igText("Config Path: %s", @as([*c]const u8, curr_path.ptr));

        // Save/Load Buttons
        const button_disabled = ctx.scene.motor_controller.getConnectionState() == .Running;

        if (button_disabled) {
            imgui.igPushStyleVar_Float(imgui.ImGuiStyleVar_Alpha, 0.5);
        }

        if (imgui.igButton("Save Configuration", imgui.ImVec2{ .x = 120, .y = 0 }) and !button_disabled) {
            self.show_save_modal = true;
        }
        imgui.igSameLine(0, 4);
        if (imgui.igButton("Load Configuration", imgui.ImVec2{ .x = 120, .y = 0 }) and !button_disabled) {
            self.show_load_modal = true;
        }

        if (button_disabled) {
            imgui.igPopStyleVar(1);
        }

        imgui.igNewLine();

        // Connection Status and Controls
        const conn_state = ctx.scene.motor_controller.getConnectionState();
        const state_str = conn_state.toString();

        // Color-coded status
        const status_color = switch (conn_state) {
            .Disconnected => imgui.ImVec4{ .x = 0.7, .y = 0.7, .z = 0.7, .w = 1.0 },
            .Connecting => imgui.ImVec4{ .x = 1.0, .y = 0.7, .z = 0.0, .w = 1.0 },
            .Connected, .ConfigSync => imgui.ImVec4{ .x = 0.0, .y = 0.7, .z = 1.0, .w = 1.0 },
            .Ready => imgui.ImVec4{ .x = 0.0, .y = 1.0, .z = 0.0, .w = 1.0 },
            .Running => imgui.ImVec4{ .x = 0.0, .y = 0.8, .z = 0.0, .w = 1.0 },
            .Failed => imgui.ImVec4{ .x = 1.0, .y = 0.0, .z = 0.0, .w = 1.0 },
        };

        imgui.igTextColored(status_color, "Status: %s", state_str.ptr);
        imgui.igSameLine(0, 10);

        // Connection control buttons
        switch (conn_state) {
            .Disconnected => {
                if (imgui.igButton("Connect", imgui.ImVec2{ .x = 120, .y = 0 })) {
                    ctx.scene.motor_controller.connect() catch |err| {
                        std.debug.print("Failed to connect: {}\n", .{err});
                    };
                }
            },
            .Failed => {
                if (imgui.igButton("Retry", imgui.ImVec2{ .x = 120, .y = 0 })) {
                    ctx.scene.motor_controller.*.retryConnection() catch |err| {
                        std.debug.print("Failed to retry connection: {}\n", .{err});
                    };
                }
            },
            .Connected, .ConfigSync, .Ready, .Running => {
                if (imgui.igButton("Disconnect", imgui.ImVec2{ .x = 120, .y = 0 })) {
                    ctx.scene.motor_controller.disconnect();
                }
            },
            else => {},
        }

        // Config editing lock warning
        if (conn_state == .Running) {
            imgui.igSameLine(0, 10);
            imgui.igTextColored(
                imgui.ImVec4{ .x = 1.0, .y = 0.7, .z = 0.0, .w = 1.0 },
                "(Configuration locked while running)",
            );
        }
        const config = ctx.scene.motor_controller.config;

        const local_ip = self.allocator.dupeZ(u8, config.local_ip) catch {
            std.debug.print("Failed to allocate mem for local ip string\n", .{});
            return;
        };
        defer self.allocator.free(local_ip);

        const controller_ip = self.allocator.dupeZ(u8, config.controller_ip) catch {
            std.debug.print("Failed to allocate mem for controller ip string\n", .{});
            return;
        };
        defer self.allocator.free(controller_ip);

        imgui.igText("Local Address: %s:%d", local_ip.ptr, config.local_port);
        imgui.igText("Controller Address: %s:%d", controller_ip.ptr, config.controller_port);
    }

    inline fn drawThrottleTesting(self: *Self, ctx: *const UIContext) void {
        if (ctx.scene.motor_controller.getConnectionState() != .Running and ctx.scene.motor_controller.getConnectionState() != .Ready) {
            imgui.igText("Please connect to controller to initiate testing!");
            return;
        }

        // Testing Mode Toggle Button
        const is_global = self.global_testing;
        Styles.pushButtonColors(is_global);
        if (imgui.igButton(
            "Global Testing",
            imgui.ImVec2{ .x = 120, .y = 0 },
        )) {
            self.global_testing = true;
        }
        Styles.popButtonColors();

        imgui.igSameLine(0, 10);

        Styles.pushButtonColors(!is_global);
        if (imgui.igButton(
            "Per Motor Testing",
            imgui.ImVec2{ .x = 120, .y = 0 },
        )) {
            self.global_testing = false;
        }
        Styles.popButtonColors();

        imgui.igNewLine();
        imgui.igSpacing();

        const motor_states = ctx.scene.motor_controller.motor_states;

        if (self.global_testing) {
            // Single slider for all motors
            var throttle: f32 = 0.0;
            if (imgui.igSliderFloat(
                "All Motors Throttle (%)",
                &throttle,
                0.0,
                ctx.scene.motor_controller.config.global_max_throttle,
                "%.1f",
                imgui.ImGuiSliderFlags_None,
            )) {
                inline for (@typeInfo(MotorController.Motors).@"enum".fields) |motor| {
                    if (motor_states[motor.value].armed) {
                        const command = MotorController.Command{
                            .kind = .SetSpeed,
                            .speed = throttle,
                            .motor = @enumFromInt(motor.value),
                        };
                        _ = ctx.scene.motor_controller.command_queue.push(command);
                    }
                }
            }

            // Add global arm/disarm buttons
            if (imgui.igButton(
                "Arm All",
                imgui.ImVec2{ .x = 120, .y = 0 },
            )) {
                inline for (@typeInfo(MotorController.Motors).@"enum".fields) |motor| {
                    if (!motor_states[motor.value].armed) {
                        const command = MotorController.Command{
                            .kind = .Arm,
                            .speed = 0,
                            .motor = @enumFromInt(motor.value),
                        };
                        _ = ctx.scene.motor_controller.command_queue.push(command);
                    }
                }
            }

            imgui.igSameLine(0, 10);

            if (imgui.igButton(
                "Disarm All",
                imgui.ImVec2{ .x = 120, .y = 0 },
            )) {
                inline for (@typeInfo(MotorController.Motors).@"enum".fields) |motor| {
                    if (motor_states[motor.value].armed) {
                        const command = MotorController.Command{
                            .kind = .Disarm,
                            .speed = 0,
                            .motor = @enumFromInt(motor.value),
                        };
                        _ = ctx.scene.motor_controller.command_queue.push(command);
                    }
                }
            }
        } else {
            // Individual sliders for each motor
            inline for (@typeInfo(MotorController.Motors).@"enum".fields) |motor| {
                const motor_enum = @as(MotorController.Motors, @enumFromInt(motor.value));
                const motor_state = &motor_states[motor.value];

                // Motor header with status
                const status_color = if (motor_state.armed)
                    imgui.ImVec4{ .x = 0.0, .y = 1.0, .z = 0.0, .w = 1.0 }
                else
                    imgui.ImVec4{ .x = 1.0, .y = 0.0, .z = 0.0, .w = 1.0 };

                const armed_status = if (motor_state.armed) std.fmt.comptimePrint("{s} Armed", .{
                    @tagName(motor_enum).ptr,
                }) else std.fmt.comptimePrint("{s} Disarmed", .{
                    @tagName(motor_enum).ptr,
                });

                imgui.igTextColored(status_color, armed_status);

                // Arm/Disarm button
                const button_label = if (motor_state.armed) std.fmt.comptimePrint("Disarm {d}", .{
                    motor.value + 1,
                }) else std.fmt.comptimePrint("Arm {d}", .{
                    motor.value + 1,
                });

                if (imgui.igButton(
                    button_label,
                    imgui.ImVec2{ .x = 80, .y = 0 },
                )) {
                    const command = MotorController.Command{
                        .kind = if (motor_state.armed) .Disarm else .Arm,
                        .speed = 0,
                        .motor = motor_enum,
                    };
                    _ = ctx.scene.motor_controller.command_queue.push(command);
                }

                imgui.igSameLine(0, 10);

                // Throttle slider (only enabled if armed)
                if (!motor_state.armed) {
                    imgui.igPushStyleVar_Float(imgui.ImGuiStyleVar_Alpha, 0.5);
                }

                var throttle = motor_state.throttle;
                if (imgui.igSliderFloat(
                    std.fmt.comptimePrint("{s} Throttle (%)", .{@tagName(motor_enum)}),
                    &throttle,
                    0.0,
                    ctx.scene.motor_controller.config.global_max_throttle,
                    "%.1f",
                    imgui.ImGuiSliderFlags_None,
                )) {
                    if (motor_state.armed) {
                        const command = MotorController.Command{
                            .kind = .SetSpeed,
                            .speed = throttle,
                            .motor = motor_enum,
                        };
                        _ = ctx.scene.motor_controller.command_queue.push(command);
                    }
                }

                if (!motor_state.armed) {
                    imgui.igPopStyleVar(1);
                }

                imgui.igSpacing();
            }
        }
    }

    inline fn drawServerConfig(self: *Self, ctx: *const UIContext) void {
        const conn_state = ctx.scene.motor_controller.getConnectionState();
        const config_locked = conn_state == .Running or conn_state == .Ready;

        if (config_locked) {
            imgui.igPushStyleVar_Float(imgui.ImGuiStyleVar_Alpha, 0.5);
        }

        // Local Server Configuration
        imgui.igText("Local Server");
        if (imgui.igInputText(
            "Local IP Address",
            &self.local_ip_buffer,
            self.local_ip_buffer.len,
            imgui.ImGuiInputTextFlags_None |
                imgui.ImGuiInputTextFlags_CharsNoBlank |
                imgui.ImGuiInputTextFlags_EnterReturnsTrue |
                if (config_locked) imgui.ImGuiInputTextFlags_ReadOnly else 0,
            null,
            null,
        )) {
            if (!config_locked) {
                const ip_len = std.mem.indexOfScalar(u8, &self.local_ip_buffer, 0) orelse self.local_ip_buffer.len;
                ctx.scene.motor_controller.config.allocator.free(ctx.scene.motor_controller.config.local_ip);
                ctx.scene.motor_controller.config.local_ip = ctx.scene.motor_controller.config.allocator.dupe(u8, self.local_ip_buffer[0..ip_len]) catch |err| {
                    std.debug.print("Failed to allocate memory for IP: {}\n", .{err});
                    return;
                };
            }
        }

        var local_port = @as(c_int, ctx.scene.motor_controller.config.local_port);
        if (imgui.igInputInt(
            "Local Port",
            &local_port,
            0,
            0,
            if (config_locked) imgui.ImGuiInputTextFlags_ReadOnly else imgui.ImGuiInputTextFlags_None,
        )) {
            if (!config_locked) {
                ctx.scene.motor_controller.config.local_port = @intCast(@max(0, @min(local_port, 65535)));
            }
        }

        // Controller Server Configuration
        imgui.igNewLine();
        imgui.igText("Controller Server");
        if (imgui.igInputText(
            "Controller's IP Address",
            &self.controller_ip_buffer,
            self.controller_ip_buffer.len,
            imgui.ImGuiInputTextFlags_None |
                if (config_locked) imgui.ImGuiInputTextFlags_ReadOnly else 0,
            null,
            null,
        )) {
            if (!config_locked) {
                const ip_len = std.mem.indexOfScalar(u8, &self.controller_ip_buffer, 0) orelse self.controller_ip_buffer.len;
                ctx.scene.motor_controller.config.allocator.free(ctx.scene.motor_controller.config.controller_ip);
                ctx.scene.motor_controller.config.controller_ip = ctx.scene.motor_controller.config.allocator.dupe(u8, self.controller_ip_buffer[0..ip_len]) catch |err| {
                    std.debug.print("Failed to allocate memory for IP: {}\n", .{err});
                    return;
                };
            }
        }

        var controller_port = @as(c_int, ctx.scene.motor_controller.config.controller_port);
        if (imgui.igInputInt(
            "Controller Port",
            &controller_port,
            0,
            0,
            if (config_locked) imgui.ImGuiInputTextFlags_ReadOnly else imgui.ImGuiInputTextFlags_None,
        )) {
            if (!config_locked) {
                ctx.scene.motor_controller.config.controller_port = @intCast(@max(0, @min(controller_port, 65535)));
            }
        }

        if (config_locked) {
            imgui.igPopStyleVar(1);
        }
    }

    inline fn drawProtocolConfig(self: *Self, ctx: *const UIContext) void {
        // DShot Protocol Selection

        inline for (@typeInfo(MotorController.Protocols).@"enum".fields) |protocol| {
            if (protocol.value > 0) {
                imgui.igSameLine(0, 4);
            }

            const is_selected = protocol.value == @intFromEnum(ctx.scene.motor_controller.config.dshot_protocol);
            Styles.pushButtonColors(is_selected);

            const protocol_str = std.fmt.allocPrintZ(self.allocator, "DShot {s}", .{@tagName(@as(MotorController.Protocols, @enumFromInt(protocol.value)))}) catch unreachable;
            defer self.allocator.free(protocol_str);

            if (imgui.igButton(
                protocol_str,
                imgui.ImVec2{ .x = 90, .y = 0 },
            )) {
                ctx.scene.motor_controller.config.dshot_protocol = @enumFromInt(protocol.value);
            }

            Styles.popButtonColors();
        }

        imgui.igNewLine();

        // Global Max Throttle
        var global_max = ctx.scene.motor_controller.config.global_max_throttle;
        if (imgui.igSliderFloat(
            "Global Max Throttle (%)",
            &global_max,
            0.0,
            100.0,
            "%.1f",
            imgui.ImGuiSliderFlags_None,
        )) {
            ctx.scene.motor_controller.config.global_max_throttle = global_max;
        }
    }

    inline fn drawMotorConfig(self: *Self, ctx: *const UIContext) void {
        // Motor Selection Buttons
        imgui.igText("Select Motor:");
        inline for (@typeInfo(MotorController.Motors).@"enum".fields) |motor| {
            if (motor.value > 0) {
                imgui.igSameLine(0, 4);
            }

            Styles.pushButtonColors(motor.value == @intFromEnum(self.selected_motor));

            const motor_string = motor.name;
            if (imgui.igButton(
                motor_string,
                imgui.ImVec2{ .x = 80, .y = 0 },
            )) {
                self.selected_motor = @enumFromInt(motor.value);
            }

            Styles.popButtonColors();
        }

        imgui.igNewLine();
        imgui.igSeparator();

        // Selected Motor Configuration
        const motor = &ctx.scene.motor_controller.config.motors[@intFromEnum(self.selected_motor)];
        var pin = @as(c_int, motor.pin);
        if (imgui.igInputInt("GPIO Pin", &pin, 0, 0, imgui.ImGuiInputTextFlags_CharsDecimal)) {
            motor.pin = @intCast(@max(0, @min(pin, 27)));
        }

        var direction: c_int = @intFromEnum(motor.direction);
        if (imgui.igCombo_Str(
            "Direction",
            &direction,
            "Clockwise\x00Counterclockwise\x00",
            2,
        )) {
            motor.direction = @enumFromInt(direction);
        }

        var max_throttle = motor.max_throttle;
        if (imgui.igSliderFloat(
            "Max Throttle (%)",
            &max_throttle,
            0.0,
            @min(100.0, ctx.scene.motor_controller.config.global_max_throttle),
            "%.1f",
            imgui.ImGuiSliderFlags_None,
        )) {
            motor.max_throttle = max_throttle;
        }
    }

    inline fn drawSaveModal(self: *Self, ctx: *const UIContext) void {
        if (self.show_save_modal) {
            imgui.igSetNextWindowSize(.{ .x = 400, .y = 150 }, imgui.ImGuiCond_FirstUseEver);
            if (imgui.igBegin("Save Configuration##modal", &self.show_save_modal, imgui.ImGuiWindowFlags_Modal)) {
                imgui.igText("Save configuration to:");
                _ = imgui.igInputText(
                    "##path",
                    &self.config_path_buffer,
                    self.config_path_buffer.len,
                    imgui.ImGuiInputTextFlags_None,
                    null,
                    null,
                );

                imgui.igText("File will be saved in the configs folder:");
                imgui.igText("%s", @as([*c]const u8, &self.cwd_buffer));

                imgui.igSpacing();
                if (imgui.igButton("Save", imgui.ImVec2{ .x = 120, .y = 0 })) {
                    ctx.scene.motor_controller.config.saveToFile(self.config_path_buffer[0 .. std.mem.indexOfScalar(u8, &self.config_path_buffer, 0) orelse self.config_path_buffer.len]) catch |err| {
                        std.debug.print("Failed to save config: {}\n", .{err});
                    };
                    self.show_save_modal = false;
                }
                imgui.igSameLine(0, 10);
                if (imgui.igButton("Cancel", imgui.ImVec2{ .x = 120, .y = 0 })) {
                    self.show_save_modal = false;
                }
            }
            imgui.igEnd();
        }
    }

    inline fn drawLoadModal(self: *Self, ctx: *const UIContext) void {
        if (self.show_load_modal) {
            imgui.igSetNextWindowSize(.{ .x = 400, .y = 300 }, imgui.ImGuiCond_FirstUseEver);
            if (imgui.igBegin("Load Configuration##modal", &self.show_load_modal, imgui.ImGuiWindowFlags_Modal)) {
                // Show current directory contents
                imgui.igText("Select configuration file:");
                imgui.igSeparator();
                imgui.igNewLine();

                if (std.fs.cwd().openDir(DroneConfig.default_config_folder, .{ .iterate = true })) |dir_handle| {
                    var dir = dir_handle;
                    var iter = dir.iterate();
                    while (iter.next() catch null) |entry| {
                        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".json")) {
                            if (imgui.igSelectable_Bool(
                                entry.name.ptr,
                                std.mem.eql(u8, entry.name, self.config_path_buffer[0 .. std.mem.indexOfScalar(u8, &self.config_path_buffer, 0) orelse 0]),
                                imgui.ImGuiSelectableFlags_None,
                                imgui.ImVec2{ .x = 0, .y = 0 },
                            )) {
                                @memcpy(self.config_path_buffer[0..@min(entry.name.len, self.config_path_buffer.len - 1)], entry.name);
                                self.config_path_buffer[entry.name.len] = 0;
                            }
                        }
                    }
                    dir.close();
                } else |_| {
                    imgui.igText("Could not read directory contents");
                }

                imgui.igSpacing();
                if (imgui.igButton("Load", imgui.ImVec2{ .x = 120, .y = 0 })) {
                    const path = self.config_path_buffer[0 .. std.mem.indexOfScalar(u8, &self.config_path_buffer, 0) orelse self.config_path_buffer.len];
                    ctx.scene.motor_controller.config = DroneConfig.loadFromFile(
                        ctx.scene.motor_controller.config.allocator,
                        path,
                    ) catch |err| {
                        std.debug.print("Failed to load config: {}\n", .{err});
                        return;
                    };
                    self.show_load_modal = false;
                }
                imgui.igSameLine(0, 10);
                if (imgui.igButton("Cancel", imgui.ImVec2{ .x = 120, .y = 0 })) {
                    self.show_load_modal = false;
                }
            }
            imgui.igEnd();
        }
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
