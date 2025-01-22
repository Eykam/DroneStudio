const std = @import("std");
const imgui = @import("bindings/c.zig").imgui;
const Scene = @import("Pipeline.zig").Scene;
const StereoMatcher = @import("ORB.zig").StereoMatcher;

const UIContext = struct {
    scene: *Scene,
    StereoMatcher: *StereoMatcher,
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

        if (imgui.igBegin("Stereo Matching Debug", &self.visible, window_flags)) {
            // Statistics Section
            imgui.igText("Statistics");
            imgui.igSeparator();
            imgui.igText("Current Matches: %d", ctx.StereoMatcher.num_matches.*);
            imgui.igText("Left Keypoints: %d", ctx.StereoMatcher.left.num_keypoints.*);
            imgui.igText("Right Keypoints: %d", ctx.StereoMatcher.right.num_keypoints.*);
            if (ctx.StereoMatcher.left.frame) |frame| {
                imgui.igText("Frame dimensions: %dx%d", frame.width, frame.height);
            }
            imgui.igNewLine();

            var params = ctx.StereoMatcher.params;
            var params_changed = false;

            // Camera Parameters Section
            imgui.igText("Camera Parameters");
            imgui.igSeparator();
            imgui.igText("Baseline (mm): %.2f", params.baseline_mm);
            imgui.igText("Focal Length (mm): %.2f", params.focal_length_mm);
            imgui.igNewLine();

            // Keypoint Detection Parameters Section
            imgui.igText("Keypoint Detection Parameters");
            imgui.igSeparator();

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
                &@as(u32, 3),
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

            // Matching Parameters Section
            imgui.igText("Matching Parameters");
            imgui.igSeparator();

            var disable_matching = params.disable_matching;
            var show_connections = params.show_connections;

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
            imgui.igSeparator();

            var max_disparity = params.max_disparity;
            if (imgui.igSliderFloat(
                "Max Disparity",
                &max_disparity,
                10.0,
                200.0,
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
                50.0,
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
                0.6,
                0.9,
                "%.2f",
                imgui.ImGuiSliderFlags_None,
            )) {
                params.lowes_ratio = lowe_ratio;
                params_changed = true;
            }

            var cost_threshold = params.cost_ratio;
            if (imgui.igSliderFloat(
                "Cost Threshold",
                &cost_threshold,
                0.3,
                0.8,
                "%.2f",
                imgui.ImGuiSliderFlags_None,
            )) {
                params.cost_ratio = cost_threshold;
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

            ctx.StereoMatcher.params_changed = params_changed;
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
