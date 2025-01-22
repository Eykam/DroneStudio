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

        if (imgui.igBegin("Stereo Matching Debug", &self.visible, imgui.ImGuiWindowFlags_None)) {
            // Display matching statistics
            imgui.igText("Current Matches: %d", ctx.StereoMatcher.num_matches.*);
            imgui.igText("Left Keypoints: %d", ctx.StereoMatcher.left.num_keypoints.*);
            imgui.igText("Right Keypoints: %d", ctx.StereoMatcher.right.num_keypoints.*);

            if (ctx.StereoMatcher.left.frame) |frame| {
                imgui.igText("Frame dimensions: %dx%d", frame.width, frame.height);
            }

            imgui.igSeparator();
            imgui.igText("Matching Parameters");

            var params = ctx.StereoMatcher.params;

            imgui.igText("Baseline (mm): %.2f", params.baseline_mm);
            imgui.igText("Focal Length (mm): %.2f", params.focal_length_mm);
            imgui.igNewLine();

            // Intensity threshold slider
            var intensity = params.intensity_threshold;
            if (imgui.igSliderScalar(
                "Intensity Threshold",
                imgui.ImGuiDataType_U8,
                &intensity,
                &@as(u8, 5),
                &@as(u8, 50),
                "%u",
                imgui.ImGuiSliderFlags_None,
            )) {
                params.intensity_threshold = intensity;
            }

            // Circle radius slider
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
            }

            // Arc length slider
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
            }

            // Max keypoints slider
            var max_keypoints = params.max_keypoints;
            if (imgui.igSliderScalar(
                "Max Keypoints",
                imgui.ImGuiDataType_U32,
                &max_keypoints,
                &@as(u32, 1000),
                &@as(u32, 100000),
                "%u",
                imgui.ImGuiSliderFlags_None,
            )) {
                params.max_keypoints = max_keypoints;
            }

            // Sigma slider
            var sigma = params.sigma;
            if (imgui.igSliderFloat("Sigma", &sigma, 0.1, 5.0, "%.2f", imgui.ImGuiSliderFlags_None)) {
                params.sigma = sigma;
            }

            // Max disparity slider
            var max_disparity = params.max_disparity;
            if (imgui.igSliderFloat("Max Disparity", &max_disparity, 10.0, 200.0, "%.1f", imgui.ImGuiSliderFlags_None)) {
                params.max_disparity = max_disparity;
            }

            // Epipolar threshold slider
            var epipolar = params.epipolar_threshold;
            if (imgui.igSliderFloat("Epipolar Threshold", &epipolar, 1.0, 50.0, "%.1f", imgui.ImGuiSliderFlags_None)) {
                params.epipolar_threshold = epipolar;
            }
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
