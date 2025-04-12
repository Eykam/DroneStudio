const std = @import("std");
const imgui = @import("bindings/c.zig").imgui;
const Scene = @import("Pipeline.zig").Scene;
const Vision = @import("Vision.zig");
const Drone = @import("Drone.zig");
const Sensors = @import("Sensors.zig");
const Math = @import("Math.zig");

const Vec3 = Math.Vec3;
const DroneConfig = Drone.DroneConfig;
const MotorController = Drone.MotorControllerClient;
// const StereoVO = Vision.StereoVO;
const CameraPose = Vision.CameraPose;

const UIContext = struct {
    scene: *Scene,
    // StereoVO: *StereoVO,
    pose_handler: *Sensors.PoseHandler,
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

pub const RootWindow = struct {
    const Self = @This();

    visible: bool,
    sidebar_width: f32 = 250.0, // Default width, will be adjusted by user
    min_sidebar_width: f32 = 150.0, // Minimum sidebar width
    max_sidebar_width: f32 = 500.0, // Maximum sidebar width
    is_resizing: bool = false, // Track if currently resizing

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = .{ .visible = true };
        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.destroy(self);
    }

    /// This is the top-level "root" that occupies the entire screen
    pub fn draw(self: *Self, ctx: *const UIContext) void {
        if (!self.visible) return;

        const viewport = imgui.igGetMainViewport();
        // Position at (0,0) in ImGui space, with the full size of the viewport:
        imgui.igSetNextWindowPos(viewport.*.WorkPos, imgui.ImGuiCond_Always, .{ .x = 0, .y = 0 });
        imgui.igSetNextWindowSize(viewport.*.WorkSize, imgui.ImGuiCond_Always);

        const window_flags =
            imgui.ImGuiWindowFlags_NoDecoration |
            imgui.ImGuiWindowFlags_NoMove |
            imgui.ImGuiWindowFlags_NoResize |
            imgui.ImGuiWindowFlags_NoCollapse |
            imgui.ImGuiWindowFlags_NoBringToFrontOnFocus |
            imgui.ImGuiWindowFlags_NoNavFocus |
            imgui.ImGuiWindowFlags_NoDocking;

        imgui.igPushStyleVar_Vec2(imgui.ImGuiStyleVar_WindowPadding, .{ .x = 0, .y = 0 });
        if (imgui.igBegin("RootWindow##FullScreen", &self.visible, window_flags)) {
            imgui.igPopStyleVar(1);

            // Get available window size
            var avail_size: imgui.ImVec2 = undefined;
            imgui.igGetContentRegionAvail(&avail_size);

            // ===== TOP BAR / NAVBAR =====
            const topbar_height = 0.0; // Increased height for better visual balance
            // const topbar_bg_color = imgui.igColorConvertFloat4ToU32(.{ .x = 0.1, .y = 0.1, .z = 0.1, .w = 1.0 });

            // // Store original cursor position for the background
            // var cursor_pos: imgui.ImVec2 = undefined;
            // imgui.igGetCursorScreenPos(&cursor_pos);

            // // Draw topbar background
            // const draw_list = imgui.igGetWindowDrawList();
            // imgui.ImDrawList_AddRectFilled(draw_list, cursor_pos, .{ .x = cursor_pos.x + avail_size.x, .y = cursor_pos.y + topbar_height }, topbar_bg_color, 0.0, imgui.ImDrawFlags_None);

            // // Create a child window for the topbar with custom styling
            // imgui.igPushStyleVar_Vec2(imgui.ImGuiStyleVar_WindowPadding, .{ .x = 15, .y = 0 });
            // imgui.igPushStyleVar_Vec2(imgui.ImGuiStyleVar_FramePadding, .{ .x = 8, .y = 6 }); // Larger padding for menu items
            // imgui.igPushStyleColor_Vec4(imgui.ImGuiCol_MenuBarBg, .{ .x = 0, .y = 0, .z = 0, .w = 0 }); // Transparent menubar
            // imgui.igPushStyleColor_Vec4(imgui.ImGuiCol_PopupBg, .{ .x = 0.15, .y = 0.15, .z = 0.15, .w = 0.98 }); // Dark popup background

            // _ = imgui.igBeginChild_Str(
            //     "Topbar",
            //     .{ .x = avail_size.x, .y = topbar_height },
            //     imgui.ImGuiChildFlags_None,
            //     imgui.ImGuiWindowFlags_NoScrollbar | imgui.ImGuiWindowFlags_MenuBar | imgui.ImGuiWindowFlags_NoNav,
            // );

            // // Use ImGui's menu bar for dropdown functionality
            // if (imgui.igBeginMenuBar()) {
            //     // Title with larger text and vertical centering
            //     const title_text = "Drone Studio";

            //     // Push larger text for the title (1.5x scale)
            //     imgui.igSetWindowFontScale(1.25);

            //     // Calculate height for vertical centering with scaled font
            //     const scaled_font_size = imgui.igGetFontSize();
            //     const menu_bar_height = imgui.igGetFrameHeight();
            //     const y_offset = (menu_bar_height - scaled_font_size) * 0.5 - 1; // small adjustment for visual center

            //     // Set cursor position for vertical centering
            //     imgui.igSetCursorPosY(imgui.igGetCursorPosY() + y_offset);

            //     // Draw the title
            //     imgui.igTextColored(.{ .x = 1.0, .y = 1.0, .z = 1.0, .w = 1.0 }, title_text);

            //     // Reset cursor Y position for menus
            //     imgui.igSetCursorPosY(imgui.igGetCursorPosY() - y_offset);

            //     // Add spacing between title and menus
            //     imgui.igSameLine(0, 40);

            //     // Style for menu items
            //     imgui.igPushStyleColor_Vec4(imgui.ImGuiCol_Text, .{ .x = 0.9, .y = 0.9, .z = 0.9, .w = 1.0 });

            //     // File menu dropdown
            //     if (imgui.igBeginMenu("File", true)) {
            //         if (imgui.igMenuItem_Bool("New Project", "Ctrl+N", false, true)) {}
            //         if (imgui.igMenuItem_Bool("Open Project", "Ctrl+O", false, true)) {}
            //         imgui.igSeparator();
            //         if (imgui.igMenuItem_Bool("Save", "Ctrl+S", false, true)) {}
            //         if (imgui.igMenuItem_Bool("Save As...", "Ctrl+Shift+S", false, true)) {}
            //         imgui.igSeparator();
            //         if (imgui.igMenuItem_Bool("Exit", "Alt+F4", false, true)) {}
            //         imgui.igEndMenu();
            //     }

            //     // View menu dropdown
            //     if (imgui.igBeginMenu("View", true)) {
            //         if (imgui.igMenuItem_Bool("Cameras", null, false, true)) {}
            //         if (imgui.igMenuItem_Bool("Configuration", null, false, true)) {}
            //         imgui.igSeparator();
            //         if (imgui.igMenuItem_Bool("Reset Layout", null, false, true)) {}
            //         imgui.igEndMenu();
            //     }

            //     // Help menu dropdown
            //     if (imgui.igBeginMenu("Help", true)) {
            //         if (imgui.igMenuItem_Bool("Documentation", null, false, true)) {}
            //         if (imgui.igMenuItem_Bool("About", null, false, true)) {}
            //         imgui.igEndMenu();
            //     }

            //     // Pop styling for menu items
            //     imgui.igPopStyleColor(1);

            //     imgui.igEndMenuBar();
            // }

            // imgui.igEndChild();

            // // Pop all pushed styles
            // imgui.igPopStyleColor(2); // Pop MenuBarBg and PopupBg colors
            // imgui.igPopStyleVar(2); // Pop WindowPadding and FramePadding

            // Adjust main area to account for topbar
            const main_area_height = avail_size.y - topbar_height;
            // Define sidebar width (15% of total width)
            const sidebar_width = avail_size.x * 0.15;

            // Setup columns: left sidebar (20%), right content area (80%)
            imgui.igColumns(2, "MainLayout", false);
            imgui.igSetColumnWidth(0, sidebar_width);

            // ===== LEFT COLUMN: SIDEBAR / MENUS =====
            // Put a visible border/background for the sidebar
            var cursor_pos: imgui.ImVec2 = undefined;
            const draw_list = imgui.igGetWindowDrawList();
            imgui.igGetCursorScreenPos(&cursor_pos);
            imgui.ImDrawList_AddRectFilled(
                draw_list,
                .{ .x = cursor_pos.x, .y = cursor_pos.y },
                .{ .x = cursor_pos.x + sidebar_width, .y = cursor_pos.y + main_area_height },
                imgui.igColorConvertFloat4ToU32(.{ .x = 0.15, .y = 0.15, .z = 0.15, .w = 1.0 }),
                0,
                imgui.ImDrawFlags_None,
            );
            // Add some padding inside the sidebar
            imgui.igPushStyleVar_Vec2(imgui.ImGuiStyleVar_WindowPadding, .{ .x = 10, .y = 10 });
            _ = imgui.igBeginChild_Str("Sidebar", .{ .x = sidebar_width - 10, .y = 0 }, imgui.ImGuiChildFlags_None, imgui.ImGuiWindowFlags_None);

            // Sidebar header
            imgui.igText("Navigation");
            imgui.igSeparator();

            // ViewportManager in sidebar
            ViewportManager(&self.visible, ctx);

            imgui.igSeparator();

            // Other menu items would go here
            imgui.igText("Other sub-menu items in the sidebar...");

            imgui.igEndChild();
            imgui.igPopStyleVar(1);

            // Move to second column (viewports area)
            imgui.igNextColumn();

            // ===== RIGHT COLUMN: MAIN VIEWPORT + SUB-VIEWPORTS =====

            // Calculate layout for viewports area
            const content_width = avail_size.x - sidebar_width;
            const main_viewport_height = main_area_height * 0.8; // Main viewport takes 70% height
            const sub_viewports_height = main_area_height - main_viewport_height;

            // 1) MAIN VIEWPORT SECTION
            imgui.igPushStyleVar_Vec2(imgui.ImGuiStyleVar_WindowPadding, .{ .x = 10, .y = 10 });
            _ = imgui.igBeginChild_Str("MainViewportArea", .{ .x = content_width, .y = main_viewport_height }, imgui.ImGuiChildFlags_None, imgui.ImGuiWindowFlags_None);

            const scene = ctx.scene;
            const camera_manager = scene.camera_manager;

            // Draw main viewport
            if (camera_manager.main_camera) |main_cam| {
                // Find the name of main_cam in your hash
                var main_cam_name: []const u8 = "unknown";
                var it_cams = camera_manager.cameras.iterator();
                while (it_cams.next()) |c_entry| {
                    if (c_entry.value_ptr.* == main_cam) {
                        main_cam_name = c_entry.key_ptr.*;
                        break;
                    }
                }

                // Look up the corresponding viewport
                const main_vp = camera_manager.viewports.get(main_cam_name);
                if (main_vp) |mvp| {
                    if (mvp.enabled) {
                        imgui.igTextColored(.{ .x = 0.0, .y = 0.8, .z = 1.0, .w = 1.0 }, "Main Camera: %s", main_cam_name.ptr);

                        // Using actual FBO dimensions for correct aspect ratio
                        const texture_width = @as(f32, @floatFromInt(mvp.fbo.width));
                        const texture_height = @as(f32, @floatFromInt(mvp.fbo.height));
                        const texture_aspect_ratio = texture_width / texture_height;

                        // Get available space
                        var region: imgui.ImVec2 = .{ .x = 0, .y = 0 };
                        imgui.igGetContentRegionAvail(&region);

                        // Calculate dimensions to maximize space usage while maintaining aspect ratio
                        var img_width = region.x;
                        var img_height = img_width / texture_aspect_ratio;

                        // If height would exceed available space, scale down
                        if (img_height > region.y - 30) {
                            img_height = region.y - 30; // Leave space for header text
                            img_width = img_height * texture_aspect_ratio;
                        }

                        // Center the image horizontally if it doesn't fill the width
                        if (img_width < region.x) {
                            imgui.igSetCursorPosX(imgui.igGetCursorPosX() + (region.x - img_width) * 0.5);
                        }

                        // Get the texture ID from the FBO
                        const main_tex: imgui.ImTextureID = @intCast(mvp.fbo.texture);

                        // Draw the image with the calculated dimensions
                        imgui.igImage(main_tex, .{ .x = img_width, .y = img_height }, .{ .x = 0, .y = 1 }, // UV coordinates for bottom-left
                            .{ .x = 1, .y = 0 }, // UV coordinates for top-right
                            .{ .x = 1, .y = 1, .z = 1, .w = 1 }, // Tint color (white)
                            .{ .x = 0.3, .y = 0.3, .z = 0.3, .w = 1.0 } // Border color (dark gray)
                        );
                    }
                }
            } else {
                imgui.igTextColored(.{ .x = 1, .y = 0, .z = 0, .w = 1 }, "No main camera set. Please select a camera");
            }

            imgui.igEndChild();
            imgui.igPopStyleVar(1);

            // 2) SUB-VIEWPORTS SECTION - Improved
            imgui.igPushStyleVar_Vec2(imgui.ImGuiStyleVar_WindowPadding, .{ .x = 10, .y = 10 });
            _ = imgui.igBeginChild_Str(
                "SubViewportsArea",
                .{ .x = content_width, .y = sub_viewports_height },
                imgui.ImGuiChildFlags_None,
                imgui.ImGuiWindowFlags_None,
            );

            imgui.igText("Other Active Cameras:");
            imgui.igSeparator();

            // Calculate layout for sub-viewports with more appropriate spacing
            const sub_panel_width = content_width;
            const min_viewport_width: f32 = 250.0;
            const viewport_spacing: f32 = 15.0;

            // Calculate how many viewports can fit in a row
            const max_per_row = @as(usize, @intFromFloat(@floor(sub_panel_width / (min_viewport_width + viewport_spacing))));
            if (max_per_row > 0) {
                var current_in_row: usize = 0;

                var it_vp = camera_manager.viewports.iterator();
                while (it_vp.next()) |entry| {
                    const vp_name = entry.key_ptr.*;
                    const vp = entry.value_ptr;

                    if (!vp.enabled) continue;

                    // Skip if this is the main camera
                    if (camera_manager.main_camera) |mc| {
                        if (camera_manager.cameras.get(vp_name) == mc) {
                            continue; // skip main
                        }
                    }

                    // Start a new row if needed
                    if (current_in_row >= max_per_row) {
                        current_in_row = 0;
                        imgui.igNewLine();
                    }

                    // Add spacing between viewports (except for first in row)
                    if (current_in_row > 0) {
                        imgui.igSameLine(0, viewport_spacing);
                    }

                    // Calculate viewport dimensions using actual FBO dimensions for correct aspect ratio
                    const tex_width = @as(f32, @floatFromInt(vp.fbo.width));
                    const tex_height = @as(f32, @floatFromInt(vp.fbo.height));
                    const actual_aspect_ratio = tex_width / tex_height;

                    const draw_width = min_viewport_width;
                    const draw_height = draw_width / actual_aspect_ratio;

                    // Begin a group for this viewport
                    imgui.igBeginGroup();

                    // Camera name with distinctive background
                    var cursor_pos_viewport: imgui.ImVec2 = undefined;
                    imgui.igGetCursorScreenPos(&cursor_pos_viewport);

                    var text_size: imgui.ImVec2 = undefined;
                    imgui.igCalcTextSize(&text_size, vp_name.ptr, null, false, 0);

                    // Draw text
                    imgui.igText("%s", vp_name.ptr);

                    // Store position before image for click detection
                    var image_pos: imgui.ImVec2 = undefined;
                    imgui.igGetCursorScreenPos(&image_pos);

                    // Get the texture ID from the FBO
                    const tex_id: imgui.ImTextureID = @intCast(vp.fbo.texture);

                    // Draw the viewport image
                    imgui.igImage(tex_id, .{ .x = draw_width, .y = draw_height }, .{ .x = 0, .y = 1 }, // UV coordinates for bottom-left
                        .{ .x = 1, .y = 0 }, // UV coordinates for top-right
                        .{ .x = 1, .y = 1, .z = 1, .w = 1 }, // Tint color (white)
                        .{ .x = 0.2, .y = 0.2, .z = 0.2, .w = 1.0 } // Border color (dark gray)
                    );

                    // Make the viewport image clickable to set as main
                    if (imgui.igIsItemClicked(imgui.ImGuiMouseButton_Left)) {
                        // Set this camera as main
                        camera_manager.main_camera = camera_manager.cameras.get(vp_name);

                        // Mark viewports for resolution update
                        scene.update_viewports = true;
                    }

                    // Add a tooltip when hovering over the image
                    if (imgui.igIsItemHovered(imgui.ImGuiHoveredFlags_None)) {
                        _ = imgui.igBeginTooltip();
                        imgui.igText("Click to set as main viewport");
                        imgui.igEndTooltip();
                    }

                    imgui.igEndGroup();
                    current_in_row += 1;
                }
            }

            imgui.igEndChild();
            imgui.igPopStyleVar(1);
        }
        imgui.igEnd();
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

// The ViewportManager function as a widget (not a window)
pub fn ViewportManager(visible: *bool, ctx: *const UIContext) void {
    _ = visible;

    const scene = ctx.scene;
    const camera_manager = scene.camera_manager;

    // Let user pick which cameras are active
    imgui.igText("Active Cameras:");
    imgui.igSeparator();

    var changed = false;
    var it_cam = camera_manager.cameras.iterator();
    while (it_cam.next()) |entry| {
        const cam_name = entry.key_ptr.*;

        // Ensure we have a bool in active_cameras
        if (!camera_manager.active_cameras.contains(cam_name)) {
            camera_manager.active_cameras.put(cam_name, false) catch {};
        }
        const active_ptr = camera_manager.active_cameras.getPtr(cam_name) orelse continue;

        // Checkbox for each camera
        if (imgui.igCheckbox(cam_name.ptr, active_ptr)) {
            changed = true;
        }
    }

    if (changed) {
        std.debug.print("Camera toggles changed.\n", .{});
        scene.update_viewports = true;
    }
}

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

    selected_motor: Drone.Protocol.Motors,
    global_throttle: f32,

    target_roll: f32 = 0.0,
    target_pitch: f32 = 0.0,
    target_yaw: f32 = 0.0,
    base_throttle: f32 = 20.0,
    roll_kp: f32 = 0.5,
    roll_ki: f32 = 0.01,
    roll_kd: f32 = 0.1,
    pitch_kp: f32 = 0.5,
    pitch_ki: f32 = 0.01,
    pitch_kd: f32 = 0.1,
    yaw_kp: f32 = 0.2,
    yaw_ki: f32 = 0.0,
    yaw_kd: f32 = 0.05,
    history_roll_error: [100]f32 = [_]f32{0.0} ** 100,
    history_pitch_error: [100]f32 = [_]f32{0.0} ** 100,
    history_yaw_error: [100]f32 = [_]f32{0.0} ** 100,
    history_index: usize = 0,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .visible = true,
            .selected_motor = .Motor_1,
            .global_throttle = 0,
            .allocator = allocator,
        };

        @memset(&self.local_ip_buffer, 0);
        @memset(&self.controller_ip_buffer, 0);
        @memset(&self.config_path_buffer, 0);
        @memset(&self.cwd_buffer, 0);

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

            if (imgui.igBeginTabBar("DroneConfigTabs", imgui.ImGuiTabBarFlags_None)) {
                if (imgui.igBeginTabItem("Motor Controller", null, 0)) {
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

                    imgui.igEndTabItem();
                }

                if (imgui.igBeginTabItem("PID Control", null, 0)) {
                    self.drawPidTab(ctx);
                    imgui.igEndTabItem();
                }

                if (imgui.igBeginTabItem("Sensors", null, 0)) {
                    self.drawSensorConfig(ctx);
                    imgui.igEndTabItem();
                }
            }

            imgui.igEndTabBar();

            // Modal Windows
            self.drawSaveModal(ctx);
            self.drawLoadModal(ctx);
        }
        imgui.igEnd();
    }

    pub fn drawPidTab(self: *Self, ctx: *const UIContext) void {
        const motor_controller = ctx.scene.motor_controller.?;
        const pid_debug = motor_controller.getPidDebugInfo();
        const deg_to_rad = std.math.pi / 180.0;

        // Update PID errors history
        if (motor_controller.connection_handler.pid_debug.active) {
            self.history_roll_error[self.history_index] = pid_debug.roll_error;
            self.history_pitch_error[self.history_index] = pid_debug.pitch_error;
            self.history_yaw_error[self.history_index] = pid_debug.yaw_error;
            self.history_index = (self.history_index + 1) % self.history_roll_error.len;
        }

        // PID Controller Status
        if (imgui.igCollapsingHeader_BoolPtr("PID Controller Status", null, imgui.ImGuiTreeNodeFlags_DefaultOpen)) {
            // Active status
            const is_active = pid_debug.active;
            const status_color = if (is_active)
                imgui.ImVec4{ .x = 0.0, .y = 1.0, .z = 0.0, .w = 1.0 }
            else
                imgui.ImVec4{ .x = 0.7, .y = 0.7, .z = 0.7, .w = 1.0 };

            const active_text = if (is_active) "ACTIVE" else "INACTIVE";
            imgui.igTextColored(status_color, "Status: %s", &active_text);

            // Enable/Disable button
            if (is_active) {
                if (imgui.igButton("Disable Orientation Control", imgui.ImVec2{ .x = 200, .y = 0 })) {
                    const command = Drone.Protocol.Command{
                        .type = .StopOrientation,
                    };
                    _ = motor_controller.command_queue.push(command);
                }
            } else {
                if (imgui.igButton("Enable Orientation Control", imgui.ImVec2{ .x = 200, .y = 0 })) {
                    // Create quaternion from euler angles
                    const quat = Math.Quaternion.from_euler(self.target_pitch * deg_to_rad, self.target_yaw * deg_to_rad, self.target_roll * deg_to_rad);

                    const command = Drone.Protocol.Command{
                        .type = .SetOrientation,
                        .pose = quat,
                    };
                    _ = motor_controller.command_queue.push(command);
                }
            }

            imgui.igSeparator();

            // Current and target orientation display
            imgui.igText("Current Orientation (degrees):");
            imgui.igText("  Roll:  %.2f", pid_debug.current_roll);
            imgui.igText("  Pitch: %.2f", pid_debug.current_pitch);
            imgui.igText("  Yaw:   %.2f", pid_debug.current_yaw);

            imgui.igText("Target Orientation (degrees):");
            imgui.igText("  Roll:  %.2f", pid_debug.target_roll);
            imgui.igText("  Pitch: %.2f", pid_debug.target_pitch);
            imgui.igText("  Yaw:   %.2f", pid_debug.target_yaw);

            imgui.igSeparator();

            // PID Error Values
            imgui.igText("PID Errors:");
            imgui.igText("  Roll:  %.2f", pid_debug.roll_error);
            imgui.igText("  Pitch: %.2f", pid_debug.pitch_error);
            imgui.igText("  Yaw:   %.2f", pid_debug.yaw_error);

            // PID Integral Values
            imgui.igText("PID Integral Values:");
            imgui.igText("  Roll:  %.2f", pid_debug.roll_integral);
            imgui.igText("  Pitch: %.2f", pid_debug.pitch_integral);
            imgui.igText("  Yaw:   %.2f", pid_debug.yaw_integral);

            imgui.igSeparator();

            // Motor outputs
            imgui.igText("Motor Outputs:");
            for (pid_debug.motor_outputs, 0..) |output, i| {
                imgui.igText("  Motor %d: %.1f%%", i, output);
            }
        }

        // Set Target Orientation
        if (imgui.igCollapsingHeader_BoolPtr("Set Target Orientation", null, imgui.ImGuiTreeNodeFlags_DefaultOpen)) {
            imgui.igText("Enter target orientation in degrees:");

            var changed = false;
            changed = imgui.igSliderFloat("Target Roll", &self.target_roll, -45.0, 45.0, "%.1f°", imgui.ImGuiSliderFlags_None) or changed;
            changed = imgui.igSliderFloat("Target Pitch", &self.target_pitch, -45.0, 45.0, "%.1f°", imgui.ImGuiSliderFlags_None) or changed;
            changed = imgui.igSliderFloat("Target Yaw", &self.target_yaw, -180.0, 180.0, "%.1f°", imgui.ImGuiSliderFlags_None) or changed;

            imgui.igSeparator();

            // Apply button
            if (imgui.igButton("Apply Target Orientation", imgui.ImVec2{ .x = 200, .y = 0 }) or changed) {
                // Create quaternion from euler angles
                const quat = Math.Quaternion.from_euler(self.target_pitch * deg_to_rad, self.target_yaw * deg_to_rad, self.target_roll * deg_to_rad);

                const command = Drone.Protocol.Command{
                    .type = .SetOrientation,
                    .pose = quat,
                };
                _ = motor_controller.command_queue.push(command);
            }

            // Reset button
            imgui.igSameLine(0, 10);
            if (imgui.igButton("Reset to Current", imgui.ImVec2{ .x = 120, .y = 0 })) {
                self.target_roll = pid_debug.current_roll;
                self.target_pitch = pid_debug.current_pitch;
                self.target_yaw = pid_debug.current_yaw;

                // Also send the command
                const quat = Math.Quaternion.from_euler(self.target_pitch * deg_to_rad, self.target_yaw * deg_to_rad, self.target_roll * deg_to_rad);

                const command = Drone.Protocol.Command{
                    .type = .SetOrientation,
                    .pose = quat,
                };
                _ = motor_controller.command_queue.push(command);
            }

            // Add "Level" button for quick leveling
            imgui.igSameLine(0, 10);
            if (imgui.igButton("Level", imgui.ImVec2{ .x = 60, .y = 0 })) {
                self.target_roll = 0.0;
                self.target_pitch = 0.0;
                // Keep current yaw
                self.target_yaw = pid_debug.current_yaw;

                const quat = Math.Quaternion.from_euler(0.0, self.target_yaw * deg_to_rad, 0.0);

                const command = Drone.Protocol.Command{
                    .type = .SetOrientation,
                    .pose = quat,
                };
                _ = motor_controller.command_queue.push(command);
            }

            imgui.igSeparator();

            // Base throttle setting
            changed = imgui.igSliderFloat("Base Throttle", &self.base_throttle, 5.0, 50.0, "%.1f%%", imgui.ImGuiSliderFlags_None);
            if (changed and pid_debug.active) {
                // Send base throttle update command
                const command = Drone.Protocol.Command{
                    .type = .UpdateBaseThrottle,
                    .speed = self.base_throttle,
                };
                _ = motor_controller.command_queue.push(command);
            }
        }

        // PID Parameters
        if (imgui.igCollapsingHeader_BoolPtr("PID Parameters", null, imgui.ImGuiTreeNodeFlags_DefaultOpen)) {
            // Update local variables from PID debug info if active
            if (pid_debug.active) {
                self.roll_kp = pid_debug.roll_kp;
                self.roll_ki = pid_debug.roll_ki;
                self.roll_kd = pid_debug.roll_kd;

                self.pitch_kp = pid_debug.pitch_kp;
                self.pitch_ki = pid_debug.pitch_ki;
                self.pitch_kd = pid_debug.pitch_kd;

                self.yaw_kp = pid_debug.yaw_kp;
                self.yaw_ki = pid_debug.yaw_ki;
                self.yaw_kd = pid_debug.yaw_kd;
            }

            // Show axes with different colors
            imgui.igTextColored(.{ .x = 1.0, .y = 0.4, .z = 0.4, .w = 1.0 }, "Roll Axis (Red)");
            var roll_changed = imgui.igSliderFloat("Roll P##roll", &self.roll_kp, 0.0, 2.0, "%.3f", imgui.ImGuiSliderFlags_None);
            roll_changed = imgui.igSliderFloat("Roll I##roll", &self.roll_ki, 0.0, 0.1, "%.3f", imgui.ImGuiSliderFlags_None) or roll_changed;
            roll_changed = imgui.igSliderFloat("Roll D##roll", &self.roll_kd, 0.0, 1.0, "%.3f", imgui.ImGuiSliderFlags_None) or roll_changed;

            if (roll_changed) {
                // Send command to update roll PID params
                const command = Drone.Protocol.Command{
                    .type = .UpdatePidParams,
                    .axis = "Roll",
                    .kp = self.roll_kp,
                    .ki = self.roll_ki,
                    .kd = self.roll_kd,
                };
                _ = motor_controller.command_queue.push(command);
            }

            imgui.igSpacing();
            imgui.igTextColored(.{ .x = 0.4, .y = 1.0, .z = 0.4, .w = 1.0 }, "Pitch Axis (Green)");
            var pitch_changed = imgui.igSliderFloat("Pitch P##pitch", &self.pitch_kp, 0.0, 2.0, "%.3f", imgui.ImGuiSliderFlags_None);
            pitch_changed = imgui.igSliderFloat("Pitch I##pitch", &self.pitch_ki, 0.0, 0.1, "%.3f", imgui.ImGuiSliderFlags_None) or pitch_changed;
            pitch_changed = imgui.igSliderFloat("Pitch D##pitch", &self.pitch_kd, 0.0, 1.0, "%.3f", imgui.ImGuiSliderFlags_None) or pitch_changed;

            if (pitch_changed) {
                // Send command to update pitch PID params
                const command = Drone.Protocol.Command{
                    .type = .UpdatePidParams,
                    .axis = "Pitch",
                    .kp = self.pitch_kp,
                    .ki = self.pitch_ki,
                    .kd = self.pitch_kd,
                };
                _ = motor_controller.command_queue.push(command);
            }

            imgui.igSpacing();
            imgui.igTextColored(.{ .x = 0.4, .y = 0.4, .z = 1.0, .w = 1.0 }, "Yaw Axis (Blue)");
            var yaw_changed = imgui.igSliderFloat("Yaw P##yaw", &self.yaw_kp, 0.0, 2.0, "%.3f", imgui.ImGuiSliderFlags_None);
            yaw_changed = imgui.igSliderFloat("Yaw I##yaw", &self.yaw_ki, 0.0, 0.1, "%.3f", imgui.ImGuiSliderFlags_None) or yaw_changed;
            yaw_changed = imgui.igSliderFloat("Yaw D##yaw", &self.yaw_kd, 0.0, 1.0, "%.3f", imgui.ImGuiSliderFlags_None) or yaw_changed;

            if (yaw_changed) {
                // Send command to update yaw PID params
                const command = Drone.Protocol.Command{
                    .type = .UpdatePidParams,
                    .axis = "Yaw",
                    .kp = self.yaw_kp,
                    .ki = self.yaw_ki,
                    .kd = self.yaw_kd,
                };
                _ = motor_controller.command_queue.push(command);
            }

            imgui.igSeparator();
            if (imgui.igButton("Apply All Parameters", imgui.ImVec2{ .x = 200, .y = 0 })) {
                // Send command to update all PID params at once
                const command = Drone.Protocol.Command{
                    .type = .UpdatePidParams,
                    .axis = "All",
                    .kp = self.roll_kp, // Using roll values for all axes
                    .ki = self.roll_ki,
                    .kd = self.roll_kd,
                };
                _ = motor_controller.command_queue.push(command);
            }
        }

        // PID Error Plot
        if (imgui.igCollapsingHeader_BoolPtr("PID Error Plot", null, imgui.ImGuiTreeNodeFlags_DefaultOpen)) {
            // Error plots
            var plot_params: imgui.ImVec2 = undefined;
            imgui.igGetContentRegionAvail(&plot_params);

            const plot_height = 100.0;
            const plot_width = plot_params.x;

            const draw_list = imgui.igGetWindowDrawList();
            var cursor_pos: imgui.ImVec2 = undefined;
            imgui.igGetCursorScreenPos(&cursor_pos);

            // Background
            imgui.ImDrawList_AddRectFilled(
                draw_list,
                cursor_pos,
                .{ .x = cursor_pos.x + plot_width, .y = cursor_pos.y + plot_height },
                imgui.igColorConvertFloat4ToU32(.{ .x = 0.1, .y = 0.1, .z = 0.1, .w = 0.7 }),
                5.0,
                imgui.ImDrawFlags_None,
            );

            // Grid lines
            const grid_color = imgui.igColorConvertFloat4ToU32(.{ .x = 0.5, .y = 0.5, .z = 0.5, .w = 0.5 });
            const center_y = cursor_pos.y + plot_height / 2.0;

            // Horizontal center line
            imgui.ImDrawList_AddLine(draw_list, .{ .x = cursor_pos.x, .y = center_y }, .{ .x = cursor_pos.x + plot_width, .y = center_y }, grid_color, 1.0);

            // Get max error for scaling
            var max_error: f32 = 5.0; // Default scale to +/- 5 degrees
            for (0..self.history_roll_error.len) |i| {
                max_error = @max(max_error, @abs(self.history_roll_error[i]));
                max_error = @max(max_error, @abs(self.history_pitch_error[i]));
                max_error = @max(max_error, @abs(self.history_yaw_error[i]));
            }

            // Scale factor
            const scale_factor = (plot_height / 2.0) / max_error;

            // Error limit markers
            imgui.ImDrawList_AddText_Vec2(
                draw_list,
                .{ .x = cursor_pos.x + 5, .y = cursor_pos.y + 5 },
                imgui.igColorConvertFloat4ToU32(.{ .x = 1.0, .y = 1.0, .z = 1.0, .w = 1.0 }),
                std.fmt.allocPrintZ(self.allocator, "+{d:.1}°", .{max_error}) catch "",
                null,
            );

            imgui.ImDrawList_AddText_Vec2(
                draw_list,
                .{ .x = cursor_pos.x + 5, .y = cursor_pos.y + plot_height - 15 },
                imgui.igColorConvertFloat4ToU32(.{ .x = 1.0, .y = 1.0, .z = 1.0, .w = 1.0 }),
                std.fmt.allocPrintZ(self.allocator, "-{d:.1}°", .{max_error}) catch "",
                null,
            );

            // Draw roll error line
            const roll_color = imgui.igColorConvertFloat4ToU32(.{ .x = 1.0, .y = 0.4, .z = 0.4, .w = 1.0 });
            var prev_point_x = cursor_pos.x;
            var prev_point_y = center_y - self.history_roll_error[self.history_index] * scale_factor;

            for (0..self.history_roll_error.len) |i| {
                const idx = (self.history_index + i) % self.history_roll_error.len;
                const x = cursor_pos.x + (@as(f32, @floatFromInt(i)) * plot_width) / self.history_roll_error.len;
                const y = center_y - self.history_roll_error[idx] * scale_factor;

                if (i > 0) {
                    imgui.ImDrawList_AddLine(draw_list, .{ .x = prev_point_x, .y = prev_point_y }, .{ .x = x, .y = y }, roll_color, 2.0);
                }

                prev_point_x = x;
                prev_point_y = y;
            }

            // Draw pitch error line
            const pitch_color = imgui.igColorConvertFloat4ToU32(.{ .x = 0.4, .y = 1.0, .z = 0.4, .w = 1.0 });
            prev_point_x = cursor_pos.x;
            prev_point_y = center_y - self.history_pitch_error[self.history_index] * scale_factor;

            for (0..self.history_pitch_error.len) |i| {
                const idx = (self.history_index + i) % self.history_pitch_error.len;
                const x = cursor_pos.x + (@as(f32, @floatFromInt(i)) * plot_width) / self.history_pitch_error.len;
                const y = center_y - self.history_pitch_error[idx] * scale_factor;

                if (i > 0) {
                    imgui.ImDrawList_AddLine(draw_list, .{ .x = prev_point_x, .y = prev_point_y }, .{ .x = x, .y = y }, pitch_color, 2.0);
                }

                prev_point_x = x;
                prev_point_y = y;
            }

            // Draw yaw error line
            const yaw_color = imgui.igColorConvertFloat4ToU32(.{ .x = 0.4, .y = 0.4, .z = 1.0, .w = 1.0 });
            prev_point_x = cursor_pos.x;
            prev_point_y = center_y - self.history_yaw_error[self.history_index] * scale_factor;

            for (0..self.history_yaw_error.len) |i| {
                const idx = (self.history_index + i) % self.history_yaw_error.len;
                const x = cursor_pos.x + (@as(f32, @floatFromInt(i)) * plot_width) / self.history_yaw_error.len;
                const y = center_y - self.history_yaw_error[idx] * scale_factor;

                if (i > 0) {
                    imgui.ImDrawList_AddLine(draw_list, .{ .x = prev_point_x, .y = prev_point_y }, .{ .x = x, .y = y }, yaw_color, 2.0);
                }

                prev_point_x = x;
                prev_point_y = y;
            }

            // Legend
            imgui.igDummy(.{ .x = 0, .y = plot_height + 5 });
            imgui.igTextColored(.{ .x = 1.0, .y = 0.4, .z = 0.4, .w = 1.0 }, "Roll");
            imgui.igSameLine(0, 20);
            imgui.igTextColored(.{ .x = 0.4, .y = 1.0, .z = 0.4, .w = 1.0 }, "Pitch");
            imgui.igSameLine(0, 20);
            imgui.igTextColored(.{ .x = 0.4, .y = 0.4, .z = 1.0, .w = 1.0 }, "Yaw");
        }
    }

    pub fn drawSensorConfig(self: *Self, ctx: *const UIContext) void {
        _ = self;

        const sensor_state = ctx.pose_handler.sensor_state;
        const pose = ctx.pose_handler.prev_pose orelse Sensors.Pose{
            .accel = Vec3.zero(),
            .gyro = Vec3.zero(),
            .mag = Vec3.zero(),
            .timestamp = 0,
        };

        const curr_beta = if (sensor_state.filter) |filter| filter.beta else 0.0;
        imgui.igText("Current Beta: %.3f", curr_beta);
        imgui.igNewLine();
        if (imgui.igButton("Reset Orientation", .{ .x = 0, .y = 0 })) {
            sensor_state.filter.?.q = Sensors.computeInitialOrientation(pose.accel, pose.mag, 0);
        }
        imgui.igNewLine();

        // Magnetometer Section
        if (imgui.igCollapsingHeader_BoolPtr("Magnetometer Calibration", null, imgui.ImGuiTreeNodeFlags_DefaultOpen)) {
            // Show last raw reading (approx)
            const mag = pose.mag;
            imgui.igText("Raw Magnetometer (NED or pre-rotated):");
            imgui.igText("  X: %.2f", mag.x());
            imgui.igText("  Y: %.2f", mag.y());
            imgui.igText("  Z: %.2f", mag.z());

            // Show current calibration values
            imgui.igText("Hard Iron: [%.2f, %.2f, %.2f]", sensor_state.mag_hard_iron.x(), sensor_state.mag_hard_iron.y(), sensor_state.mag_hard_iron.z());
            imgui.igText("Soft Iron: [%.2f, %.2f, %.2f]", sensor_state.mag_soft_iron.x(), sensor_state.mag_soft_iron.y(), sensor_state.mag_soft_iron.z());

            // If we are in magnetometer calibration mode, show countdown
            if (sensor_state.calibration_type == .Magnetometer and sensor_state.calibrating) {
                const samples = ctx.pose_handler.sensor_state.samples;
                const sample_count = ctx.pose_handler.sensor_state.sample_count;
                const percentage = @as(f32, @floatFromInt(sample_count)) / @as(f32, @floatFromInt(samples));

                const freq = ctx.pose_handler.mag_freq;
                const remaining_secs = @as(f32, @floatFromInt(samples - sample_count)) / @as(f32, @floatFromInt(freq));

                imgui.igTextColored(.{ .x = 1.0, .y = 0.65, .z = 0.0, .w = 1.0 }, "Calibrating... Approx. %.1f seconds left", remaining_secs);
                imgui.igProgressBar(percentage, .{ .x = -1.0, .y = 0.0 }, null);
            }

            // "Calibrate" button
            const calibrating = sensor_state.calibrating;
            if (calibrating) {
                imgui.igBeginDisabled(true);
            }

            if (imgui.igButton("Calibrate Magnetometer", .{ .x = 200, .y = 0 })) {
                if (!sensor_state.calibrating) {
                    sensor_state.start_calibration(.Magnetometer);
                }
            }

            if (calibrating) {
                imgui.igEndDisabled();
            }

            imgui.igNewLine();
        }

        // Accelerometer & Gyro Calibration
        if (imgui.igCollapsingHeader_BoolPtr("Accelerometer & Gyro Calibration", null, imgui.ImGuiTreeNodeFlags_DefaultOpen)) {
            // Raw readings
            imgui.igText("Raw Accel:");
            imgui.igText("  X: %.2f", pose.accel.x());
            imgui.igText("  Y: %.2f", pose.accel.y());
            imgui.igText("  Z: %.2f", pose.accel.z());

            imgui.igText("Raw Gyro:");
            imgui.igText("  X: %.2f", pose.gyro.x());
            imgui.igText("  Y: %.2f", pose.gyro.y());
            imgui.igText("  Z: %.2f", pose.gyro.z());

            // Show offsets
            imgui.igText(
                "Accel Offset: [%.2f, %.2f, %.2f]",
                sensor_state.accel_offset.x(),
                sensor_state.accel_offset.y(),
                sensor_state.accel_offset.z(),
            );
            imgui.igText(
                "Gyro Offset: [%.2f, %.2f, %.2f]",
                sensor_state.gyro_offset.x(),
                sensor_state.gyro_offset.y(),
                sensor_state.gyro_offset.z(),
            );

            // +/- 90° rotation test buttons (stubs)
            imgui.igText("Rotation Test:");
            if (imgui.igButton("X +90°", .{ .x = 60, .y = 0 })) {
                // TODO: implement manual rotate
            }
            imgui.igSameLine(0, 4);
            if (imgui.igButton("X -90°", .{ .x = 60, .y = 0 })) {
                // ...
            }
            imgui.igSameLine(0, 10);
            if (imgui.igButton("Y +90°", .{ .x = 60, .y = 0 })) {
                // ...
            }
            imgui.igSameLine(0, 4);
            if (imgui.igButton("Y -90°", .{ .x = 60, .y = 0 })) {
                // ...
            }
            imgui.igSameLine(0, 10);
            if (imgui.igButton("Z +90°", .{ .x = 60, .y = 0 })) {
                // ...
            }
            imgui.igSameLine(0, 4);
            if (imgui.igButton("Z -90°", .{ .x = 60, .y = 0 })) {
                // ...
            }

            // If currently calibrating Accel/Gyro, show countdown
            if (sensor_state.calibration_type == .AccelGyro and sensor_state.calibrating) {
                const samples = ctx.pose_handler.sensor_state.samples;
                const sample_count = ctx.pose_handler.sensor_state.sample_count;
                const percentage = @as(f32, @floatFromInt(sample_count)) / @as(f32, @floatFromInt(samples));

                const accel_gyro_freq = ctx.pose_handler.accel_gyro_freq;
                const remaining_secs = @as(f32, @floatFromInt(samples - sample_count)) / @as(f32, @floatFromInt(accel_gyro_freq));

                imgui.igTextColored(.{ .x = 1.0, .y = 0.65, .z = 0.0, .w = 1.0 }, "Calibrating... Approx. %.1f seconds left", remaining_secs);
                imgui.igProgressBar(percentage, .{ .x = -1.0, .y = 0.0 }, null);
            }

            const calibrating = sensor_state.calibrating;
            if (calibrating) {
                imgui.igBeginDisabled(true);
            }
            // Calibrate button
            if (imgui.igButton("Calibrate Accel & Gyro", .{ .x = 200, .y = 0 })) {
                if (!sensor_state.calibrating) {
                    sensor_state.start_calibration(.AccelGyro);
                }
            }

            if (calibrating) {
                imgui.igEndDisabled();
            }
        }
    }

    inline fn drawFixedHeader(self: *Self, ctx: *const UIContext) void {
        imgui.igSeparator();

        imgui.igText("Motor Controller ");
        imgui.igNewLine();

        // Connection Status and Controls
        const motor_controller = ctx.scene.motor_controller.?;
        const conn_state = motor_controller.getConnectionState();
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

        // Connection control buttons
        switch (conn_state) {
            .Disconnected => {
                if (imgui.igButton("Connect", imgui.ImVec2{ .x = 120, .y = 0 })) {
                    motor_controller.connect() catch |err| {
                        std.debug.print("Failed to connect: {any}\n", .{err});
                    };
                }
            },
            .Failed => {
                if (imgui.igButton("Retry", imgui.ImVec2{ .x = 120, .y = 0 })) {
                    motor_controller.retryConnection() catch |err| {
                        std.debug.print("Failed to retry connection: {any}\n", .{err});
                    };
                }
            },
            .Connected, .ConfigSync, .Ready, .Running => {
                if (imgui.igButton("Disconnect", imgui.ImVec2{ .x = 120, .y = 0 })) {
                    motor_controller.disconnect();
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
        const config = motor_controller.config;

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
        imgui.igNewLine();
        imgui.igSeparator();

        imgui.igText("IMU Status");
        const sensor_state = ctx.pose_handler.sensor_state;

        const is_receiving = (ctx.pose_handler.accel_gyro_freq > 0);
        if (is_receiving) {
            imgui.igTextColored(.{ .x = 0.0, .y = 1.0, .z = 0.0, .w = 1.0 }, "Status: Receiving Data");
        } else {
            imgui.igTextColored(.{ .x = 1.0, .y = 0.0, .z = 0.0, .w = 1.0 }, "Status: No Data");
        }

        imgui.igText("Accel & Gyro Throughput: %d packets/s", ctx.pose_handler.accel_gyro_freq);
        imgui.igText("Magnetometer Throughput: %d packets/s", ctx.pose_handler.mag_freq);
        imgui.igText("Stale Packets: %d", ctx.pose_handler.stale_count);
        imgui.igNewLine();

        // Current Orientation
        if (sensor_state.filter) |filter| {
            const q = filter.q;
            const euler = q.to_euler();
            const rad_to_deg = 180.0 / std.math.pi;

            imgui.igText("Orientation (deg):");
            imgui.igText("  Roll:  %.2f", euler[0] * rad_to_deg);
            imgui.igText("  Pitch: %.2f", euler[1] * rad_to_deg);
            imgui.igText("  Yaw:   %.2f", euler[2] * rad_to_deg);
            imgui.igNewLine();
        } else {
            imgui.igText("Orientation: Not available (filter null)");
        }

        imgui.igSeparator();

        // Config Path Display
        const curr_path = std.mem.sliceTo(&self.config_path_buffer, 0);
        imgui.igText("Config Path: %s", @as([*c]const u8, curr_path.ptr));

        // Save/Load Buttons
        const button_disabled = motor_controller.getConnectionState() == .Running;

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
    }

    inline fn drawThrottleTesting(self: *Self, ctx: *const UIContext) void {
        const motor_controller = ctx.scene.motor_controller.?;
        if (motor_controller.getConnectionState() != .Running and motor_controller.getConnectionState() != .Ready) {
            imgui.igText("Please connect to controller to initiate testing!");
            return;
        }

        const motor_states = motor_controller.motor_states;

        // Individual sliders for each motor
        inline for (@typeInfo(Drone.Protocol.Motors).@"enum".fields) |motor| {
            const motor_enum = @as(Drone.Protocol.Motors, @enumFromInt(motor.value));
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

            // Direction buttons (only enabled when disarmed)
            if (!motor_state.armed) {
                if (imgui.igButton(
                    std.fmt.comptimePrint("Reverse###{d}", .{motor.value}),
                    imgui.ImVec2{ .x = 80, .y = 0 },
                )) {
                    const command = Drone.Protocol.Command{
                        .type = .ReverseDirection,
                        .motor = motor_enum,
                    };
                    _ = motor_controller.command_queue.push(command);
                }
                imgui.igSameLine(0, 10);
            }

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
                const command = Drone.Protocol.Command{
                    .type = if (motor_state.armed) .Disarm else .Arm,
                    .motor = motor_enum,
                };
                _ = motor_controller.command_queue.push(command);
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
                motor_controller.config.global_max_throttle,
                "%.1f",
                imgui.ImGuiSliderFlags_None,
            )) {
                if (motor_state.armed) {
                    const command = Drone.Protocol.Command{
                        .type = .SetSpeed,
                        .speed = throttle,
                        .motor = motor_enum,
                    };
                    _ = motor_controller.command_queue.push(command);
                }
            }

            if (!motor_state.armed) {
                imgui.igPopStyleVar(1);
            }

            imgui.igSpacing();
        }

        if (imgui.igSliderFloat(
            "All Armed Motors Throttle (%)",
            &self.global_throttle,
            0.0,
            motor_controller.config.global_max_throttle,
            "%.1f",
            imgui.ImGuiSliderFlags_None,
        )) {
            inline for (@typeInfo(Drone.Protocol.Motors).@"enum".fields) |motor| {
                if (motor_states[motor.value].armed) {
                    const command = Drone.Protocol.Command{
                        .type = .SetSpeed,
                        .speed = self.global_throttle,
                        .motor = @enumFromInt(motor.value),
                    };
                    _ = motor_controller.command_queue.push(command);
                }
            }
        }
    }

    inline fn drawServerConfig(self: *Self, ctx: *const UIContext) void {
        const motor_controller = ctx.scene.motor_controller.?;
        const conn_state = motor_controller.getConnectionState();
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
                motor_controller.config.allocator.free(motor_controller.config.local_ip);
                motor_controller.config.local_ip = motor_controller.config.allocator.dupe(u8, self.local_ip_buffer[0..ip_len]) catch |err| {
                    std.debug.print("Failed to allocate memory for IP: {}\n", .{err});
                    return;
                };
            }
        }

        var local_port = @as(c_int, motor_controller.config.local_port);
        if (imgui.igInputInt(
            "Local Port",
            &local_port,
            0,
            0,
            if (config_locked) imgui.ImGuiInputTextFlags_ReadOnly else imgui.ImGuiInputTextFlags_None,
        )) {
            if (!config_locked) {
                motor_controller.config.local_port = @intCast(@max(0, @min(local_port, 65535)));
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
                motor_controller.config.allocator.free(motor_controller.config.controller_ip);
                motor_controller.config.controller_ip = motor_controller.config.allocator.dupe(u8, self.controller_ip_buffer[0..ip_len]) catch |err| {
                    std.debug.print("Failed to allocate memory for IP: {}\n", .{err});
                    return;
                };
            }
        }

        var controller_port = @as(c_int, motor_controller.config.controller_port);
        if (imgui.igInputInt(
            "Controller Port",
            &controller_port,
            0,
            0,
            if (config_locked) imgui.ImGuiInputTextFlags_ReadOnly else imgui.ImGuiInputTextFlags_None,
        )) {
            if (!config_locked) {
                motor_controller.config.controller_port = @intCast(@max(0, @min(controller_port, 65535)));
            }
        }

        if (config_locked) {
            imgui.igPopStyleVar(1);
        }
    }

    inline fn drawProtocolConfig(self: *Self, ctx: *const UIContext) void {
        // DShot Protocol Selection
        const motor_controller = ctx.scene.motor_controller.?;

        imgui.igNewLine();
        inline for (@typeInfo(Drone.DSHOT).@"enum".fields) |protocol| {
            if (protocol.value > 0) {
                imgui.igSameLine(0, 4);
            }

            const is_selected = protocol.value == @intFromEnum(motor_controller.config.dshot_protocol);
            Styles.pushButtonColors(is_selected);

            const protocol_str = std.fmt.allocPrintZ(self.allocator, "DShot {d}", .{protocol.value}) catch unreachable;
            defer self.allocator.free(protocol_str);

            if (imgui.igButton(
                protocol_str,
                imgui.ImVec2{ .x = 90, .y = 0 },
            )) {
                motor_controller.config.dshot_protocol = @enumFromInt(protocol.value);
            }

            Styles.popButtonColors();
        }

        imgui.igNewLine();

        // Global Max Throttle
        var global_max = motor_controller.config.global_max_throttle;
        if (imgui.igSliderFloat(
            "Global Max Throttle (%)",
            &global_max,
            0.0,
            100.0,
            "%.1f",
            imgui.ImGuiSliderFlags_None,
        )) {
            motor_controller.config.global_max_throttle = global_max;
        }
    }

    inline fn drawMotorConfig(self: *Self, ctx: *const UIContext) void {
        // Motor Selection Buttons
        const motor_controller = ctx.scene.motor_controller.?;

        imgui.igText("Select Motor:");
        inline for (@typeInfo(Drone.Protocol.Motors).@"enum".fields) |motor| {
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
        const motor = &motor_controller.config.motors[@intFromEnum(self.selected_motor)];
        var pin = @as(c_int, motor.pin);
        if (imgui.igInputInt("GPIO Pin", &pin, 0, 0, imgui.ImGuiInputTextFlags_CharsDecimal)) {
            motor.pin = @intCast(@max(0, @min(pin, 27)));
        }

        imgui.igBeginDisabled(true);
        var direction: c_int = @intFromEnum(motor.direction);
        if (imgui.igCombo_Str(
            "Direction",
            &direction,
            "Clockwise\x00Counterclockwise\x00",
            2,
        )) {
            motor.direction = @enumFromInt(direction);
        }
        imgui.igEndDisabled();

        var max_throttle = @min(motor.max_throttle, motor_controller.config.global_max_throttle);
        if (imgui.igSliderFloat(
            "Max Throttle (%)",
            &max_throttle,
            0.0,
            @min(100.0, motor_controller.config.global_max_throttle),
            "%.1f",
            imgui.ImGuiSliderFlags_None,
        )) {
            motor.max_throttle = max_throttle;
        }
    }

    inline fn drawSaveModal(self: *Self, ctx: *const UIContext) void {
        const motor_controller = ctx.scene.motor_controller.?;

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
                    motor_controller.config.saveToFile(self.config_path_buffer[0 .. std.mem.indexOfScalar(u8, &self.config_path_buffer, 0) orelse self.config_path_buffer.len]) catch |err| {
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
        const motor_controller = ctx.scene.motor_controller.?;

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
                    motor_controller.config = DroneConfig.loadFromFile(
                        motor_controller.config.allocator,
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

pub const BatteryStatusWindow = struct {
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
        const work_size = viewport.*.WorkSize;

        // Position in top right
        const window_pos = imgui.ImVec2{
            .x = work_pos.x + work_size.x - PAD,
            .y = work_pos.y + PAD,
        };
        const window_pos_pivot = imgui.ImVec2{
            .x = 1.0, // Right-aligned
            .y = 0.0, // Top-aligned
        };

        // Window setup
        imgui.igSetNextWindowPos(window_pos, imgui.ImGuiCond_Always, window_pos_pivot);
        imgui.igSetNextWindowViewport(viewport.*.ID);
        window_flags |= imgui.ImGuiWindowFlags_NoMove;
        imgui.igSetNextWindowBgAlpha(0.35); // Semi-transparent background
        imgui.igSetNextWindowSize(
            .{ .x = 185, .y = 115 }, // Fixed window dimensions
            imgui.ImGuiCond_Always,
        );

        // Window content
        if (imgui.igBegin("Battery Status", &self.visible, window_flags)) {
            const motor_controller = ctx.scene.motor_controller.?;
            const battery_info = motor_controller.connection_handler.battery_info;

            // Get the draw list for custom rendering
            const draw_list = imgui.igGetWindowDrawList();

            // Calculate dimensions for battery icon
            const battery_height = 30.0;
            const battery_width = 150.0;
            var cursor_pos: imgui.ImVec2 = undefined;
            imgui.igGetCursorScreenPos(&cursor_pos);
            const battery_x = cursor_pos.x + 10.0;
            const battery_y = cursor_pos.y + 5.0;

            const battery_outline_color = imgui.igColorConvertFloat4ToU32(
                .{ .x = 0.3, .y = 0.3, .z = 0.3, .w = 1.0 },
            );

            // Battery body outline (using individual coordinates for clarity)
            const body_min_x = battery_x;
            const body_min_y = battery_y;
            const body_max_x = battery_x + battery_width;
            const body_max_y = battery_y + battery_height;

            imgui.ImDrawList_AddRectFilled(
                draw_list,
                .{ .x = body_min_x, .y = body_min_y },
                .{ .x = body_max_x, .y = body_max_y },
                battery_outline_color,
                5.0,
                imgui.ImDrawFlags_RoundCornersAll,
            );

            // Battery cap (positive terminal)
            const cap_width = 6.0;
            const cap_height = 12.0;
            const cap_min_x = battery_x + battery_width;
            const cap_min_y = battery_y + (battery_height - cap_height) * 0.5;
            const cap_max_x = battery_x + battery_width + cap_width;
            const cap_max_y = battery_y + (battery_height + cap_height) * 0.5;

            imgui.ImDrawList_AddRectFilled(
                draw_list,
                .{ .x = cap_min_x, .y = cap_min_y },
                .{ .x = cap_max_x, .y = cap_max_y },
                battery_outline_color,
                2.0,
                imgui.ImDrawFlags_RoundCornersRight,
            );

            const fill_color_vec4: imgui.ImVec4 = if (battery_info.percentage > 75.0)
                .{ .x = 0.0, .y = 0.8, .z = 0.2, .w = 1.0 } // Green
            else if (battery_info.percentage > 25.0)
                .{ .x = 0.9, .y = 0.7, .z = 0.0, .w = 1.0 } // Yellow
            else
                .{ .x = 1.0, .y = 0.2, .z = 0.2, .w = 1.0 }; // Red

            const fill_color = imgui.igColorConvertFloat4ToU32(fill_color_vec4);

            // Calculate fill width based on percentage
            const padding = 3.0;
            const fill_width = @max(0.0, (battery_width - padding * 2.0) * (battery_info.percentage / 100.0));

            // Draw battery fill level
            if (fill_width > 0) {
                const fill_min_x = battery_x + padding;
                const fill_min_y = battery_y + padding;
                const fill_max_x = battery_x + padding + fill_width;
                const fill_max_y = battery_y + battery_height - padding;

                imgui.ImDrawList_AddRectFilled(
                    draw_list,
                    .{ .x = fill_min_x, .y = fill_min_y },
                    .{ .x = fill_max_x, .y = fill_max_y },
                    fill_color,
                    3.0,
                    imgui.ImDrawFlags_RoundCornersAll,
                );
            }

            // Add battery percentage text centered on the battery
            var text_buf: [16]u8 = undefined;
            const text = std.fmt.bufPrintZ(&text_buf, "{d:.1}%", .{battery_info.percentage}) catch "??%";

            var text_size: imgui.ImVec2 = undefined;
            imgui.igCalcTextSize(
                &text_size,
                text,
                null,
                false,
                0,
            );
            const text_x = battery_x + (battery_width - text_size.x) * 0.5;
            const text_y = battery_y + (battery_height - text_size.y) * 0.5;

            const text_shadow_color = imgui.igColorConvertFloat4ToU32(.{ .x = 0.0, .y = 0.0, .z = 0.0, .w = 0.5 });
            const text_color = imgui.igColorConvertFloat4ToU32(.{ .x = 1.0, .y = 1.0, .z = 1.0, .w = 1.0 });

            // Draw text shadow for better readability
            imgui.ImDrawList_AddText_Vec2(
                draw_list,
                .{ .x = text_x + 1, .y = text_y + 1 },
                text_shadow_color,
                text,
                null,
            );

            // Draw main text
            imgui.ImDrawList_AddText_Vec2(
                draw_list,
                .{ .x = text_x, .y = text_y },
                text_color,
                text,
                null,
            );

            // Make space for the battery visual
            imgui.igDummy(.{ .x = 0, .y = battery_height + 10 });

            // Battery information
            const cell_count = @intFromEnum(battery_info.type);
            imgui.igText("%dS LiPo - %.2fV", cell_count, battery_info.voltage);

            // Failsafe status
            if (battery_info.failsafe_active) {
                imgui.igTextColored(.{ .x = 1.0, .y = 0.0, .z = 0.0, .w = 1.0 }, "FAILSAFE ACTIVE");
            } else {
                imgui.igTextColored(.{ .x = 0.0, .y = 0.8, .z = 0.2, .w = 1.0 }, "Failsafe: Inactive");
            }

            // Warning message for low battery
            if (battery_info.percentage <= 25.0) {
                imgui.igSpacing();
                imgui.igTextColored(.{ .x = 1.0, .y = 0.0, .z = 0.0, .w = 1.0 }, "WARNING: Low battery!");
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
