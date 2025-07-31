const std = @import("std");
const imgui = @import("bindings/c.zig").imgui;
const Core = @import("ecs/Core.zig");
const ECSManager = @import("ecs/ECSManager.zig");
const Transform = @import("ecs/components/Transform.zig").TransformComponent;
const Renderable = @import("ecs/components/Renderer.zig").Renderable;
const Physics = @import("ecs/components/Physics.zig").PhysicsComponent;
const Controller = @import("ecs/components/Controller.zig").ControllerComponent;
const Camera = @import("ecs/components/Camera.zig").CameraComponent;
const Viewport = @import("ecs/components/Viewports.zig").ViewportComponent;
const Recorder = @import("ecs/components/Recorder.zig");
const IMUSensor = @import("ecs/components/IMUSensor.zig");
const FlightController = @import("ecs/components/FlightController.zig");
const FlightInput = @import("ecs/components/FlightInput.zig");

// const Vision = @import("Vision.zig");
// const Drone = @import("Drone.zig");
// const Sensors = @import("Sensors.zig");
const Math = @import("Math.zig");

const Vec3 = Math.Vec3;
// const DroneConfig = Drone.DroneConfig;
// const MotorController = Drone.MotorControllerClient;
// const StereoVO = Vision.StereoVO;
// const CameraPose = Vision.CameraPose;

inline fn cstr(s: [:0]const u8) [*:0]const u8 {
    return s;
}

const UIContext = struct {
    ecs: *ECSManager,
    // scene: *Scene,
    // StereoVO: *StereoVO,
    // pose_handler: *Sensors.PoseHandler,
};

const CameraMetadataOverlay = struct {
    const Self = @This();

    visible: bool = true,

    pub fn init() Self {
        return .{};
    }

    pub fn drawOverlay(self: *Self, ctx: *const UIContext, image_pos: imgui.ImVec2, img_w: f32) void {
        if (!self.visible) return;

        if (ctx.ecs.camera_system.active_camera_eid) |camera_eid| {
            if (ctx.ecs.transform_components.get(camera_eid)) |transform| {
                const pos = transform.world_transform.get_position();
                const trs = transform.world_transform.decomposeTRS();
                const euler_rad = trs.rotation.to_euler();

                const roll = Math.degrees(euler_rad[2]);
                const pitch = Math.degrees(euler_rad[0]);
                const yaw = Math.degrees(euler_rad[1]);

                const overlay_flags = imgui.ImGuiWindowFlags_NoTitleBar |
                    imgui.ImGuiWindowFlags_NoResize |
                    imgui.ImGuiWindowFlags_NoMove |
                    imgui.ImGuiWindowFlags_NoCollapse |
                    imgui.ImGuiWindowFlags_NoScrollbar |
                    imgui.ImGuiWindowFlags_AlwaysAutoResize |
                    imgui.ImGuiWindowFlags_NoSavedSettings |
                    imgui.ImGuiWindowFlags_NoInputs;

                // Position in top-right corner of the rendered image (with padding)
                const overlay_pos = imgui.ImVec2{
                    .x = image_pos.x + img_w - 200, // 200px from right edge of image
                    .y = image_pos.y + 10, // 10px from top of image
                };

                imgui.igSetNextWindowPos(overlay_pos, imgui.ImGuiCond_Always, .{ .x = 0, .y = 0 });

                imgui.igPushStyleColor_Vec4(imgui.ImGuiCol_WindowBg, .{ .x = 0.2, .y = 0.2, .z = 0.2, .w = 0.7 });
                imgui.igPushStyleVar_Vec2(imgui.ImGuiStyleVar_WindowPadding, .{ .x = 10, .y = 8 });
                imgui.igPushStyleVar_Float(imgui.ImGuiStyleVar_WindowRounding, 6.0);

                if (imgui.igBegin("##CameraOverlay", &self.visible, overlay_flags)) {
                    imgui.igTextColored(.{ .x = 0.9, .y = 0.9, .z = 0.9, .w = 1.0 }, "Camera Position:");
                    imgui.igText("X: %.2f  Y: %.2f  Z: %.2f", pos.x(), pos.y(), pos.z());

                    imgui.igSeparator();

                    imgui.igTextColored(.{ .x = 0.9, .y = 0.9, .z = 0.9, .w = 1.0 }, "Orientation:");
                    imgui.igText("Roll:  %.1f°", roll);
                    imgui.igText("Pitch: %.1f°", pitch);
                    imgui.igText("Yaw:   %.1f°", yaw);
                }
                imgui.igEnd();

                imgui.igPopStyleVar(2);
                imgui.igPopStyleColor(1);
            }
        }
    }
};

const FlightControlOverlay = struct {
    const Self = @This();

    visible: bool = true,
    last_update_time: f64 = 0,
    update_interval: f64 = 0.5, // Update every 500ms
    
    // Cached display values to avoid flickering
    cached_gyro: [3]f32 = [3]f32{0, 0, 0},
    cached_accel: [3]f32 = [3]f32{0, 0, 0},
    cached_setpoint_thrust: f32 = 0,
    cached_setpoint_rates: [3]f32 = [3]f32{0, 0, 0},
    cached_attitude: [4]f32 = [4]f32{0, 0, 0, 1},
    cached_rate_estimate: [3]f32 = [3]f32{0, 0, 0},
    cached_input_thrust: f32 = 0,
    cached_input_rates: [3]f32 = [3]f32{0, 0, 0},
    cached_keys: [4]bool = [4]bool{false, false, false, false}, // W, S, A, D
    cached_mouse: [2]f32 = [2]f32{0, 0},
    has_imu_data: bool = false,

    pub fn init() Self {
        return .{};
    }

    pub fn drawOverlay(self: *Self, ctx: *const UIContext, image_pos: imgui.ImVec2, _: f32) void {
        if (!self.visible) return;

        // Get current time and check if we should update cached values
        const current_time = ctx.ecs.globals.last_frame_time;
        const should_update = (current_time - self.last_update_time) >= self.update_interval;

        // Find the drone entity (first entity with flight controller)
        var drone_eid: ?Core.EntityID = null;
        var fc_iter = ctx.ecs.flight_controller_components.iterator();
        if (fc_iter.next()) |entry| {
            drone_eid = entry.entity_id;
        }

        if (drone_eid) |eid| {
            // Update cached values if enough time has passed
            if (should_update) {
                self.updateCachedValues(ctx, eid);
                self.last_update_time = current_time;
            }
            const overlay_flags = imgui.ImGuiWindowFlags_NoTitleBar |
                imgui.ImGuiWindowFlags_NoResize |
                imgui.ImGuiWindowFlags_NoMove |
                imgui.ImGuiWindowFlags_NoCollapse |
                imgui.ImGuiWindowFlags_NoScrollbar |
                imgui.ImGuiWindowFlags_NoSavedSettings |
                imgui.ImGuiWindowFlags_NoInputs;

            // Position in top-left corner of the rendered image (with padding)
            const overlay_pos = imgui.ImVec2{
                .x = image_pos.x + 10, // 10px from left edge of image
                .y = image_pos.y + 10, // 10px from top of image
            };

            imgui.igSetNextWindowPos(overlay_pos, imgui.ImGuiCond_Always, .{ .x = 0, .y = 0 });
            imgui.igSetNextWindowSize(.{ .x = 320, .y = 360 }, imgui.ImGuiCond_Always);

            imgui.igPushStyleColor_Vec4(imgui.ImGuiCol_WindowBg, .{ .x = 0.1, .y = 0.1, .z = 0.1, .w = 0.8 });
            imgui.igPushStyleVar_Vec2(imgui.ImGuiStyleVar_WindowPadding, .{ .x = 12, .y = 10 });
            imgui.igPushStyleVar_Float(imgui.ImGuiStyleVar_WindowRounding, 8.0);

            if (imgui.igBegin("##FlightControlOverlay", &self.visible, overlay_flags)) {
                // Header
                imgui.igTextColored(.{ .x = 0.3, .y = 0.8, .z = 1.0, .w = 1.0 }, "Flight Control Debug");
                imgui.igSeparator();

                // IMU Data
                imgui.igTextColored(.{ .x = 0.9, .y = 0.7, .z = 0.3, .w = 1.0 }, "IMU Sensor:");
                
                if (self.has_imu_data) {
                    imgui.igText("Gyro (rad/s): %.3f, %.3f, %.3f", self.cached_gyro[0], self.cached_gyro[1], self.cached_gyro[2]);
                    imgui.igText("Accel (m/s²): %.3f, %.3f, %.3f", self.cached_accel[0], self.cached_accel[1], self.cached_accel[2]);
                } else {
                    imgui.igTextColored(.{ .x = 0.8, .y = 0.4, .z = 0.4, .w = 1.0 }, "No IMU data available");
                }
                
                imgui.igText("Sample Rate: 1000 Hz");
                imgui.igSeparator();

                // Flight Controller Data
                imgui.igTextColored(.{ .x = 0.3, .y = 0.9, .z = 0.4, .w = 1.0 }, "Flight Controller:");
                
                // Control setpoints
                imgui.igText("Setpoints:");
                imgui.igText("  Thrust: %.2f N", self.cached_setpoint_thrust);
                imgui.igText("  Rates (rad/s): %.2f, %.2f, %.2f", 
                    self.cached_setpoint_rates[0], self.cached_setpoint_rates[1], self.cached_setpoint_rates[2]);
                
                // Current estimates
                imgui.igText("Estimates:");
                imgui.igText("  Attitude: %.2f, %.2f, %.2f, %.2f", 
                    self.cached_attitude[0], self.cached_attitude[1], 
                    self.cached_attitude[2], self.cached_attitude[3]);
                imgui.igText("  Rates: %.2f, %.2f, %.2f", 
                    self.cached_rate_estimate[0], self.cached_rate_estimate[1], self.cached_rate_estimate[2]);
                
                imgui.igSeparator();

                // Flight Input Data
                imgui.igTextColored(.{ .x = 0.9, .y = 0.5, .z = 0.9, .w = 1.0 }, "Input Commands:");
                
                imgui.igText("Thrust: %.2f N", self.cached_input_thrust);
                imgui.igText("Rates: %.2f, %.2f, %.2f", 
                    self.cached_input_rates[0], self.cached_input_rates[1], self.cached_input_rates[2]);
                
                // Input state
                imgui.igText("Keys: W:%s S:%s A:%s D:%s", 
                    if (self.cached_keys[0]) cstr("ON") else cstr("OFF"),
                    if (self.cached_keys[1]) cstr("ON") else cstr("OFF"),
                    if (self.cached_keys[2]) cstr("ON") else cstr("OFF"),
                    if (self.cached_keys[3]) cstr("ON") else cstr("OFF"));
                imgui.igText("Mouse: %.1f, %.1f", self.cached_mouse[0], self.cached_mouse[1]);
            }
            imgui.igEnd();

            imgui.igPopStyleVar(2);
            imgui.igPopStyleColor(1);
        }
    }

    fn updateCachedValues(self: *Self, ctx: *const UIContext, eid: Core.EntityID) void {
        // Update IMU data
        if (ctx.ecs.imu_sensor_components.get(eid)) |imu_component| {
            if (imu_component.getLatestSample()) |sample| {
                self.cached_gyro = sample.gyro;
                self.cached_accel = sample.accel;
                self.has_imu_data = true;
            } else {
                self.has_imu_data = false;
            }
        }

        // Update Flight Controller data
        if (ctx.ecs.flight_controller_components.get(eid)) |fc_component| {
            self.cached_setpoint_thrust = fc_component.setpoints.desired_thrust;
            self.cached_setpoint_rates = fc_component.setpoints.desired_rates;
            self.cached_attitude = [4]f32{
                fc_component.attitude_estimate.x(),
                fc_component.attitude_estimate.y(),
                fc_component.attitude_estimate.z(),
                fc_component.attitude_estimate.w(),
            };
            self.cached_rate_estimate = [3]f32{
                fc_component.rate_estimate.x(),
                fc_component.rate_estimate.y(),
                fc_component.rate_estimate.z(),
            };
        }

        // Update Flight Input data
        if (ctx.ecs.flight_input_components.get(eid)) |input_component| {
            self.cached_input_thrust = input_component.control_commands.desired_thrust;
            self.cached_input_rates = input_component.control_commands.desired_rates;
            self.cached_keys = [4]bool{
                input_component.input_state.throttle_up,
                input_component.input_state.throttle_down,
                input_component.input_state.yaw_left,
                input_component.input_state.yaw_right,
            };
            self.cached_mouse = [2]f32{
                input_component.input_state.mouse_dx,
                input_component.input_state.mouse_dy,
            };
        }
    }
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
    sidebar_width: f32 = 0, // Default width, will be adjusted by user
    is_resizing: bool = false, // Track if currently resizing

    entities_window: *EntitiesWindow,
    timeline_recorder: TimelineRecorder,
    camera_overlay: CameraMetadataOverlay,
    flight_control_overlay: FlightControlOverlay,

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .visible = true,
            .entities_window = try EntitiesWindow.init(allocator),
            .timeline_recorder = TimelineRecorder.init(allocator),
            .camera_overlay = CameraMetadataOverlay.init(),
            .flight_control_overlay = FlightControlOverlay.init(),
        };
        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.destroy(self);
    }

    /// This is the top-level "root" that occupies the entire screen
    pub fn draw(self: *Self, ctx: *const UIContext) void {
        if (!self.visible) return;

        const vp = imgui.igGetMainViewport();
        imgui.igSetNextWindowPos(vp.*.WorkPos, imgui.ImGuiCond_Always, .{ .x = 0, .y = 0 });
        imgui.igSetNextWindowSize(vp.*.WorkSize, imgui.ImGuiCond_Always);

        const root_flags =
            imgui.ImGuiWindowFlags_NoDecoration | imgui.ImGuiWindowFlags_NoMove | imgui.ImGuiWindowFlags_NoCollapse | imgui.ImGuiWindowFlags_NoBringToFrontOnFocus | imgui.ImGuiWindowFlags_NoNavFocus | imgui.ImGuiWindowFlags_NoDocking;

        imgui.igPushStyleVar_Vec2(imgui.ImGuiStyleVar_WindowPadding, .{ .x = 0, .y = 0 });
        if (!imgui.igBegin("RootWindow##FullScreen", &self.visible, root_flags)) {
            imgui.igPopStyleVar(1);
            imgui.igEnd();
            return;
        }
        imgui.igPopStyleVar(1);

        const bg_col = imgui.igColorConvertFloat4ToU32(.{ .x = 0.12, .y = 0.14, .z = 0.20, .w = 1.0 }); // dark‑blue‑gray
        imgui.igPushStyleColor_U32(imgui.ImGuiCol_ChildBg, bg_col);
        imgui.igPopStyleColor(1); // restore style

        var avail: imgui.ImVec2 = undefined;
        imgui.igGetContentRegionAvail(&avail);

        const topbar_height: f32 = 0;
        const main_area_h: f32 = avail.y - topbar_height;

        if (self.sidebar_width == 0)
            self.sidebar_width = avail.x * 0.15; // 15 % first frame

        self.sidebar_width = std.math.clamp(
            self.sidebar_width,
            200.0, // min 200 px
            avail.x * 0.6, //maximum of 60% of window width
        );

        // ──────────────────────────────── Begin 2‑column resizable table
        if (imgui.igBeginTable("MainLayoutTable", 2, imgui.ImGuiTableFlags_Resizable | imgui.ImGuiTableFlags_NoPadInnerX | imgui.ImGuiTableFlags_BordersInnerV | imgui.ImGuiTableFlags_SizingFixedFit, .{ .x = 0, .y = 0 }, 0)) {
            // column 0 = sidebar (fixed), 1 = content (stretch)
            imgui.igTableSetupColumn("Sidebar", imgui.ImGuiTableColumnFlags_WidthFixed, self.sidebar_width, 0);
            imgui.igTableSetupColumn("Content", imgui.ImGuiTableColumnFlags_WidthStretch, 0, 1);

            imgui.igTableNextRow(imgui.ImGuiLogFlags_None, 0);

            // ========================================================== Sidebar
            _ = imgui.igTableSetColumnIndex(0);

            imgui.igPushStyleVar_Vec2(imgui.ImGuiStyleVar_WindowPadding, .{ .x = 10, .y = 10 });
            _ = imgui.igBeginChild_Str("SidebarChild", .{ .x = 0, .y = main_area_h }, // width 0 ⇒ use column width
                imgui.ImGuiChildFlags_None, imgui.ImGuiWindowFlags_None);

            const hdr_flags: imgui.ImGuiTreeNodeFlags =
                imgui.ImGuiTreeNodeFlags_DefaultOpen | imgui.ImGuiTreeNodeFlags_SpanAvailWidth;

            imgui.igText("Navigation");
            imgui.igSeparator();

            if (imgui.igCollapsingHeader_TreeNodeFlags("Viewports###Viewports", hdr_flags))
                ViewportManager(&self.visible, ctx);

            imgui.igSeparator();

            if (imgui.igCollapsingHeader_TreeNodeFlags("Entities###SidebarEntities", hdr_flags)) {
                const ent_h: f32 = 680.0;
                _ = imgui.igBeginChild_Str("EntitiesSidebarChild", .{ .x = 0, .y = ent_h }, imgui.ImGuiChildFlags_None, imgui.ImGuiWindowFlags_HorizontalScrollbar);
                self.entities_window.drawSidebar(ctx);
                imgui.igEndChild();
            }

            imgui.igEndChild();
            imgui.igPopStyleVar(1);

            // ========================================================== Content
            _ = imgui.igTableSetColumnIndex(1);

            var contentAvail: imgui.ImVec2 = undefined;
            imgui.igGetContentRegionAvail(&contentAvail);

            const content_w = contentAvail.x;
            const main_viewport_h = main_area_h * 0.8;
            const sub_viewports_h = main_area_h - main_viewport_h;

            // ---------- Main viewport ----------
            imgui.igPushStyleVar_Vec2(imgui.ImGuiStyleVar_WindowPadding, .{ .x = 10, .y = 10 });
            _ = imgui.igBeginChild_Str("MainViewportArea", .{ .x = content_w, .y = main_viewport_h }, imgui.ImGuiChildFlags_None, imgui.ImGuiWindowFlags_None);

            var camera_system = &ctx.ecs.camera_system;
            var vp_store = ctx.ecs.viewport_components;

            if (camera_system.active_camera_eid) |main_cam| {
                if (vp_store.get(main_cam)) |main_entry| {
                    const mvp = main_entry.vp;
                    if (mvp.enabled) {
                        imgui.igTextColored(.{ .x = 0, .y = 0.8, .z = 1, .w = 1 }, "Main Camera: %s", mvp.name.ptr);

                        // keep aspect ratio
                        const tex_w: f32 = @floatFromInt(mvp.fbo.width);
                        const tex_h: f32 = @floatFromInt(mvp.fbo.height);
                        const aspect = tex_w / tex_h;

                        var region: imgui.ImVec2 = undefined;
                        imgui.igGetContentRegionAvail(&region);

                        var img_w = region.x;
                        var img_h = img_w / aspect;
                        if (img_h > region.y - 30) {
                            img_h = region.y - 30;
                            img_w = img_h * aspect;
                        }
                        if (img_w < region.x)
                            imgui.igSetCursorPosX(imgui.igGetCursorPosX() + (region.x - img_w) * 0.5);

                        // Get the position of the image before drawing it
                        var image_pos: imgui.ImVec2 = undefined;
                        imgui.igGetCursorScreenPos(&image_pos);

                        const tex_id: imgui.ImTextureID = @intCast(mvp.fbo.texture);
                        imgui.igImage(tex_id, .{ .x = img_w, .y = img_h }, .{ .x = 0, .y = 1 }, .{ .x = 1, .y = 0 }, .{ .x = 1, .y = 1, .z = 1, .w = 1 }, .{ .x = 0.3, .y = 0.3, .z = 0.3, .w = 1 });

                        // Draw camera metadata overlay on top of the image
                        self.camera_overlay.drawOverlay(ctx, image_pos, img_w);
                        
                        // Draw flight control overlay
                        self.flight_control_overlay.drawOverlay(ctx, image_pos, img_w);
                    }
                }
            } else {
                imgui.igTextColored(.{ .x = 1, .y = 0, .z = 0, .w = 1 }, "No main camera set. Please select a camera");
            }

            imgui.igEndChild();
            imgui.igPopStyleVar(1);

            // ---------- Sub‑viewports ----------
            imgui.igPushStyleVar_Vec2(imgui.ImGuiStyleVar_WindowPadding, .{ .x = 10, .y = 10 });
            _ = imgui.igBeginChild_Str("SubViewportsArea", .{ .x = content_w, .y = sub_viewports_h }, imgui.ImGuiChildFlags_None, imgui.ImGuiWindowFlags_None);

            imgui.igText("Other Active Cameras:");
            imgui.igSeparator();

            const min_vp_w: f32 = 250.0;
            const spacing: f32 = 15.0;
            const max_per_row: usize = @intFromFloat(@floor(content_w / (min_vp_w + spacing)));

            if (max_per_row > 0) {
                var in_row: usize = 0;
                var it = vp_store.iterator();
                while (it.next()) |entry| {
                    const eid = entry.entity_id;
                    const _vp = entry.component.vp;

                    if (!_vp.enabled) continue;
                    if (camera_system.active_camera_eid) |main_eid| {
                        if (eid.id == main_eid.id) continue; // skip main
                    }

                    if (in_row >= max_per_row) {
                        in_row = 0;
                        imgui.igNewLine();
                    }
                    if (in_row > 0) imgui.igSameLine(0, spacing);

                    const tex_w: f32 = @floatFromInt(_vp.fbo.width);
                    const tex_h: f32 = @floatFromInt(_vp.fbo.height);
                    const aspect = tex_w / tex_h;
                    const draw_w = min_vp_w;
                    const draw_h = draw_w / aspect;

                    imgui.igBeginGroup();
                    imgui.igText("%s", _vp.name.ptr);

                    const tex_id: imgui.ImTextureID = @intCast(_vp.fbo.texture);
                    imgui.igImage(tex_id, .{ .x = draw_w, .y = draw_h }, .{ .x = 0, .y = 1 }, .{ .x = 1, .y = 0 }, .{ .x = 1, .y = 1, .z = 1, .w = 1 }, .{ .x = 0.2, .y = 0.2, .z = 0.2, .w = 1 });

                    if (imgui.igIsItemClicked(imgui.ImGuiMouseButton_Left)) {
                        camera_system.active_camera_eid = eid;

                        if (ctx.ecs.findControllerAncestor(eid)) |ctrl_eid| {
                            ctx.ecs.control_system.active_controller_eid = ctrl_eid;
                        } else {
                            ctx.ecs.control_system.active_controller_eid = null; // no controller
                        }
                    }
                    if (imgui.igIsItemHovered(imgui.ImGuiHoveredFlags_None)) {
                        _ = imgui.igBeginTooltip();
                        imgui.igText("Click to set as main viewport");
                        imgui.igEndTooltip();
                    }
                    imgui.igEndGroup();
                    in_row += 1;
                }
            }

            self.timeline_recorder.draw(ctx);
            imgui.igEndChild();
            imgui.igPopStyleVar(1);

            // Remember the user‑chosen sidebar width for next frame
            self.sidebar_width = imgui.igGetColumnWidth(0);

            imgui.igEndTable();
        } // end table

        imgui.igEnd(); // end root window
    }

    // The ViewportManager function as a widget (not a window)
    pub fn ViewportManager(visible: *bool, ctx: *const UIContext) void {
        _ = visible;

        var viewports = ctx.ecs.viewport_components;
        // Let user pick which viewports are active
        imgui.igText("Active Viewports:");
        imgui.igSeparator();

        var it_cam = viewports.iterator();
        while (it_cam.next()) |entry| {
            const viewport = entry.component;
            _ = imgui.igCheckbox(viewport.vp.name, &viewport.active);
        }
    }
};

const TimelineRecorder = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    scrub_pos: f32 = 0,
    mode: enum { Record, Playback } = .Record,
    load_popup_open: bool = false,
    selected_file: []const u8 = "",
    file_list: std.ArrayList([:0]const u8),

    pub fn init(alloc: std.mem.Allocator) Self {
        return Self{
            .allocator = alloc,
            .file_list = std.ArrayList([:0]const u8).init(alloc),
        };
    }

    /// Draws the timeline bar. Call *once per frame*.
    /// Place it immediately after the Sub‑viewports child in RootWindow.
    pub fn draw(self: *Self, ctx: *const UIContext) void {
        const ecs = ctx.ecs;
        var rec = ecs.recorder_system;

        var avail: imgui.ImVec2 = undefined;
        imgui.igGetContentRegionAvail(&avail);
        const h: f32 = 70.0;
        _ = imgui.igBeginChild_Str("TimelineWidget", .{ .x = avail.x, .y = h }, imgui.ImGuiChildFlags_None, imgui.ImGuiWindowFlags_None);

        if (self.mode == .Record) {
            const start_label = if (rec.is_recording) "Stop" else "Start";
            if (imgui.igButton(start_label, .{ .x = 50, .y = 0 })) {
                if (!rec.is_recording) {
                    rec.toggle() catch |err| {
                        std.debug.print("Error starting recording: {any}\n", .{err});
                    };
                } else {
                    const ts = std.time.milliTimestamp(); // or format your own
                    const path = std.fmt.allocPrint(self.allocator, "captures/rec_{d}.tmp", .{ts}) catch |err| {
                        std.debug.print("Failed to create Capture path: {any}\n", .{err});
                        return;
                    };
                    rec.saveToDisk(path) catch |err| std.debug.print("Save failed: {any}\n", .{err});
                    self.mode = .Playback;
                    self.scrub_pos = 0;
                }
            }
            imgui.igSameLine(0, 8);

            if (rec.duration == 0) imgui.igBeginDisabled(true);
            if (imgui.igButton("Pause", .{ .x = 50, .y = 0 })) {
                rec.toggle() catch {};
            }
            if (rec.duration == 0) imgui.igEndDisabled();
            imgui.igSameLine(0, 8);

            // Reset (disabled if nothing recorded)
            if (rec.duration == 0) imgui.igBeginDisabled(true);
            if (imgui.igButton("Reset", .{ .x = 50, .y = 0 })) {
                rec.reset() catch {};
                self.scrub_pos = 0;
            }
            if (rec.duration == 0) imgui.igEndDisabled();
            imgui.igSameLine(0, 8);

            // Reset (disabled if nothing recorded)
            if (rec.is_recording) imgui.igBeginDisabled(true);
            if (imgui.igButton("Load", .{ .x = 50, .y = 0 })) {
                self.load_popup_open = true;
            }
            if (rec.is_recording) imgui.igEndDisabled();
            imgui.igSameLine(0, 8);

            const mins: i32 = @intFromFloat(@floor(rec.duration / 60));
            const secs = @mod(@as(i32, @intFromFloat(@floor(rec.duration))), 60);
            imgui.igText("Len: %02d:%02d  |  Size: %.02f MB\x00", mins, secs, rec.getMegabytes());

            imgui.igPushItemWidth(-1);
            var max_dur = rec.duration;
            if (max_dur == 0) max_dur = 1; // avoid 0‑range slider
            if (imgui.igSliderFloat("##scrub", &self.scrub_pos, 0, max_dur, "%.2fs", 0)) {
                rec.seek(ecs, self.scrub_pos) catch |err| {
                    std.debug.print("Error seeking: {any}\n", .{err});
                };
            }
            imgui.igPopItemWidth();
        } else {
            // Show play/pause or just scrub slider
            if (rec.is_playback) {
                self.scrub_pos += @floatCast(ctx.ecs.globals.dt);
                if (self.scrub_pos > rec.duration) {
                    self.scrub_pos = 0;
                    rec.is_playback = false;
                } else {
                    rec.seek(ctx.ecs, self.scrub_pos) catch |err| {
                        std.debug.print("Failed to playback capture: {any}\n", .{err});
                    };
                }
            }

            const start_label = if (rec.is_playback) "Pause" else "Start";
            if (imgui.igButton(start_label, .{ .x = 50, .y = 0 })) {
                rec.is_playback = !rec.is_playback;
            }
            imgui.igSameLine(0, 8);
            if (imgui.igButton("Load", .{ .x = 50, .y = 0 })) {
                rec.is_playback = false;
                self.load_popup_open = true;
            }
            imgui.igSameLine(0, 8);
            if (imgui.igButton("Record Mode", .{ .x = 100, .y = 0 })) {
                rec.is_playback = false;
                rec.reset() catch {};
                self.mode = .Record;
                self.scrub_pos = 0;
            }

            imgui.igSameLine(0, 8);
            const curr_min: i32 = @intFromFloat(@floor(self.scrub_pos / 60));
            const curr_sec = @mod(@as(i32, @intFromFloat(@floor(self.scrub_pos))), 60);
            const mins: i32 = @intFromFloat(@floor(rec.duration / 60));
            const secs = @mod(@as(i32, @intFromFloat(@floor(rec.duration))), 60);
            imgui.igText("%02d:%02d / %02d:%02d   |  Size: %.02f MB\x00", curr_min, curr_sec, mins, secs, rec.getMegabytes());

            imgui.igPushItemWidth(-1);
            const max_dur = rec.duration;
            if (imgui.igSliderFloat("##scrub", &self.scrub_pos, 0, max_dur, "%.2fs", 0)) {
                rec.seek(ctx.ecs, self.scrub_pos) catch {};
            }
            imgui.igPopItemWidth();
        }

        if (self.load_popup_open) {
            imgui.igOpenPopup_Str("Load Capture\x00", imgui.ImGuiPopupFlags_None);
            self.load_popup_open = false;
        }

        if (imgui.igBeginPopupModal("Load Capture\x00", null, imgui.ImGuiWindowFlags_AlwaysAutoResize)) {
            // on first open, enumerate files
            if (self.file_list.items.len == 0) {
                var dir = std.fs.cwd().openDir("captures", .{ .iterate = true }) catch @panic("Failed to open captures dir");
                defer dir.close();

                var dir_it = dir.iterate();
                while (dir_it.next() catch @panic("Failed to walk captures dir")) |entry| {
                    if (entry.kind == .file) {
                        const name = self.allocator.dupeZ(u8, entry.name) catch @panic("Failed to dupe capture name");
                        self.file_list.append(name) catch @panic("Failed to append capture file to entries");
                    }
                }
            }

            // list them
            for (self.file_list.items) |fname| {
                const selected = std.mem.eql(u8, fname, self.selected_file);
                if (imgui.igSelectable_Bool(fname.ptr, selected, imgui.ImGuiSelectableFlags_NoAutoClosePopups, .{ .x = 0, .y = 0 })) {
                    self.selected_file = fname;
                }
            }

            if (imgui.igButton("Load Selected", .{ .x = 120, .y = 0 }) and self.selected_file.len > 0) {
                const fullpath = std.fmt.allocPrint(self.allocator, "captures/{s}", .{self.selected_file}) catch |err| {
                    std.debug.print("Failed to create fullpath for selected capture: {any}\n", .{err});
                    @panic("Failed to create fullpath for selected capture");
                };

                rec.reset() catch {};
                rec.loadFromDisk(fullpath) catch |err| {
                    std.debug.print("Failed to load from disk: {any}\n", .{err});
                };
                self.mode = .Playback;
                self.scrub_pos = 0;
                imgui.igCloseCurrentPopup();
                self.file_list.clearAndFree();
            }
            imgui.igSameLine(0, 8);
            if (imgui.igButton("Cancel", .{ .x = 80, .y = 0 })) {
                imgui.igCloseCurrentPopup();
                self.file_list.clearAndFree();
            }

            imgui.igEndPopup();
        }
        imgui.igEndChild();
    }
};

/// Window that visualises all entities → components → field values
pub const EntitiesWindow = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    visible: bool = true,
    selected_id: ?Core.EntityID = null,
    show_collision_debug: bool = false,
    global_renderable_visibility: bool = true,

    pub fn init(alloc: std.mem.Allocator) !*Self {
        const self = try alloc.create(Self);
        self.* = .{ .allocator = alloc };
        return self;
    }
    pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
        alloc.destroy(self);
    }

    // ──────────────────────────────────────────────────────────────────────
    pub fn draw(self: *Self, ctx: *const @import("UI.zig").UIContext) void {
        if (!self.visible) return;
        if (!imgui.igBegin("Entities", &self.visible, imgui.ImGuiWindowFlags_None)) {
            imgui.igEnd();
            return;
        }

        // Global controls at the top
        if (imgui.igTreeNode_Str("Global Controls")) {
            // Collision debug toggle
            const old_collision_state = self.show_collision_debug;
            if (imgui.igCheckbox("Show Collision Debug", &self.show_collision_debug)) {
                // State changed, toggle collision debug visualization
                ctx.ecs.collision_system.setDebugWireframes(self.show_collision_debug) catch |err| {
                    std.debug.print("Failed to toggle collision debug: {}\n", .{err});
                    self.show_collision_debug = old_collision_state; // Revert on error
                };
            }
            if (imgui.igIsItemHovered(imgui.ImGuiHoveredFlags_None)) {
                imgui.igSetTooltip("Show wireframe boxes around colliders.\nGreen = Dynamic, Blue = Static");
            }

            // Global renderable visibility toggle
            if (imgui.igCheckbox("Global Renderable Visibility", &self.global_renderable_visibility)) {
                // Toggle visibility for all renderables
                var it = ctx.ecs.renderer_components.iterator();
                while (it.next()) |entry| {
                    entry.component.is_visible = self.global_renderable_visibility;
                }
            }
            if (imgui.igIsItemHovered(imgui.ImGuiHoveredFlags_None)) {
                imgui.igSetTooltip("Toggle visibility of all renderable objects in the scene");
            }

            imgui.igTreePop();
        }
        imgui.igSeparator();

        // split into two columns: hierarchy (left) | details (right)
        if (imgui.igBeginTable("EntitiesTable", 2, imgui.ImGuiTableFlags_Resizable | imgui.ImGuiTableFlags_BordersInnerV, .{ .x = 0, .y = 0 }, 0)) {
            imgui.igTableSetupColumn("Hierarchy", imgui.ImGuiTableColumnFlags_WidthFixed, 250.0, 0);
            imgui.igTableSetupColumn("Details", imgui.ImGuiTableColumnFlags_WidthStretch, 0.0, 0);

            imgui.igTableNextRow(imgui.ImGuiTableRowFlags_None, 0.0);
            _ = imgui.igTableSetColumnIndex(0);

            // ───────── LEFT SIDE : hierarchy tree ─────────
            const ecs = ctx.ecs;
            if (imgui.igBeginChild_Str("HierarchyScroll", .{ .x = 0, .y = 0 }, imgui.ImGuiChildFlags_None, imgui.ImGuiWindowFlags_None)) {

                // Find roots (transform.parent == null).  Entities without a
                // TransformComponent are also shown at root level.
                var seen = std.AutoHashMap(usize, void).init(self.allocator);
                defer seen.deinit();

                // First: all entities that _have_ a transform and no parent.
                var transform_it = ecs.transform_components.iterator();
                while (transform_it.next()) |entry| {
                    if (entry.component.parent == null) {
                        drawHierarchyRecursive(ecs, entry.entity_id, &seen, self);
                    }
                }
                // Second: entities that lack a transform altogether (e.g. globals).
                var entity_it = ecs.world.entities.iterator();
                while (entity_it.next()) |eid_entry| {
                    const eid = eid_entry.value_ptr.*;
                    if (!ecs.transform_components.has(eid) and !seen.contains(eid.id)) {
                        _ = drawHierarchyItem(eid, self, .{});
                    }
                }
            }
            imgui.igEndChild();

            _ = imgui.igTableSetColumnIndex(1);

            // ───────── RIGHT SIDE : component details ─────────
            if (imgui.igBeginChild_Str("DetailsScroll", .{ .x = 0, .y = 0 }, imgui.ImGuiChildFlags_None, imgui.ImGuiWindowFlags_None)) {
                if (self.selected_id) |eid| {
                    showAllComponents(ecs, eid);
                } else {
                    imgui.igTextDisabled("Select an entity to view its components");
                }
            }
            imgui.igEndChild();

            imgui.igEndTable();
        }

        imgui.igEnd(); // Entities window
    }

    pub fn drawSidebar(self: *Self, ctx: *const @import("UI.zig").UIContext) void {
        if (!self.visible) return;

        imgui.igText("Entities"); // header
        imgui.igSeparator();

        // Global controls
        if (imgui.igTreeNode_Str("Global Controls")) {
            // Collision debug toggle
            const old_collision_state = self.show_collision_debug;
            if (imgui.igCheckbox("Show Collision Debug", &self.show_collision_debug)) {
                // State changed, toggle collision debug visualization
                ctx.ecs.collision_system.setDebugWireframes(self.show_collision_debug) catch |err| {
                    std.debug.print("Failed to toggle collision debug: {}\n", .{err});
                    self.show_collision_debug = old_collision_state; // Revert on error
                };
            }
            if (imgui.igIsItemHovered(imgui.ImGuiHoveredFlags_None)) {
                imgui.igSetTooltip("Show wireframe boxes around colliders.\nGreen = Dynamic, Blue = Static");
            }

            // Global renderable visibility toggle
            if (imgui.igCheckbox("Global Renderable Visibility", &self.global_renderable_visibility)) {
                // Toggle visibility for all renderables
                var it = ctx.ecs.renderer_components.iterator();
                while (it.next()) |entry| {
                    entry.component.is_visible = self.global_renderable_visibility;
                }
            }
            if (imgui.igIsItemHovered(imgui.ImGuiHoveredFlags_None)) {
                imgui.igSetTooltip("Toggle visibility of all renderable objects in the scene");
            }

            imgui.igTreePop();
        }
        imgui.igSeparator();

        // Two-column layout: hierarchy | details
        if (imgui.igBeginTable("SidebarEntitiesTable", 2, imgui.ImGuiTableFlags_Resizable | imgui.ImGuiTableFlags_BordersInnerV, .{ .x = 0, .y = 0 }, 0)) {
            imgui.igTableSetupColumn("Hierarchy", imgui.ImGuiTableColumnFlags_WidthFixed, 180.0, 0);
            imgui.igTableSetupColumn("Details", imgui.ImGuiTableColumnFlags_WidthStretch, 0.0, 0);

            imgui.igTableNextRow(imgui.ImGuiTableRowFlags_None, 0.0);
            _ = imgui.igTableSetColumnIndex(0);

            // ---------- hierarchy ----------
            const ecs = ctx.ecs;
            if (imgui.igBeginChild_Str("SidebarHierarchyScroll", .{ .x = 0, .y = 0 }, imgui.ImGuiChildFlags_None, imgui.ImGuiWindowFlags_None)) {
                var seen = std.AutoHashMap(usize, void).init(self.allocator);
                defer seen.deinit();

                // roots first
                var tf_it = ecs.transform_components.iterator();
                while (tf_it.next()) |e| {
                    if (e.component.parent == null) {
                        drawHierarchyRecursive(ecs, e.entity_id, &seen, self);
                    }
                }
                // loose entities
                var ent_it = ecs.world.entities.iterator();
                while (ent_it.next()) |eid_entry| {
                    const eid = eid_entry.value_ptr.*;
                    if (!ecs.transform_components.has(eid) and !seen.contains(eid.id)) {
                        _ = drawHierarchyItem(eid, self, .{});
                    }
                }
            }
            imgui.igEndChild();

            _ = imgui.igTableSetColumnIndex(1);

            // ---------- component details ----------
            if (imgui.igBeginChild_Str("SidebarDetailsScroll", .{ .x = 0, .y = 0 }, imgui.ImGuiChildFlags_None, imgui.ImGuiWindowFlags_None)) {
                if (self.selected_id) |eid| {
                    showAllComponents(ecs, eid);
                } else {
                    imgui.igTextDisabled("Select an entity to view its components");
                }
            }
            imgui.igEndChild();

            imgui.igEndTable();
        }
    }

    // ───── helpers ───────────────────────────────────────────────────────
    fn drawHierarchyRecursive(
        ecs: *ECSManager,
        eid: Core.EntityID,
        seen: *std.AutoHashMap(usize, void),
        self: *Self,
    ) void {
        seen.put(eid.id, {}) catch {};

        const opened = drawHierarchyItem(eid, self, .{});
        if (!opened) return;

        if (ecs.transform_components.get(eid)) |tf| {
            for (tf.children.items) |child_eid| {
                drawHierarchyRecursive(ecs, child_eid, seen, self);
            }
        }
        imgui.igTreePop();
    }

    const HierItemOpts = struct { name: []const u8 = "" };

    fn drawHierarchyItem(
        eid: Core.EntityID,
        self: *Self,
        opts: HierItemOpts,
    ) bool {
        // unique label to avoid ImGui id collisions
        var label_buf: [64]u8 = undefined;
        const label = switch (opts.name.len > 0) {
            true => std.fmt.bufPrintZ(&label_buf, "{s}##{d}", .{ opts.name, eid.id }) catch unreachable,
            false => std.fmt.bufPrintZ(&label_buf, "Entity {d}", .{eid.id}) catch unreachable,
        };

        const is_selected = self.selected_id != null and self.selected_id.?.id == eid.id;

        const flags: imgui.ImGuiTreeNodeFlags = imgui.ImGuiTreeNodeFlags_SpanAvailWidth | imgui.ImGuiTreeNodeFlags_OpenOnArrow | (if (is_selected) imgui.ImGuiTreeNodeFlags_Selected else 0);

        const opened = imgui.igTreeNodeEx_Str(label.ptr, flags);
        if (imgui.igIsItemClicked(imgui.ImGuiMouseButton_Left)) {
            self.selected_id = eid;
        }

        return opened;
    }

    // ───── component-detail section ──────────────────────────────────────
    fn showAllComponents(ecs: *ECSManager, eid: Core.EntityID) void {
        // Show entity controls directly (not in a tree node)
        showEntityControls(ecs, eid);
        imgui.igSeparator();

        showTransform(ecs, eid);
        showRenderable(ecs, eid);
        showPhysics(ecs, eid);
        showCamera(ecs, eid);
        showController(ecs, eid);
        showViewport(ecs, eid);
    }

    fn showEntityControls(ecs: *ECSManager, eid: Core.EntityID) void {
        // Debug wireframe toggle (only for entities with physics components)
        if (ecs.physics_components.has(eid)) {
            var debug_enabled = false; // TODO: Get actual debug state for this entity
            if (imgui.igCheckbox("Debug Wireframe", &debug_enabled)) {
                // TODO: Toggle debug wireframe for this specific entity
                std.debug.print("TODO: Toggle debug wireframe for entity {d}\n", .{eid.id});
            }
            if (imgui.igIsItemHovered(imgui.ImGuiHoveredFlags_None)) {
                imgui.igSetTooltip("Toggle debug wireframe for this entity and its children");
            }
        }

        // Visibility toggle - always show for all entities
        var is_visible = true;
        if (ecs.renderer_components.get(eid)) |renderable| {
            // Entity has renderable - use its visibility state
            is_visible = renderable.is_visible;
        } else {
            // No renderable - check if ALL children with renderables are visible
            // If any child is hidden, we show unchecked
            is_visible = areAllChildrenVisible(ecs, eid);
        }
        
        if (imgui.igCheckbox("Visible", &is_visible)) {
            // Toggle visibility for this entity and recursively for children
            toggleRenderableRecursive(ecs, eid, is_visible);
        }
        if (imgui.igIsItemHovered(imgui.ImGuiHoveredFlags_None)) {
            imgui.igSetTooltip("Toggle visibility for this entity and all its children");
        }
    }

    fn toggleRenderableRecursive(ecs: *ECSManager, eid: Core.EntityID, visible: bool) void {
        // Toggle this entity's renderable if it has one
        if (ecs.renderer_components.get(eid)) |renderable| {
            renderable.is_visible = visible;
        }

        // Recursively toggle children
        if (ecs.transform_components.get(eid)) |transform| {
            for (transform.children.items) |child_eid| {
                toggleRenderableRecursive(ecs, child_eid, visible);
            }
        }
    }
    
    fn areAllChildrenVisible(ecs: *ECSManager, eid: Core.EntityID) bool {
        // Check if ALL children with renderables are visible
        // Returns true if no renderables found or all are visible
        if (ecs.transform_components.get(eid)) |transform| {
            for (transform.children.items) |child_eid| {
                // Check this child
                if (ecs.renderer_components.get(child_eid)) |renderable| {
                    if (!renderable.is_visible) return false;
                }
                // Recursively check grandchildren
                if (!areAllChildrenVisible(ecs, child_eid)) return false;
            }
        }
        return true; // All renderables are visible (or no renderables found)
    }

    inline fn bulletMat4(label: [:0]const u8, m: Math.Mat4) void {
        if (imgui.igTreeNode_Str(label.ptr)) {
            imgui.igText("*Column-major order (each row in Mat4 represents a column)");
            for (0..4) |r| {
                imgui.igBulletText("[%.2f  %.2f  %.2f  %.2f]", m.base.data[0 * 4 + r], m.base.data[1 * 4 + r], m.base.data[2 * 4 + r], m.base.data[3 * 4 + r]);
            }
            imgui.igTreePop();
        }
    }

    fn showTransform(ecs: *ECSManager, eid: Core.EntityID) void {
        if (!ecs.transform_components.has(eid)) return;
        if (imgui.igTreeNode_Str("Transform")) {
            const t = ecs.transform_components.get(eid).?;
            imgui.igBulletText("pos : (%.2f, %.2f, %.2f)", t.position[0], t.position[1], t.position[2]);
            const q = t.rotation;
            imgui.igBulletText("rot : (%.2f, %.2f, %.2f, %.2f)", q.x(), q.y(), q.z(), q.w());
            imgui.igBulletText("scale: (%.2f, %.2f, %.2f)", t.scale[0], t.scale[1], t.scale[2]);

            bulletMat4("Local Matrix", t.local_transform);
            bulletMat4("World Matrix", t.world_transform);
            imgui.igTreePop();
        }
    }
    fn showRenderable(ecs: *ECSManager, eid: Core.EntityID) void {
        if (!ecs.renderer_components.has(eid)) return;
        if (imgui.igTreeNode_Str("Renderable")) {
            const r = ecs.renderer_components.get(eid).?;
            imgui.igBulletText("mesh    : %.*s", @as(c_int, @intCast(r.mesh_name.len)), r.mesh_name.ptr);
            imgui.igBulletText("material: %s", if (r.material_name) |m|
                @as([*:0]const u8, @ptrCast(m.ptr)) // sentinel already present
            else
                "null");
            imgui.igBulletText("visible : %s", cstr(if (r.is_visible) "true" else "false"));
            imgui.igTreePop();
        }
    }
    fn showPhysics(ecs: *ECSManager, eid: Core.EntityID) void {
        if (!ecs.physics_components.has(eid)) return;
        if (imgui.igTreeNode_Str("Physics")) {
            const p = ecs.physics_components.get(eid).?;
            imgui.igBulletText("body_type: %s", cstr(@tagName(p.body_type)));
            imgui.igBulletText("mass     : %.3f", p.mass);
            imgui.igTreePop();
        }
    }
    fn showCamera(ecs: *ECSManager, eid: Core.EntityID) void {
        if (!ecs.camera_components.has(eid)) return;
        if (imgui.igTreeNode_Str("Camera")) {
            const cam = ecs.camera_components.get(eid).?;
            imgui.igBulletText("active: %s", cstr(if (cam.active) "true" else "false"));
            imgui.igBulletText("fov/near/far: %.1f / %.2f / %.1f", cam.fov, cam.near_plane, cam.far_plane);
            imgui.igTreePop();
        }
    }
    fn showController(ecs: *ECSManager, eid: Core.EntityID) void {
        if (!ecs.controller_components.has(eid)) return;
        if (imgui.igTreeNode_Str("Controller")) {
            const c = ecs.controller_components.get(eid).?;
            imgui.igBulletText("move_speed: %.2f", c.move_speed);
            imgui.igBulletText("# bindings: %d", c.key_bindings.items.len);
            imgui.igTreePop();
        }
    }
    fn showViewport(ecs: *ECSManager, eid: Core.EntityID) void {
        if (!ecs.viewport_components.has(eid)) return;
        if (imgui.igTreeNode_Str("Viewport")) {
            const vp = ecs.viewport_components.get(eid).?.vp;
            imgui.igBulletText("name : %.*s", @as(c_int, @intCast(vp.name.len)), vp.name.ptr);
            imgui.igBulletText("size : %dx%d", vp.fbo.width, vp.fbo.height);
            imgui.igBulletText("enabled: %s", cstr(if (vp.enabled) "true" else "false"));
            imgui.igTreePop();
        }
    }
};
