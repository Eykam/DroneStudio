const std = @import("std");
const Camera = @import("Camera.zig");
const gl = @import("../../bindings/gl.zig");
const c = @import("../../bindings/c.zig");
const Core = @import("../Core.zig");
const Viewports = @import("Viewports.zig");
const Controller = @import("Controller.zig");
const Transform = @import("Transform.zig");

const glfw = gl.glfw;
const glad = gl.glad;
const imgui = c.imgui;
const CameraSystem = Camera.CameraSystem;
const ControlSystem = Controller.ControlSystem;
const ViewportSystem = Viewports.ViewportSystem;
const TransformSystem = Transform.TransformSystem;

pub const GlobalsComponent = struct {
    last_frame_time: f64 = 0,
    dt: f64 = 0,
    last_fps_time: f64 = 0,
    frame_count: u32 = 0,
    avg_fps: f32 = 0,

    scene_width: u32 = 0,
    scene_height: u32 = 0,

    last_mouse_x: f64 = 0.0,
    last_mouse_y: f64 = 0.0,
    mouse_dx: f64 = 0.0,
    mouse_dy: f64 = 0.0,

    first_mouse: bool = true,
    zoom: f32 = 90.0,
    keys: [1024]bool = .{false} ** 1024,

    paused: bool = false,
    fly: bool = false,
    menu: bool = false,
    reset_requested: bool = false,

    first_person_view: bool = true,
};

pub const GlobalsSystem = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    globals: GlobalsComponent,
    camera_system: *CameraSystem,
    control_system: *ControlSystem,
    viewport_system: *ViewportSystem,
    transform_system: *TransformSystem,
    window: ?*glfw.GLFWwindow,

    pub fn init(allocator: std.mem.Allocator, globals: GlobalsComponent) !*Self {
        if (glfw.glfwInit() == 0) {
            std.debug.print("Failed to initialize GLFW\n", .{});
            return error.FailedToInitializeGLFW;
        }

        const window = try createWindow();

        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .globals = globals,
            .camera_system = undefined,
            .control_system = undefined,
            .viewport_system = undefined,
            .transform_system = undefined,
            .window = window,
        };

        var width: i32 = undefined;
        var height: i32 = undefined;
        glfw.glfwGetWindowSize(window, &width, &height);

        self.globals.scene_width = @intCast(width);
        self.globals.scene_height = @intCast(height);
        self.setupCallbacks(window);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.cleanupImGui();
    }

    pub fn setupCallbacks(self: *Self, window: ?*glfw.struct_GLFWwindow) void {
        if (window == null) return;

        glfw.glfwSetWindowUserPointer(window, @ptrCast(self));

        const ptr = glfw.glfwGetWindowUserPointer(window);
        if (ptr == null) {
            std.debug.print("Failed to set window user pointer\n", .{});
            return;
        }

        _ = glfw.glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
        _ = glfw.glfwSetCursorPosCallback(window, mouseCallback);
        _ = glfw.glfwSetCharCallback(window, charCallback);
        _ = glfw.glfwSetKeyCallback(window, keyCallback);
        _ = glfw.glfwSetScrollCallback(window, scrollCallback);
        _ = glfw.glfwSetMouseButtonCallback(window, mouseButtonCallback);
    }

    const GSLWError = error{ FailedToCreateWindow, FailedToInitialize };

    pub fn createWindow() !?*glfw.GLFWwindow {
        glfw.glfwWindowHint(glfw.GLFW_FOCUSED, glfw.GLFW_TRUE);
        glfw.glfwWindowHint(glfw.GLFW_FOCUS_ON_SHOW, glfw.GLFW_TRUE);
        glfw.glfwWindowHint(glfw.GLFW_CLIENT_API, glfw.GLFW_OPENGL_API);
        glfw.glfwWindowHint(glfw.GLFW_CONTEXT_CREATION_API, glfw.GLFW_NATIVE_CONTEXT_API);
        glfw.glfwWindowHint(glfw.GLFW_DOUBLEBUFFER, glfw.GLFW_TRUE);
        glfw.glfwWindowHint(glfw.GLFW_SRGB_CAPABLE, glfw.GLFW_TRUE);
        glfw.glfwWindowHint(glfw.GLFW_DECORATED, glfw.GLFW_TRUE); // Remove window decorations
        glfw.glfwWindowHint(glfw.GLFW_MAXIMIZED, glfw.GLFW_TRUE); // Start maximized

        const monitor = glfw.glfwGetPrimaryMonitor();
        const video_mode = glfw.glfwGetVideoMode(monitor);

        // Create window with monitor's resolution
        const width = video_mode.*.width;
        const height = video_mode.*.height;

        const window = glfw.glfwCreateWindow(width, height, "Drone Studio", null, null) orelse return null;

        // Position the window at the monitor's position
        var xpos: i32 = undefined;
        var ypos: i32 = undefined;
        glfw.glfwGetMonitorPos(monitor, &xpos, &ypos);
        glfw.glfwSetWindowPos(window, xpos, ypos);

        // Rest of your initialization code...
        glfw.glfwMakeContextCurrent(window);
        glfw.glfwSwapInterval(0);

        // Initialize OpenGL Loader
        std.debug.print("Loading Glad...\n", .{});
        if (glad.gladLoadGLLoader(@ptrCast(&glfw.glfwGetProcAddress)) == 0) {
            std.debug.print("Failed to initialize GLAD\n", .{});
            return GSLWError.FailedToInitialize;
        }

        std.debug.print("Initializing viewport...\n\n", .{});
        glad.glViewport(0, 0, width, height);
        glad.glEnable(glad.GL_DEPTH_TEST);
        glad.glDepthFunc(glad.GL_LESS);

        // Initialize ImGui
        _ = imgui.igCreateContext(null);
        const io = imgui.igGetIO();
        io.*.ConfigFlags |= imgui.ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
        io.*.ConfigFlags |= imgui.ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls

        const optional_window: ?*imgui.GLFWwindow = @ptrCast(window);
        if (!imgui.ImGui_ImplGlfw_InitForOpenGL(optional_window, false)) {
            return null;
        }
        if (!imgui.ImGui_ImplOpenGL3_Init("#version 330")) {
            return null;
        }

        // Set up cursor mode BEFORE changing monitor settings
        glfw.glfwSetInputMode(window, glfw.GLFW_CURSOR, glfw.GLFW_CURSOR_DISABLED);
        if (glfw.glfwRawMouseMotionSupported() == glfw.GLFW_TRUE) {
            glfw.glfwSetInputMode(window, glfw.GLFW_RAW_MOUSE_MOTION, glfw.GLFW_TRUE);
        }

        // Force focus and raise window
        glfw.glfwFocusWindow(window);
        glfw.glfwShowWindow(window);

        return window;
    }

    pub fn cleanupImGui() void {
        imgui.ImGui_ImplOpenGL3_Shutdown();
        imgui.ImGui_ImplGlfw_Shutdown();
        imgui.igDestroyContext(null);
    }

    fn framebufferSizeCallback(window: ?*glfw.GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
        if (window == null) return;

        const user_ptr = glfw.glfwGetWindowUserPointer(window);
        if (user_ptr == null) {
            std.debug.print("Error: Window user pointer is null in framebufferSizeCallback\n", .{});
            return;
        }

        const scene = @as(*Self, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

        glad.glViewport(0, 0, width, height);

        // Update Scene's width and height
        scene.globals.scene_width = @intCast(width);
        scene.globals.scene_height = @intCast(height);
    }

    fn mouseCallback(window: ?*glfw.struct_GLFWwindow, xpos: f64, ypos: f64) callconv(.C) void {
        if (window == null) return;

        // Only process mouse input if window is focused
        if (glfw.glfwGetWindowAttrib(window, glfw.GLFW_FOCUSED) != glfw.GLFW_TRUE) {
            return;
        }

        const scene = @as(*Self, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));
        const io = imgui.igGetIO();

        if (scene.globals.first_mouse) {
            scene.globals.last_mouse_x = xpos;
            scene.globals.last_mouse_y = ypos;
            scene.globals.first_mouse = false;
            return;
        }

        scene.globals.mouse_dx = xpos - scene.globals.last_mouse_x;
        scene.globals.mouse_dy = scene.globals.last_mouse_y - ypos; // Reversed Y

        scene.globals.last_mouse_x = xpos;
        scene.globals.last_mouse_y = ypos;

        if (scene.globals.menu) {
            if (io.*.WantCaptureMouse) {
                // Let ImGui handle the mouse if it wants it
                scene.globals.last_mouse_x = xpos;
                scene.globals.last_mouse_y = ypos;
                return;
            }
        }
    }

    // New method to handle drone mouse movement for yaw and pitch
    fn mouseButtonCallback(window: ?*glfw.struct_GLFWwindow, button: c_int, action: c_int, mods: c_int) callconv(.C) void {
        if (window == null) return;

        const scene = @as(*Self, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

        if (scene.globals.menu) {
            imgui.ImGui_ImplGlfw_MouseButtonCallback(@ptrCast(window), button, action, mods);

            const io = imgui.igGetIO();
            if (io.*.WantCaptureMouse) {
                return;
            }
        }
    }

    fn charCallback(window: ?*glfw.struct_GLFWwindow, character: c_uint) callconv(.C) void {
        if (window == null) return;

        const scene = @as(*Self, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

        if (scene.globals.menu) {
            imgui.ImGui_ImplGlfw_CharCallback(@ptrCast(window), character);
        }
    }

    fn keyCallback(window: ?*glfw.struct_GLFWwindow, key: c_int, scancode: c_int, action: c_int, mods: c_int) callconv(.C) void {
        if (window == null) return;

        const scene = @as(*Self, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));
        // const io = imgui.igGetIO();
        if (scene.globals.menu) {
            imgui.ImGui_ImplGlfw_KeyCallback(@ptrCast(window), key, scancode, action, mods);
        }

        if (key < 0 or key >= 1024) return;

        if (action == glfw.GLFW_PRESS)
            scene.globals.keys[@intCast(key)] = true
        else if (action == glfw.GLFW_RELEASE)
            scene.globals.keys[@intCast(key)] = false;

        if (action == glfw.GLFW_PRESS or action == glfw.GLFW_REPEAT) {
            switch (key) {
                glfw.GLFW_KEY_ESCAPE => {
                    scene.globals.menu = !scene.globals.menu;
                    const current_mode = glfw.glfwGetInputMode(window, glfw.GLFW_CURSOR);

                    const cursor_mode = switch (current_mode) {
                        glfw.GLFW_CURSOR_NORMAL => glfw.GLFW_CURSOR_DISABLED,
                        glfw.GLFW_CURSOR_HIDDEN => glfw.GLFW_CURSOR_NORMAL,
                        glfw.GLFW_CURSOR_DISABLED => glfw.GLFW_CURSOR_NORMAL,
                        else => unreachable,
                    };

                    glfw.glfwSetInputMode(window, glfw.GLFW_CURSOR, cursor_mode);
                },
                glfw.GLFW_KEY_M => {
                    glfw.glfwIconifyWindow(window);
                },
                glfw.GLFW_KEY_RIGHT_BRACKET => {
                    glfw.glfwDestroyWindow(window);
                },
                glfw.GLFW_KEY_P => {
                    scene.globals.paused = !scene.globals.paused;
                },
                glfw.GLFW_KEY_F => {
                    scene.globals.fly = !scene.globals.fly;
                },

                glfw.GLFW_KEY_V => {
                    scene.toggleViewport();
                },
                glfw.GLFW_KEY_R => {
                    scene.globals.reset_requested = true;
                    std.debug.print("Reset requested!\n", .{});
                },

                else => {},
            }
        }
    }

    /// Cycle through enabled viewports and switch both the active camera
    /// and the active controller.
    pub fn toggleViewport(self: *Self) void {
        const vps = self.viewport_system.viewports;
        const cam_sys = self.camera_system;
        const ctrl_sys = self.control_system;

        var target_cam_eid: Core.EntityID = blk: {
            // No camera yet → just pick the first viewport entity
            if (cam_sys.active_camera_eid) |eid| break :blk eid;

            // first() is safe because at least one viewport exists
            var first = vps.iterator();
            const first_cam = first.next().?.entity_id;
            break :blk first_cam;
        };

        if (cam_sys.active_camera_eid) |_| {
            var it = vps.iterator();
            var first_seen: ?Core.EntityID = null;
            var choose_next = false;

            while (it.next()) |entry| {
                if (first_seen == null) first_seen = entry.entity_id;

                if (choose_next) {
                    target_cam_eid = entry.entity_id;
                    break;
                }
                if (entry.entity_id.id == cam_sys.active_camera_eid.?.id)
                    choose_next = true;
            }
            if (choose_next and target_cam_eid.id == cam_sys.active_camera_eid.?.id)
                target_cam_eid = first_seen.?; // wrapped around
        }

        cam_sys.set_active(target_cam_eid);

        const ctrl_eid: ?Core.EntityID = blk: {
            var current = target_cam_eid;
            while (true) {
                if (self.control_system.controller_components.has(current))
                    break :blk current;

                const tf_opt = self.transform_system.transform_components.get(current);
                if (tf_opt == null or tf_opt.?.parent == null)
                    break :blk null;

                current = tf_opt.?.parent.?; // climb one level
            }
        };

        std.debug.print("New active ctrl: {any}\n", .{if (ctrl_eid) |eid| eid.id else null});
        ctrl_sys.active_controller_eid = ctrl_eid; // may be null → movement disabled
    }

    fn scrollCallback(window: ?*glfw.struct_GLFWwindow, xoffset: f64, yoffset: f64) callconv(.C) void {
        if (window == null) return;

        const scene = @as(*Self, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

        if (scene.globals.menu and imgui.igGetIO().*.WantCaptureMouse) {
            imgui.ImGui_ImplGlfw_ScrollCallback(@ptrCast(window), xoffset, yoffset);
            return;
        }

        const zoomSensitivity: f32 = 0.1;
        const newZoom = scene.globals.zoom - @as(f32, @floatCast(yoffset)) * zoomSensitivity * scene.globals.zoom;

        // Clamp the zoom level to prevent weird behavior at larger FOV's
        if (newZoom < 1.0) {
            scene.globals.zoom = 1.0;
        } else if (newZoom >= 120) {
            scene.globals.zoom = 120;
        } else {
            scene.globals.zoom = newZoom;
        }

        // if (scene.globals.camera_manager) |camera_manager| {
        //     camera_manager.main_camera.?.process_scroll_wheel(newZoom);
        // }
    }
};
