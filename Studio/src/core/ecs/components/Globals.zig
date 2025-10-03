const std = @import("std");
const Camera = @import("Camera.zig");
const gl = @import("../../bindings/gl.zig");
const c = @import("../../bindings/c.zig");
const Core = @import("../Core.zig");
const Viewports = @import("Viewports.zig");
const Controller = @import("Controller.zig");
const Transform = @import("Transform.zig");
const Collisions = @import("Collisions.zig");
const ECSManager = @import("../ECSManager.zig");

const glfw = gl.glfw;
const glad = gl.glad;
const imgui = c.imgui;
const CameraSystem = Camera.CameraSystem;
const ControlSystem = Controller.ControlSystem;
const ViewportSystem = Viewports.ViewportSystem;
const TransformSystem = Transform.TransformSystem;
const CollisionSystem = Collisions.CollisionSystem;

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
    collision_system: ?*CollisionSystem = null, // Optional reference to collision system
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
        cleanupImGui();

        // Destroy window and terminate GLFW
        if (self.window) |window| {
            glfw.glfwDestroyWindow(window);
        }
        glfw.glfwTerminate();

        // Free self
        self.allocator.destroy(self);
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
        // glfw.glfwSetInputMode(window, glfw.GLFW_CURSOR, glfw.GLFW_CURSOR_NORMAL);
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

        const dx = xpos - scene.globals.last_mouse_x;
        const dy = scene.globals.last_mouse_y - ypos; // Reversed Y

        scene.globals.mouse_dx = dx;
        scene.globals.mouse_dy = dy;
        scene.globals.last_mouse_x = xpos;
        scene.globals.last_mouse_y = ypos;

        if (scene.globals.menu) {
            if (io.*.WantCaptureMouse) {
                return;
            }
        }

        // Forward mouse movement to control system
        if (dx != 0 or dy != 0) {
            var event = Controller.Event{ .mouse = Controller.MouseEvent{
                .x = @floatCast(xpos),
                .y = @floatCast(ypos),
                .dx = @floatCast(dx),
                .dy = @floatCast(dy),
                .dt = @floatCast(scene.globals.dt),
            } };

            scene.control_system.handleEvent(&event);
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

        // Forward to control system
        const mouse_button = switch (button) {
            0 => Controller.MouseButton.left,
            1 => Controller.MouseButton.right,
            2 => Controller.MouseButton.middle,
            else => return,
        };

        const mouse_action = switch (action) {
            0 => Controller.MouseAction.release,
            1 => Controller.MouseAction.press,
            else => return,
        };

        var event = Controller.Event{ .mouse = Controller.MouseEvent{
            .x = @floatCast(scene.globals.last_mouse_x),
            .y = @floatCast(scene.globals.last_mouse_y),
            .dx = 0,
            .dy = 0,
            .button = mouse_button,
            .action = mouse_action,
            .mods = mods,
            .dt = @floatCast(scene.globals.dt),
        } };

        scene.control_system.handleEvent(&event);
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

        if (scene.globals.menu) {
            imgui.ImGui_ImplGlfw_KeyCallback(@ptrCast(window), key, scancode, action, mods);
            const io = imgui.igGetIO();
            if (io.*.WantCaptureKeyboard and key != glfw.GLFW_KEY_ESCAPE) {
                return; // Let ESC through even when ImGui wants keyboard
            }
        }

        if (key < 0 or key >= 1024) return;

        // Update global key state for legacy code
        if (action == glfw.GLFW_PRESS)
            scene.control_system.updateKeyState(key, true)
        else if (action == glfw.GLFW_RELEASE)
            scene.control_system.updateKeyState(key, false);

        // Forward to control system
        var event = Controller.Event{ .key = Controller.InputEvent{
            .key = Controller.Key.fromGLFW(key),
            .scancode = scancode,
            .action = @enumFromInt(action),
            .mods = mods,
            .dt = @floatCast(scene.globals.dt),
        } };

        scene.control_system.handleEvent(&event);
    }

    fn scrollCallback(window: ?*glfw.struct_GLFWwindow, xoffset: f64, yoffset: f64) callconv(.C) void {
        if (window == null) return;

        const scene = @as(*Self, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

        if (scene.globals.menu and imgui.igGetIO().*.WantCaptureMouse) {
            imgui.ImGui_ImplGlfw_ScrollCallback(@ptrCast(window), xoffset, yoffset);
            return;
        }

        // Forward to control system
        var event = Controller.Event{ .mouse = Controller.MouseEvent{
            .x = @floatCast(scene.globals.last_mouse_x),
            .y = @floatCast(scene.globals.last_mouse_y),
            .dx = 0,
            .dy = 0,
            .scroll_x = @floatCast(xoffset),
            .scroll_y = @floatCast(yoffset),
            .dt = @floatCast(scene.globals.dt),
        } };

        scene.control_system.handleEvent(&event);

        // Legacy zoom handling - should be moved to a controller
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

// Global application controller
pub const GlobalController = struct {
    pub fn createComponent() Controller.ControllerComponent {
        var controller = Controller.ControllerComponent.init(0, "Global", .Application);

        // ESC - Toggle menu
        controller.addBinding(.{
            .key = .Escape,
            .handler = handleMenuToggle,
            .context = null, // Will be set to ECSManager when processing
        }) catch unreachable;

        // M - Minimize window
        controller.addBinding(.{
            .key = .M,
            .handler = handleMinimize,
            .context = null,
        }) catch unreachable;

        // ] - Quit application
        controller.addBinding(.{
            .key = .RightBracket,
            .handler = handleQuit,
            .context = null,
        }) catch unreachable;

        return controller;
    }

    fn handleMenuToggle(event: *Controller.InputEvent, context: ?*anyopaque) void {
        if (event.action != .Press and event.action != .Repeat) return; // Only on PRESS or REPEAT

        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        const globals_system = ecs.globals_system;

        globals_system.globals.menu = !globals_system.globals.menu;

        if (globals_system.window) |window| {
            const current_mode = glfw.glfwGetInputMode(window, glfw.GLFW_CURSOR);
            const cursor_mode = switch (current_mode) {
                glfw.GLFW_CURSOR_NORMAL => glfw.GLFW_CURSOR_DISABLED,
                glfw.GLFW_CURSOR_HIDDEN => glfw.GLFW_CURSOR_NORMAL,
                glfw.GLFW_CURSOR_DISABLED => glfw.GLFW_CURSOR_NORMAL,
                else => glfw.GLFW_CURSOR_NORMAL,
            };

            glfw.glfwSetInputMode(window, glfw.GLFW_CURSOR, cursor_mode);
        }
        event.consume();
    }

    fn handleMinimize(event: *Controller.InputEvent, context: ?*anyopaque) void {
        if (event.action != .Press) return; // Only on PRESS

        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        if (ecs.globals_system.window) |window| {
            glfw.glfwIconifyWindow(window);
        }
        event.consume();
    }

    fn handleQuit(event: *Controller.InputEvent, context: ?*anyopaque) void {
        if (event.action != .Press) return; // Only on PRESS

        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        if (ecs.globals_system.window) |window| {
            glfw.glfwSetWindowShouldClose(window, glfw.GLFW_TRUE);
        }
        event.consume();
    }
};
