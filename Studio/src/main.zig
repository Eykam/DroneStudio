const std = @import("std");
const Transformations = @import("Transformations.zig");
const Vec3 = Transformations.Vec3;
const Pipeline = @import("Pipeline.zig");
const Scene = Pipeline.Scene;
const Shape = @import("Shape.zig");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("glad/glad.h"); // Include GLAD
    @cInclude("GLFW/glfw3.h");
});

fn getCurrentMonitor(window: ?*c.struct_GLFWwindow) ?*c.GLFWmonitor {
    if (window == null) return null;

    var monitor_count: i32 = undefined;
    const monitors = c.glfwGetMonitors(&monitor_count);
    if (monitors == null or monitor_count == 0) return null;

    // Get window position
    var win_x: i32 = undefined;
    var win_y: i32 = undefined;
    c.glfwGetWindowPos(window, &win_x, &win_y);

    // Get window size
    var win_width: i32 = undefined;
    var win_height: i32 = undefined;
    c.glfwGetWindowSize(window, &win_width, &win_height);

    // Find the monitor that contains the window center
    const center_x = win_x + @divTrunc(win_width, 2);
    const center_y = win_y + @divTrunc(win_height, 2);

    var i: usize = 0;
    while (i < @as(i32, @intCast(monitor_count))) : (i += 1) {
        const mon = monitors[i];
        var mx: i32 = undefined;
        var my: i32 = undefined;
        var mw: i32 = undefined;
        var mh: i32 = undefined;
        c.glfwGetMonitorWorkarea(mon, &mx, &my, &mw, &mh);

        if (center_x >= mx and center_x < mx + mw and
            center_y >= my and center_y < my + mh)
        {
            return mon;
        }
    }

    // Default to primary monitor if no match found
    return c.glfwGetPrimaryMonitor();
}

// Window creation with proper monitor handling
pub fn createWindow() ?*c.GLFWwindow {
    // Set window hints before creation
    c.glfwWindowHint(c.GLFW_FOCUSED, c.GLFW_TRUE);
    c.glfwWindowHint(c.GLFW_FOCUS_ON_SHOW, c.GLFW_TRUE);
    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_OPENGL_API);
    c.glfwWindowHint(c.GLFW_CONTEXT_CREATION_API, c.GLFW_NATIVE_CONTEXT_API);

    const width: i32 = @intFromFloat(1920 * 0.75);
    const height: i32 = @intFromFloat(1080 * 0.75);

    const window = c.glfwCreateWindow(width, height, "Drone Studio", null, null) orelse return null;

    // Make context current immediately after window creation
    c.glfwMakeContextCurrent(window);

    // Set up cursor mode BEFORE changing monitor settings
    c.glfwSetInputMode(window, c.GLFW_CURSOR, c.GLFW_CURSOR_DISABLED);
    if (c.glfwRawMouseMotionSupported() == c.GLFW_TRUE) {
        c.glfwSetInputMode(window, c.GLFW_RAW_MOUSE_MOTION, c.GLFW_TRUE);
    }

    // Get the monitor the window should be on
    const monitor = getCurrentMonitor(window);
    if (monitor != null) {
        // Get monitor position and video mode
        var x: i32 = undefined;
        var y: i32 = undefined;
        c.glfwGetMonitorPos(monitor, &x, &y);

        const video_mode = c.glfwGetVideoMode(monitor);
        if (video_mode != null) {
            // Correctly dereference the video mode pointer
            const mode_width = video_mode.*.width;
            const mode_height = video_mode.*.height;
            const mode_refresh = video_mode.*.refreshRate;

            c.glfwSetWindowMonitor(window, monitor, 0, 0, mode_width, mode_height, mode_refresh);
        }
    }

    // Force focus and raise window (helpful for WSL)
    c.glfwFocusWindow(window);
    c.glfwShowWindow(window);

    return window;
}

fn processInput(scene: *Scene, debug: bool) void {
    var camera_speed_base: f32 = 2.5; // Units per second
    if (scene.appState.keys[@as(usize, c.GLFW_KEY_LEFT_SHIFT)]) {
        camera_speed_base = 5.0;
    }

    const camera_speed = camera_speed_base * scene.appState.delta_time;

    // Forward
    if (scene.appState.keys[@as(usize, c.GLFW_KEY_W)]) {
        const movement = scene.appState.camera_front.scale(camera_speed);
        const newCoords = Vec3.add(scene.appState.camera_pos, Vec3{
            .x = movement.x,
            .y = 0.0,
            .z = movement.z,
        });

        if (debug) {
            std.debug.print("==============\nForward\n", .{});
            std.debug.print("Initial: {d:.6}\n", .{[_]f32{
                scene.appState.camera_pos.x,
                scene.appState.camera_pos.y,
                scene.appState.camera_pos.z,
            }});
            std.debug.print("Offset: {d:.6}\n", .{[_]f32{
                movement.x,
                0,
                movement.z,
            }});
            std.debug.print("New: {d:.6}\n", .{[_]f32{
                newCoords.x,
                newCoords.y,
                newCoords.z,
            }});
        }

        scene.appState.camera_pos = newCoords;
    }

    // Left
    if (scene.appState.keys[@as(usize, c.GLFW_KEY_A)]) {
        const right = Vec3.cross(scene.appState.camera_front, scene.appState.camera_up).normalize();
        const movement = right.scale(-camera_speed);
        const newCoords = Vec3.add(scene.appState.camera_pos, Vec3{
            .x = movement.x,
            .y = 0.0,
            .z = movement.z,
        });

        if (debug) {
            std.debug.print("==============\nLeft\n", .{});
            std.debug.print("Initial: {d:.6}\n", .{[_]f32{
                scene.appState.camera_pos.x,
                scene.appState.camera_pos.y,
                scene.appState.camera_pos.z,
            }});
            std.debug.print("Offset: {d:.6}\n", .{[_]f32{
                movement.x,
                0.0,
                movement.z,
            }});
            std.debug.print("New: {d:.6}\n", .{[_]f32{
                newCoords.x,
                newCoords.y,
                newCoords.z,
            }});
        }

        scene.appState.camera_pos = newCoords;
    }

    // Backward
    if (scene.appState.keys[@as(usize, c.GLFW_KEY_S)]) {
        const movement = scene.appState.camera_front.scale(-camera_speed);
        const newCoords = Vec3.add(scene.appState.camera_pos, Vec3{
            .x = movement.x,
            .y = 0.0,
            .z = movement.z,
        });

        if (debug) {
            std.debug.print("\nBack\n", .{});
            std.debug.print("Initial: {d:.6}\n", .{[_]f32{
                scene.appState.camera_pos.x,
                scene.appState.camera_pos.y,
                scene.appState.camera_pos.z,
            }});
            std.debug.print("Offset: {d:.6}\n", .{[_]f32{
                movement.x,
                0.0,
                movement.z,
            }});
            std.debug.print("New: {d:.6}\n", .{[_]f32{
                newCoords.x,
                newCoords.y,
                newCoords.z,
            }});
        }

        scene.appState.camera_pos = newCoords;
    }

    // Right
    if (scene.appState.keys[@as(usize, c.GLFW_KEY_D)]) {
        const right = Vec3.cross(scene.appState.camera_front, scene.appState.camera_up).normalize();
        const movement = right.scale(camera_speed);
        const newCoords = Vec3.add(scene.appState.camera_pos, Vec3{
            .x = movement.x,
            .y = 0.0,
            .z = movement.z,
        });

        if (debug) {
            std.debug.print("==============\nRight\n", .{});
            std.debug.print("Initial: {d:.6}\n", .{[_]f32{
                scene.appState.camera_pos.x,
                scene.appState.camera_pos.y,
                scene.appState.camera_pos.z,
            }});
            std.debug.print("Offset: {d:.6}\n", .{[_]f32{
                movement.x,
                0.0,
                movement.z,
            }});
            std.debug.print("New: {d:.6}\n", .{[_]f32{
                newCoords.x,
                newCoords.y,
                newCoords.z,
            }});
        }

        scene.appState.camera_pos = newCoords;
    }

    // Zoom controls can remain in the keyCallback if they are discrete actions
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const alloc = arena.allocator();

    // Initialize GLFW
    if (c.glfwInit() == 0) {
        std.debug.print("Failed to initialize GLFW\n", .{});
        return;
    }

    defer c.glfwTerminate();

    const window = createWindow() orelse {
        std.debug.print("Failed to create window\n", .{});
        return;
    };
    defer c.glfwDestroyWindow(window);

    if (c.glfwGetInputMode(window, c.GLFW_CURSOR) != c.GLFW_CURSOR_DISABLED) {
        std.debug.print("Warning: Cursor not disabled as expected\n", .{});
    }

    //Initializing Scene
    var scene = try Scene.init(alloc, window);
    defer scene.deinit();

    scene.setupCallbacks(window);

    //Initializing Entities
    const grid = try Shape.Grid.init(alloc, 1000, 5);
    const triangle = try Shape.Triangle.init(null);
    const axis = try Shape.Axis.init(alloc, Vec3{ .x = 0.0, .y = 0.5, .z = 0.0 }, 10.0);
    // const box = try Shape.Box.init();

    //Adding Entities to Scene
    try scene.addObject("grid", grid);
    try scene.addObject("triangle", triangle);
    try scene.addObject("axis", axis);
    // try scene.addObject("rectangle", box);

    //Debugging Entities
    grid.debug();
    triangle.debug();
    axis.debug();
    // box.debug();

    std.debug.print("\nIntial Camera Pos: {d}\n", .{[_]f32{
        scene.appState.camera_pos.x,
        scene.appState.camera_pos.y,
        scene.appState.camera_pos.z,
    }});

    //Render loop
    while (c.glfwWindowShouldClose(window) == 0) {
        if (c.glfwGetWindowAttrib(window, c.GLFW_FOCUSED) == c.GLFW_FALSE) {
            continue; // Skip frame if window is not focused
        }

        const current_time = c.glfwGetTime();

        // Calculate delta time
        scene.appState.delta_time = @floatCast(current_time - scene.appState.last_frame_time);
        scene.appState.last_frame_time = current_time;

        processInput(&scene, false);
        scene.render(window);
    }
}
