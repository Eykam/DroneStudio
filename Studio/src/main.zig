const std = @import("std");
const Transformations = @import("Transformations.zig");
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
    const width: i32 = @intFromFloat(1920 * 0.75);
    const height: i32 = @intFromFloat(1080 * 0.75);

    // Create window initially in windowed mode
    const window = c.glfwCreateWindow(width, height, "Drone Studio", null, null) orelse return null;

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

    const window = createWindow();
    defer c.glfwDestroyWindow(window.?);

    // Define transformation matrices
    // const eye = [3]f32{ 0.0, 0.0, 5.0 }; // Camera position
    // const center = [3]f32{ 0.0, 0.0, 0.0 }; // Look at origin
    // const up = [3]f32{ 0.0, 1.0, 0.0 }; // Up vector
    // const view = Transformations.lookAt(eye, center, up);

    //Initializing Scene
    var scene = try Scene.init(alloc, window);
    defer scene.deinit();

    scene.setupCallbacks(window);

    //Initializing Entities
    const grid = try Shape.Grid.init(alloc, 10, 10);
    const triangle = try Shape.Triangle.init(null);
    // const box = try Shape.Box.init();

    //Adding Entities to Scene
    try scene.addObject("grid", grid);
    try scene.addObject("triangle", triangle);
    // try scene.addObject("rectangle", box);

    //Debugging Entities
    grid.debug();
    triangle.debug();
    // box.debug();

    //Render loop
    while (c.glfwWindowShouldClose(window) == 0) {
        scene.render(window, null);
    }
}
