const std = @import("std");
const Transformations = @import("Transformations.zig");
const Vec3 = Transformations.Vec3;
const Pipeline = @import("Pipeline.zig");
const Scene = Pipeline.Scene;
const Shape = @import("Shape.zig");
const Node = @import("Node.zig");
const Mesh = @import("Mesh.zig");
const _Secrets = @import("Secrets.local.zig"); // replace Secrets.example.zig with Secrets.local.zig
const Secrets = _Secrets{};
const UDP = @import("UDP.zig");
const Sensors = @import("Sensors.zig");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("glad/glad.h"); // Include GLAD
    @cInclude("GLFW/glfw3.h");
});

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

    const window = Pipeline.createWindow() orelse {
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
    var gridNode = try Shape.Grid.init(alloc, 1000, 5);
    var axisNode = try Shape.Axis.init(alloc, Vec3{ .x = 0.0, .y = 0.5, .z = 0.0 }, 10.0);
    var triangleNode = try Shape.Triangle.init(alloc, Vec3{ .x = 0.0, .y = 1.0, .z = 10.0 }, null);
    var boxNode = try Shape.Box.init(alloc, null, null, null, null);

    //Adding Nodes to Environment (parent node)
    var environment = try Node.init(alloc, null);
    try environment.addChild(&gridNode);
    try environment.addChild(&axisNode);
    try environment.addChild(&triangleNode);
    try environment.addChild(&boxNode);

    //Adding environment to scene
    try scene.addNode("Environment", &environment);

    //Debugging Entities
    scene.getNodeNames();

    // grid.debug();
    // axis.debug();
    // triangle.debug();
    // box.debug();

    std.debug.print("\nIntial Camera Pos: {d}\n", .{[_]f32{
        scene.camera.position.x,
        scene.camera.position.y,
        scene.camera.position.z,
    }});

    //Initialize UDP server
    var server = UDP.init(
        Secrets.host_ip,
        Secrets.host_port,
        Secrets.client_ip,
        Secrets.client_port,
    );

    var pose_handler = UDP.Handler(Sensors.PoseHandler).init(Sensors.PoseHandler.init(&boxNode.mesh.?));
    const pose_interface = pose_handler.interface();

    try server.start(pose_interface);

    //Render loop
    while (c.glfwWindowShouldClose(window) == 0) {
        if (c.glfwGetWindowAttrib(window, c.GLFW_FOCUSED) == c.GLFW_FALSE) {
            continue; // Skip frame if window is not focused
        }

        const current_time = c.glfwGetTime();

        // Calculate delta time
        scene.appState.delta_time = @floatCast(current_time - scene.appState.last_frame_time);
        scene.appState.last_frame_time = current_time;

        scene.processInput(false);
        scene.render(window);
    }
}
