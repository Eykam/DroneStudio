const std = @import("std");
const Math = @import("Math.zig");
const Vec3 = Math.Vec3;
const Pipeline = @import("Pipeline.zig");
const Scene = Pipeline.Scene;
const Shape = @import("Shape.zig");
const Node = @import("Node.zig");
const Mesh = @import("Mesh.zig");
const _Secrets = @import("Secrets.local.zig"); // replace Secrets.example.zig with Secrets.local.zig
const Secrets = _Secrets{};
const UDP = @import("UDP.zig");
const Sensors = @import("Sensors.zig");
const Video = @import("Video.zig");

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
    const gridNode = try Shape.Grid.init(alloc, 1000, 5);
    const axisNode = try Shape.Axis.init(alloc, Vec3{ .x = 0.0, .y = 0.5, .z = 0.0 }, 10.0);
    const triangleNode = try Shape.Triangle.init(alloc, Vec3{ .x = 0.0, .y = 1.0, .z = 10.0 }, null);
    const droneAxis = try Shape.Axis.init(alloc, Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 }, 2.0);
    const boxNode = try Shape.Box.init(alloc, null, null, null, null);

    var canvasNode = try Node.init(alloc, null);

    var canvasNodeLeft = try Shape.Plane.init(alloc, Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 }, 12.8, 7.2);
    var canvasNodeRight = try Shape.Plane.init(alloc, Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 }, 12.8, 7.2);
    canvasNodeLeft.setRotation(Math.Quaternion{ .w = 1, .x = 1.0, .y = 0, .z = 0 });
    canvasNodeLeft.setPosition(-6.45, 3.6, -5);
    canvasNodeRight.setRotation(Math.Quaternion{ .w = 1, .x = 1.0, .y = 0, .z = 0 });
    canvasNodeRight.setPosition(6.45, 3.6, -5);
    try canvasNode.addChild(canvasNodeLeft);
    try canvasNode.addChild(canvasNodeRight);

    //Initializing drone node group (axis & box rotated by PoseHandler)
    var droneNode = try Node.init(alloc, null);
    droneNode.setPosition(0, 0.5, 0);
    try droneNode.addChild(boxNode);
    try droneNode.addChild(droneAxis);

    //Adding Nodes to Environment (parent node)
    var environment = try Node.init(alloc, null);
    try environment.addChild(gridNode);
    try environment.addChild(axisNode);
    try environment.addChild(triangleNode);
    try environment.addChild(droneNode);
    try environment.addChild(canvasNode);

    //Adding environment to scene
    try scene.addNode("Environment", environment);

    //Debugging Entities
    scene.getSceneGraph();

    std.debug.print("\nIntial Camera Pos: {d}\n", .{[_]f32{
        scene.camera.position.x,
        scene.camera.position.y,
        scene.camera.position.z,
    }});

    //Initialize UDP servers
    var imu_server = UDP.init(
        Secrets.host_ip,
        Secrets.host_port_imu,
        Secrets.client_ip,
        Secrets.client_port_imu,
    );

    var video_server = UDP.init(
        Secrets.host_ip,
        Secrets.host_port_video,
        Secrets.client_ip,
        Secrets.client_port_video,
    );

    const pose_handler = Sensors.PoseHandler.init(droneNode);
    var pose_udp_handler = UDP.Handler(Sensors.PoseHandler).init(pose_handler);
    const pose_interface = pose_udp_handler.interface();
    try imu_server.start(pose_interface);

    const width: usize = 1280;
    const height: usize = 1080;
    var video_handler = try Video.VideoHandler.init(
        alloc,
        canvasNodeLeft,
        null,
        width,
        height,
        Video.frameCallback,
    );
    defer video_handler.deinit();

    //use UDP servers allocator instead??
    var video_udp_handler = UDP.Handler(Video.VideoHandler).init(video_handler);
    const video_interface = video_udp_handler.interface();
    try video_server.start(video_interface);

    //Render loop
    while (c.glfwWindowShouldClose(window) == 0) {
        c.glfwPollEvents();

        if (c.glfwGetWindowAttrib(window, c.GLFW_FOCUSED) == c.GLFW_FALSE or scene.width == 0 or scene.height == 0) {
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
