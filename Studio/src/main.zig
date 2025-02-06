const std = @import("std");
const _Secrets = @import("Secrets.local.zig"); // replace Secrets.example.zig with Secrets.local.zig
const Secrets = _Secrets{};

const Math = @import("Math.zig");
const Vec3 = Math.Vec3;

const Pipeline = @import("Pipeline.zig");
const Scene = Pipeline.Scene;
const Shape = @import("Shape.zig");
const Node = @import("Node.zig");
const Mesh = @import("Mesh.zig");

const UDP = @import("UDP.zig");
const Sensors = @import("Sensors.zig");

const Video = @import("Video.zig");
const Vision = @import("Vision.zig");
const gl = @import("bindings/gl.zig");
const glfw = gl.glfw;

const UI = @import("UI.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // ====================================================== Window Setup ======================================================

    // Initialize GLFW
    if (glfw.glfwInit() == 0) {
        std.debug.print("Failed to initialize GLFW\n", .{});
        return;
    }

    defer glfw.glfwTerminate();

    const window = try Pipeline.createWindow() orelse {
        std.debug.print("Failed to create window\n", .{});
        return;
    };
    defer glfw.glfwDestroyWindow(window);

    if (glfw.glfwGetInputMode(window, glfw.GLFW_CURSOR) != glfw.GLFW_CURSOR_DISABLED) {
        std.debug.print("Warning: Cursor not disabled as expected\n", .{});
    }

    // =============================================== Scene Graph Initialization ===============================================
    var scene = try Scene.init(alloc, window);
    defer scene.deinit();
    scene.setupCallbacks(window);

    //Initializing Entities
    const gridNode = try Shape.Grid.init(alloc, 1000, 5);
    const axisNode = try Shape.Axis.init(alloc, Vec3{ .x = 0.0, .y = 0.5, .z = 0.0 }, 10.0);
    const triangleNode = try Shape.Triangle.init(alloc, Vec3{ .x = 0.0, .y = 1.0, .z = 10.0 }, null);
    // const droneAxis = try Shape.Axis.init(alloc, Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 }, 2.0);
    // const boxNode = try Shape.Box.init(alloc, null, null, null, null);

    const canvas_width = 12.8;
    const canvas_height = 7.2;
    const texture_dims = [_]u32{ 1280, 720 };

    var canvasNode = try Node.init(alloc, null, null, null);
    canvasNode.setRotation(Math.Quaternion{ .w = 1, .x = 1.0, .y = 0, .z = 0 });

    var canvasNodeLeft = try Shape.TexturedPlane.init(
        alloc,
        null,
        canvas_width,
        canvas_height,
        .{ .w = texture_dims[0], .h = texture_dims[1] },
    );
    canvasNodeLeft.setPosition(-(canvas_width / 2.0) - 0.1, canvas_height / 2.0, 5);
    try canvasNode.addChild(canvasNodeLeft);

    var canvasNodeRight = try Shape.TexturedPlane.init(
        alloc,
        null,
        canvas_width,
        canvas_height,
        .{ .w = texture_dims[0], .h = texture_dims[1] },
    );
    canvasNodeRight.setPosition((canvas_width / 2.0) + 0.1, canvas_height / 2.0, 5);
    try canvasNode.addChild(canvasNodeRight);

    var canvasNodeCombined = try Shape.TexturedPlane.init(
        alloc,
        null,
        canvas_width,
        canvas_height,
        .{ .w = texture_dims[0], .h = texture_dims[1] },
    );
    canvasNodeCombined.setPosition(0, canvas_height / 2.0, -10);
    try canvasNode.addChild(canvasNodeCombined);

    var canvasNodeTemporal = try Shape.TexturedPlane.init(
        alloc,
        null,
        canvas_width,
        canvas_height,
        .{ .w = texture_dims[0], .h = texture_dims[1] },
    );
    // Position it next to the combined canvas
    canvasNodeTemporal.setPosition(0, (3.0 * canvas_height / 2.0) + 0.2, -10);
    try canvasNode.addChild(canvasNodeTemporal);

    //Initializing drone node group (axis & box rotated by PoseHandler)
    // var droneNode = try Node.init(alloc, null, null, null);
    // droneNode.setPosition(0, 0.5, 0);
    // try droneNode.addChild(boxNode);
    // try droneNode.addChild(droneAxis);

    //Adding Nodes to Environment (parent node)
    var environment = try Node.init(alloc, null, null, null);
    try environment.addChild(gridNode);
    try environment.addChild(axisNode);
    try environment.addChild(triangleNode);
    // try environment.addChild(droneNode);
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

    // ======================================================= IMU Setup =======================================================

    //Initialize UDP servers
    // var imu_server = UDP.init(
    //     Secrets.host_ip,
    //     Secrets.host_port_imu,
    //     Secrets.client_ip,
    //     Secrets.client_port_imu,
    // );

    // const pose_handler = Sensors.PoseHandler.init(droneNode);
    // var pose_udp_handler = UDP.Handler(Sensors.PoseHandler).init(pose_handler);
    // const pose_interface = pose_udp_handler.interface();
    // try imu_server.start(pose_interface);

    // ================================================= Stereo Matching Setup =================================================

    const StereoVO = try Vision.StereoVO.init(
        alloc,
        canvasNodeLeft,
        canvasNodeRight,
        canvasNodeCombined,
        canvasNodeTemporal,
        null,
    );
    defer StereoVO.deinit();

    // ============================================= FFMPEG Video Processing Setup =============================================

    try Video.initializeFFmpegNetwork();
    defer Video.deinitFFmpegNetwork();

    var video_handler_left = try Video.VideoHandler.start(
        alloc,
        canvasNodeLeft,
        Secrets.sdp_content_left,
        null,
        Video.frameCallback,
        null,
        StereoVO.left,
    );
    defer video_handler_left.join();

    var video_handler_right = try Video.VideoHandler.start(
        alloc,
        canvasNodeRight,
        Secrets.sdp_content_right,
        null,
        Video.frameCallback,
        null,
        StereoVO.right,
    );
    defer video_handler_right.join();

    // ==================================================== UI Window Setup ====================================================

    const windows = [_]type{ UI.OverlayWindow, UI.StereoDebugWindow };
    const TWindowManager = UI.WindowManager(&windows);
    const WindowManager = try TWindowManager.init(
        alloc,
        .{
            .scene = scene,
            .StereoVO = StereoVO,
        },
    );

    // ====================================================== Render Loop ======================================================

    while (glfw.glfwWindowShouldClose(window) == 0) {
        glfw.glfwPollEvents();

        if (glfw.glfwGetWindowAttrib(window, glfw.GLFW_FOCUSED) == glfw.GLFW_FALSE or scene.width == 0 or scene.height == 0) {
            continue; // Skip frame if window is not focused
        }

        const current_time = glfw.glfwGetTime();

        // Calculate delta time
        scene.appState.delta_time = @floatCast(current_time - scene.appState.last_frame_time);
        scene.appState.last_frame_time = current_time;

        WindowManager.drawAll();

        scene.processInput(false);
        scene.render(window);

        if (!scene.appState.paused) {
            try StereoVO.update();
        } else if (StereoVO.params_changed) {
            try StereoVO.match();
            StereoVO.free_matches();
            StereoVO.params_changed = false;
        }
    }
}

test {}
