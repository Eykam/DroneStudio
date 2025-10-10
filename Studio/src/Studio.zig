const std = @import("std");
const gl = @import("core/bindings/gl.zig");
const UI = @import("core/UI.zig");
const Core = @import("core/ecs/Core.zig");
const ECSManager = @import("core/ecs/ECSManager.zig");
const Renderer = @import("core/ecs/components/Renderer.zig");
const Physics = @import("core/ecs/components/Physics.zig");
const Collisions = @import("core/ecs/components/Collisions.zig");
const FreeCamera = @import("core/ecs/prefabs/FreeCamera.zig");
const Drone = @import("core/ecs/prefabs/Drone.zig");
const HintzeHall = @import("core/ecs/prefabs/HintzeHall.zig");
const Ground = @import("core/ecs/prefabs/Ground.zig");
const Box = @import("core/ecs/prefabs/Box.zig");
const Robot = @import("core/ecs/prefabs/Robot.zig");
const Farm = @import("core/ecs/prefabs/Farm.zig");
const Math = @import("core/Math.zig");
const c = @import("core/bindings/c.zig");
const imgui = c.imgui;

const glfw = gl.glfw;

var should_exit = std.atomic.Value(bool).init(false);

fn handleSignal(_: c_int) callconv(.C) void {
    should_exit.store(true, .release);
}

pub fn main() !void {
    std.debug.print("STUDIO MAIN: main() function called!\n", .{});
    // var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    // defer arena.deinit();
    // const alloc = arena.allocator();

    var gpa: std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }) = .init;
    const alloc = gpa.allocator();
    defer {
        const status = gpa.deinit();
        if (status == .leak) @panic("memory leaks detected");
    }
    // =============================================== Scene Graph Initialization ===============================================
    const ECS = try ECSManager.init(alloc);
    defer ECS.deinit();

    std.posix.sigaction(std.posix.SIG.INT, &std.posix.Sigaction{
        .handler = .{ .handler = handleSignal },
        .mask = std.posix.empty_sigset,
        .flags = 0,
    }, null);

    const window = ECS.globals_system.window;
    const scene_width = 1920;
    const scene_height = 1080;

    // _ = try Drone.spawn(alloc, ECS, scene_width, scene_height);
    // _ = try HintzeHall.spawn(alloc, ECS);
    _ = try Farm.spawn(alloc, ECS);
    // _ = try Box.spawn(alloc, ECS, .ConvexHull, .{ 0, 10, 0 }, .{ 1, 1, 1 }, 1.0, scene_width, scene_height);
    // _ = try Robot.spawn(alloc, ECS, .{ .position = .{ 0, 10, 0 } }, scene_width, scene_height);
    // _ = try Ground.spawn(alloc, ECS, .{});
    _ = FreeCamera.spawn(alloc, ECS, scene_width, scene_height);

    std.debug.print("{}\n", .{ECS.world.resource_manager});
    // Position the hall
    // const hall_transform = ECS.transform_components.get(hintze_hall_entity).?;
    // hall_transform.setPosition(0, -1.0, 0);

    std.debug.print("Initializing UI...\n", .{});
    const windows = [_]type{
        UI.RootWindow,
    };

    const TWindowManager = UI.WindowManager(&windows);
    var WindowManager = try TWindowManager.init(alloc, .{ .ecs = ECS });
    defer WindowManager.deinit();

    std.debug.print("Starting main loop...\n", .{});
    while (!should_exit.load(.acquire) and glfw.glfwWindowShouldClose(window) == 0) {
        glfw.glfwPollEvents();

        if (glfw.glfwGetWindowAttrib(window, glfw.GLFW_FOCUSED) == glfw.GLFW_FALSE) {
            continue; // Skip frame if window is not focused
        }

        const current_time = glfw.glfwGetTime();

        WindowManager.drawAll();

        try ECS.update(current_time);

        // The rest: ImGui rendering
        imgui.igRender();
        imgui.ImGui_ImplOpenGL3_RenderDrawData(imgui.igGetDrawData());

        glfw.glfwSwapBuffers(window);
        glfw.glfwPollEvents();
    }
}

fn skip() !void {
    return error.SkipZigTest;
}

test "debug wireframe memory leak with HintzeHall" {
    try skip();
    const allocator = std.testing.allocator;

    std.debug.print("\n=== Testing Debug Wireframe Memory Leak with HintzeHall ===\n", .{});

    // Create a minimal world and ECS manager
    var ecs_manager = try ECSManager.init(allocator);
    defer ecs_manager.deinit();

    // Create HintzeHall entity with collision meshes
    const hall_entity = try HintzeHall.spawn(allocator, ecs_manager);
    std.debug.print("Created HintzeHall entity: {}\n", .{hall_entity.id});

    // Enable debug wireframes
    try ecs_manager.collision_system.setDebugWireframes(true);
    std.debug.print("Enabled debug wireframes\n", .{});

    // Add some debug output to the wireframe system
    if (ecs_manager.collision_system.debug_wireframe_system) |_| {
        std.debug.print("Debug wireframe system initialized and enabled\n", .{});
    }

    // Simulate multiple update cycles to observe memory growth
    for (0..20) |frame| {
        // Update collision system (which updates debug wireframes)
        try ecs_manager.update(50 * std.time.ns_per_ms);

        // Every 5 frames, print status
        if (frame % 5 == 0) {
            std.debug.print("Frame {}: Updated collision system\n", .{frame});

            // Check if wireframes are being generated
            if (ecs_manager.collision_system.physics_thread) |physics_thread| {
                const wireframes = physics_thread.getDebugWireframes();
                var total_vertices: usize = 0;
                for (wireframes) |wf| {
                    total_vertices += wf.vertices.len;
                }
                std.debug.print("  Wireframes: {} entities, {} total vertices\n", .{ wireframes.len, total_vertices });
            }
        }

        // Small delay to let physics thread process
        std.time.sleep(50 * std.time.ns_per_ms);
    }

    // Disable debug wireframes
    try ecs_manager.collision_system.setDebugWireframes(false);
    std.debug.print("Disabled debug wireframes\n", .{});

    // Run a few more frames to see if memory is released
    for (0..5) |_| {
        try ecs_manager.update(50 * std.time.ns_per_ms);
        std.time.sleep(50 * std.time.ns_per_ms);
    }

    std.debug.print("Test completed - monitor memory usage externally\n", .{});
}
