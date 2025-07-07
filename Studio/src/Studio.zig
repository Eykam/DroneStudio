const std = @import("std");
const gl = @import("core/bindings/gl.zig");
const UI = @import("core/UI.zig");
const ECSManager = @import("core/ecs/ECSManager.zig");
const Renderer = @import("core/ecs/components/Renderer.zig");
const FreeCamera = @import("core/ecs/prefabs/FreeCamera.zig");
const Drone = @import("core/ecs/prefabs/Drone.zig");
const Ground = @import("core/ecs/prefabs/Ground.zig");
const c = @import("core/bindings/c.zig");
const imgui = c.imgui;

const glfw = gl.glfw;

var should_exit = std.atomic.Value(bool).init(false);

fn handleSignal(_: c_int) callconv(.C) void {
    should_exit.store(true, .release);
}

pub fn main() !void {
    std.debug.print("STUDIO MAIN: main() function called!\n", .{});
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

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

    const free_cam = try FreeCamera.generate(alloc, .{}, scene_width, scene_height);
    _ = try ECS.spawn(free_cam);

    _ = try Drone.spawn(alloc, ECS, scene_width, scene_height);

    // const resource_manager = ECS.world.resource_manager;
    // const hintze_hall_resource = try resource_manager.loadGLTFModelCached(
    //     alloc,
    //     "assets/hintze_hall/scene.gltf",
    // );

    // var hintze_hall_result = try ECS.createEntitiesFromModel(hintze_hall_resource);
    // const hintze_hall_entity = hintze_hall_result.root_entity;

    // // Create static collision for the hall using the actual hall mesh
    // var hall_collider = try Collisions.ColliderComponent.initFromModel(
    //     alloc,
    //     hintze_hall_resource,
    //     .{ .TriangleMesh = .{} },
    //     ECS.world.resource_manager,
    // );
    // var hall_body = Collisions.RigidBodyComponent.init(0.0, hall_collider.bullet_shape.?);

    // // Add components to the hall entity
    // try hall_collider.attach(ECS, hintze_hall_entity);
    // try hall_body.attach(ECS, hintze_hall_entity);

    // // Clean up entity map
    // hintze_hall_result.entity_map.deinit();
    // ECS.transform_components.get(hintze_hall_entity).?.setPosition(0, -1.0, 0);

    // Create ground plane using the Ground prefab
    _ = try Ground.spawn(alloc, ECS, .{
        .size = 100.0,
        .position = .{ 0.0, -10.0, 0.0 },
        .color = .{ 0.4, 0.6, 0.4 }, // Slightly darker green
    });

    std.debug.print("Initializing UI...\n", .{});
    const windows = [_]type{
        UI.RootWindow,
    };

    const TWindowManager = UI.WindowManager(&windows);
    const WindowManager = try TWindowManager.init(
        alloc,
        .{
            .ecs = ECS,
        },
    );

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

    // var scene = try Scene.init(alloc, window);
    // defer scene.deinit();
    // scene.setupCallbacks(window);
    // scene.setAmbientColor(0.8, 0.9, 1.0);
    // scene.setAmbientStrength(0.25);

    // const hintze_hall = try GLTF.loadAsNode(alloc, "./assets/hintze_hall/scene.gltf", "hintze_hall");
    // hintze_hall.setPosition(0, -1.0, 0);
    // // const warehouse = try GLTF.loadAsNode(alloc, "./assets/warehouse/scene.gltf", "warehouse");
    // // warehouse.setPosition(-75, 0, 0);

    // //Initializing Entities
    // const gridNode = try Shape.Grid.init(alloc, 100, 5);
    // const axisNode = try Shape.Axis.init(alloc, Vec3.init(0.0, 0.5, 0.0), 10.0);
    // const droneAxis = try Shape.Axis.init(alloc, Vec3.init(0.0, 0.0, 0.0), 2.0);
    // const droneBody = try GLTF.loadAsNode(alloc, "./assets/drone/scene.gltf", "drone");
    // const droneSensorCamera = try SensorCamera.init(
    //     alloc,
    //     "imx708_sensor",
    //     16.0 / 9.0, // Aspect ratio (16:9)
    //     Vec3.init(0.0, 0.0, -0.5), // Position can be adjusted as needed
    //     3.04, // Focal length in mm (calculated for 102° FOV)
    //     6.287, // Sensor width in mm (1/2.3" sensor)
    //     4.712, // Sensor height in mm
    //     1280, // Resolution width in pixels
    //     720, // Resolution height in pixels
    // );
    // try droneSensorCamera.base.toggle_debug_mode(alloc);
    // try droneBody.addChild(droneSensorCamera.base.node);

    // const aspect_ratio = scene.width / scene.height;
    // const freeCamera = try FreeCamera.init(alloc, "free", aspect_ratio, null, null, null);
    // const droneFirstPerson = try DroneCamera.init(alloc, "simulation_first_person", aspect_ratio, Vec3.init(0.0, 0.0, -0.5), null, null);
    // //TODO: Remove third / first person. Instead create toggle that will change the position.
    // const droneThirdPerson = try DroneCamera.init(alloc, "simulation_third_person", aspect_ratio, Vec3.init(0.0, 1, 5.0), null, null);
    // try droneThirdPerson.base.toggle_debug_mode(alloc);

    // var freeCamera_union = Camera{ .Free = freeCamera.* };
    // var droneFirstPerson_union = Camera{ .DroneControl = droneFirstPerson.* };
    // var droneThirdPerson_union = Camera{ .DroneControl = droneThirdPerson.* };
    // var droneSensorCamera_union = Camera{ .SensorCamera = droneSensorCamera.* };

    // try scene.camera_manager.register_camera(&freeCamera_union);
    // try scene.camera_manager.register_camera(&droneFirstPerson_union);
    // try scene.camera_manager.register_camera(&droneThirdPerson_union);
    // try scene.camera_manager.register_camera(&droneSensorCamera_union);

    // // const canvas_width = 12.8;
    // // const canvas_height = 7.2;
    // // const texture_dims = [_]u32{ 1280, 720 };

    // // var canvasNode = try Node.init(alloc, null, null, null);
    // // canvasNode.setRotation(Math.Quaternion.init(1.0, 0, 0, 1.0));

    // // var canvasNodeLeft = try Shape.TexturedPlane.init(
    // //     alloc,
    // //     null,
    // //     canvas_width,
    // //     canvas_height,
    // //     .{ .w = texture_dims[0], .h = texture_dims[1] },
    // // );
    // // canvasNodeLeft.setPosition(-(canvas_width / 2.0) - 0.1, canvas_height / 2.0, 5);
    // // try canvasNode.addChild(canvasNodeLeft);

    // // var canvasNodeRight = try Shape.TexturedPlane.init(
    // //     alloc,
    // //     null,
    // //     canvas_width,
    // //     canvas_height,
    // //     .{ .w = texture_dims[0], .h = texture_dims[1] },
    // // );
    // // canvasNodeRight.setPosition((canvas_width / 2.0) + 0.1, canvas_height / 2.0, 5);
    // // try canvasNode.addChild(canvasNodeRight);

    // // var canvasNodeCombined = try Shape.TexturedPlane.init(
    // //     alloc,
    // //     null,
    // //     canvas_width,
    // //     canvas_height,
    // //     .{ .w = texture_dims[0], .h = texture_dims[1] },
    // // );
    // // canvasNodeCombined.setPosition(0, canvas_height / 2.0, -10);
    // // try canvasNode.addChild(canvasNodeCombined);

    // // var canvasNodeTemporal = try Shape.TexturedPlane.init(
    // //     alloc,
    // //     null,
    // //     canvas_width,
    // //     canvas_height,
    // //     .{ .w = texture_dims[0], .h = texture_dims[1] },
    // // );
    // // // Position it next to the combined canvas
    // // canvasNodeTemporal.setPosition(0, (3.0 * canvas_height / 2.0) + 0.2, -10);
    // // try canvasNode.addChild(canvasNodeTemporal);

    // //Initializing drone node group (axis & box rotated by PoseHandler)
    // var droneNode = try Node.init(alloc, null, null, null);
    // droneNode.setPosition(0, 0.5, 0);
    // try droneNode.addChild(droneBody);
    // try droneNode.addChild(droneAxis);
    // try droneNode.addChild(droneFirstPerson.base.node);
    // try droneNode.addChild(droneThirdPerson.base.node);

    // //Adding Nodes to Environment (parent node)
    // var environment = try Node.init(alloc, null, null, null);
    // try environment.addChild(hintze_hall);
    // // try environment.addChild(warehouse);
    // try environment.addChild(freeCamera.base.node);
    // try environment.addChild(gridNode);
    // try environment.addChild(axisNode);
    // try environment.addChild(droneNode);
    // // try environment.addChild(canvasNode);

    // //Adding environment to scene
    // try scene.addNode("Environment", environment);

    // //Debugging Entities
    // scene.getSceneGraph();

    // std.debug.print("\nIntial Camera Pos: {d}\n", .{scene.camera_manager.main_camera.?.get_base().position});

    // // ======================================================= Motor controller & IMU Setup =======================================================
    // const motor_controller = try Drone.MotorControllerClient.init(
    //     alloc,
    //     null,
    // );
    // scene.motor_controller = motor_controller;

    // //Initialize UDP servers
    // var imu_server = UDP.init(
    //     Secrets.host_ip,
    //     Secrets.host_port_imu,
    //     Secrets.client_ip,
    //     Secrets.client_port_imu,
    // );

    // var pose_handler = try Sensors.PoseHandler.init(alloc, droneNode, motor_controller.config);
    // motor_controller.sensor_state = pose_handler.sensor_state;

    // var pose_udp_handler = UDP.Handler(Sensors.PoseHandler).init(&pose_handler);
    // const pose_interface = pose_udp_handler.interface();
    // try imu_server.start(pose_interface);
    // // ================================================= Stereo Matching Setup =================================================

    // // const scene_manager = try SceneManager.init(
    // //     alloc,
    // //     environment,
    // //     texture_dims[0],
    // //     texture_dims[1],
    // //     null,
    // //     null,
    // //     &pose_handler,
    // // );
    // // defer scene_manager.deinit();

    // // ============================================= FFMPEG Video Processing Setup =============================================

    // // try Video.initializeFFmpegNetwork();
    // // defer Video.deinitFFmpegNetwork();

    // // var video_handler_left = try Video.VideoHandler.start(
    // //     alloc,
    // //     canvasNodeLeft,
    // //     Secrets.sdp_content_left,
    // //     null,
    // //     Video.frameCallback,
    // //     null,
    // //     scene_manager,
    // //     .left,
    // // );
    // // defer video_handler_left.join();

    // // var video_handler_right = try Video.VideoHandler.start(
    // //     alloc,
    // //     canvasNodeRight,
    // //     Secrets.sdp_content_right,
    // //     null,
    // //     Video.frameCallback,
    // //     null,
    // //     scene_manager,
    // //     .right,
    // // );
    // // defer video_handler_right.join();

    // // ==================================================== UI Window Setup ====================================================

    // const windows = [_]type{
    //     // UI.OverlayWindow,
    //     UI.RootWindow,
    //     // UI.StereoDebugWindow,
    //     // UI.DroneConfigWindow,
    //     // UI.BatteryStatusWindow,
    // };

    // const TWindowManager = UI.WindowManager(&windows);
    // const WindowManager = try TWindowManager.init(
    //     alloc,
    //     .{
    //         .scene = scene,
    //         // .StereoVO = StereoVO,
    //         .pose_handler = &pose_handler,
    //     },
    // );

    // // ====================================================== Render Loop ======================================================

    // while (glfw.glfwWindowShouldClose(window) == 0) {
    //     glfw.glfwPollEvents();

    //     if (glfw.glfwGetWindowAttrib(window, glfw.GLFW_FOCUSED) == glfw.GLFW_FALSE or scene.width == 0 or scene.height == 0) {
    //         continue; // Skip frame if window is not focused
    //     }

    //     const current_time = glfw.glfwGetTime();

    //     // Calculate delta time
    //     scene.appState.delta_time = @floatCast(current_time - scene.appState.last_frame_time);
    //     scene.appState.last_frame_time = current_time;

    //     WindowManager.drawAll();

    //     scene.processInput(false);
    //     scene.render(window);

    //     if (!scene.appState.paused) {
    //         // const start = try std.time.Instant.now();
    //         // try scene_manager.processFramePair();
    //         // const end = try std.time.Instant.now();
    //         // const debug_str = "= Total Stereo Pipeline Execution Time: {d:.3} ms =\n";
    //         // std.debug.print("\n{s}\n", .{"=" ** (debug_str.len - 1)});
    //         // std.debug.print(debug_str, .{@as(f64, @floatFromInt(end.since(start))) / 1e6});
    //         // std.debug.print("{s}\n", .{"=" ** (debug_str.len - 1)});
    //     }
    //     // } else if (StereoVO.params_changed) {
    //     //     try StereoVO.match();
    //     //     StereoVO.free_matches();
    //     //     StereoVO.params_changed = false;
    //     // }
    // }
}

test "ResourceManager - basic functionality with ECS context" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // Initialize ECS which sets up OpenGL context
    const ECS = ECSManager.init(alloc) catch |err| {
        std.debug.print("Failed to initialize ECS for testing: {}\n", .{err});
        return; // Skip test if ECS setup fails (e.g., no display)
    };
    defer ECS.deinit();

    // Test ResourceManager functionality
    const resource_manager = ECS.world.resource_manager;
    
    // Test that ResourceManager initializes with default shaders
    try testing.expect(resource_manager.shaders.count() >= 2);
    
    // Test shader retrieval
    const standard_shader = resource_manager.getShader("standard_shader");
    try testing.expect(standard_shader != null);
    try testing.expect(standard_shader.?.program_id > 0);
    
    const pbr_shader = resource_manager.getShader("pbr_shader");
    try testing.expect(pbr_shader != null);
    try testing.expect(pbr_shader.?.program_id > 0);
    
    // Test that mesh, texture, material maps start empty
    try testing.expect(resource_manager.meshes.count() == 0);
    try testing.expect(resource_manager.textures.count() == 0);
    try testing.expect(resource_manager.materials.count() == 0);
}

test "ResourceManager - material creation and management" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // Initialize ECS for OpenGL context
    const ECS = ECSManager.init(alloc) catch |err| {
        std.debug.print("Failed to initialize ECS for testing: {}\n", .{err});
        return; // Skip test if ECS setup fails
    };
    defer ECS.deinit();

    const resource_manager = ECS.world.resource_manager;
    const ResourceManager = @import("core/ecs/ResourceManager.zig");
    
    // Create a test PBR material
    const test_material = ResourceManager.MaterialVariant{
        .PBR = ResourceManager.Material(.PBR){
            .data = .{
                .baseColorFactor = .{ 1.0, 0.5, 0.2, 1.0 },
                .metallicFactor = 0.8,
                .roughnessFactor = 0.3,
            },
        },
    };
    
    // Load material with default shader
    try resource_manager.loadMaterial("test_material", test_material, null);
    
    // Test that material was loaded
    try testing.expect(resource_manager.materials.count() == 1);
    
    // Test getting the material
    const loaded_material = resource_manager.materials.get("test_material");
    try testing.expect(loaded_material != null);
    try testing.expect(loaded_material.?.ref_count == 1);
    
    // Test material type detection
    try testing.expect(loaded_material.?.material.getType() == .PBR);
}

test "ResourceManager - cache functionality" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const ResourceManager = @import("core/ecs/ResourceManager.zig");
    
    // Test cache path generation (doesn't need OpenGL)
    const path1 = ResourceManager.cachePath(alloc, "assets/test/model.gltf");
    defer alloc.free(path1);
    
    // Should replace slashes with dashes and add .bin extension
    try testing.expect(std.mem.indexOf(u8, path1, "assets-test-model.gltf.bin") != null);
    try testing.expect(std.mem.indexOf(u8, path1, ".asset-cache") != null);
    
    // Test that same path generates same cache path
    const path2 = ResourceManager.cachePath(alloc, "assets/test/model.gltf");
    defer alloc.free(path2);
    
    try testing.expectEqualStrings(path1, path2);
}

test "ResourceManager - GLTF caching with actual model" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // Initialize ECS for OpenGL context
    const ECS = ECSManager.init(alloc) catch |err| {
        std.debug.print("Failed to initialize ECS for testing: {}\n", .{err});
        return; // Skip test if ECS setup fails
    };
    defer ECS.deinit();

    const resource_manager = ECS.world.resource_manager;
    
    // Test with the drone model
    const drone_path = "assets/drone/scene.gltf";
    
    // First load - should be fresh
    std.debug.print("\n=== First load (fresh) ===\n", .{});
    const resource1 = try resource_manager.loadGLTFModelCached(alloc, drone_path);
    
    // Check that resources were loaded
    std.debug.print("After first load:\n", .{});
    std.debug.print("  Meshes loaded: {}\n", .{resource_manager.meshes.count()});
    std.debug.print("  Textures loaded: {}\n", .{resource_manager.textures.count()});
    std.debug.print("  Materials loaded: {}\n", .{resource_manager.materials.count()});
    std.debug.print("  Entities in model: {}\n", .{resource1.entities.len});
    
    const initial_mesh_count = resource_manager.meshes.count();
    const initial_texture_count = resource_manager.textures.count();
    const initial_material_count = resource_manager.materials.count();
    
    try testing.expect(initial_mesh_count > 0);
    try testing.expect(resource1.entities.len > 0);
    
    // Print entity details
    for (resource1.entities, 0..) |entity, i| {
        std.debug.print("  Entity[{}]: name={s}, mesh={s}, material={s}\n", .{
            i,
            entity.name orelse "null",
            entity.mesh_name orelse "null", 
            entity.material_name orelse "null",
        });
    }
    
    // Clean up first resource
    resource1.deinit();
    
    // Second load - should be from cache
    std.debug.print("\n=== Second load (cached) ===\n", .{});
    const resource2 = try resource_manager.loadGLTFModelCached(alloc, drone_path);
    
    // Check that resources were loaded from cache
    std.debug.print("After second load:\n", .{});
    std.debug.print("  Meshes loaded: {}\n", .{resource_manager.meshes.count()});
    std.debug.print("  Textures loaded: {}\n", .{resource_manager.textures.count()});
    std.debug.print("  Materials loaded: {}\n", .{resource_manager.materials.count()});
    std.debug.print("  Entities in model: {}\n", .{resource2.entities.len});
    
    // Should have same counts as before
    try testing.expectEqual(initial_mesh_count, resource_manager.meshes.count());
    try testing.expectEqual(initial_texture_count, resource_manager.textures.count());
    try testing.expectEqual(initial_material_count, resource_manager.materials.count());
    try testing.expectEqual(resource1.entities.len, resource2.entities.len);
    
    // Print cached entity details
    for (resource2.entities, 0..) |entity, i| {
        std.debug.print("  Entity[{}]: name={s}, mesh={s}, material={s}\n", .{
            i,
            entity.name orelse "null",
            entity.mesh_name orelse "null",
            entity.material_name orelse "null",
        });
    }
    
    // Verify entities have proper mesh and material references
    for (resource2.entities) |entity| {
        if (entity.mesh_name) |mesh_name| {
            const has_mesh = resource_manager.meshes.contains(mesh_name);
            if (!has_mesh) {
                std.debug.print("ERROR: Missing mesh '{s}' in resource manager!\n", .{mesh_name});
            }
            try testing.expect(has_mesh);
        }
        
        if (entity.material_name) |material_name| {
            const has_material = resource_manager.materials.contains(material_name);
            if (!has_material) {
                std.debug.print("ERROR: Missing material '{s}' in resource manager!\n", .{material_name});
            }
            try testing.expect(has_material);
        }
    }
    
    resource2.deinit();
}

test {
    // _ = @import("core/ecs/components/PhysicsThread.zig"); // Commented out collision system testing for now
}
