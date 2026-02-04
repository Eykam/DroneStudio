const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const gl = @import("../../bindings/gl.zig");
const Opengl = @import("../graphics/OpenGL.zig");
const Camera = @import("../components/Camera.zig");
const ECSManager = @import("../ECSManager.zig");
const Controller = @import("../components/Controller.zig");
const Transform = @import("../components/Transform.zig");
const Viewport = @import("../components/Viewports.zig");
const Collisions = @import("../components/Collisions.zig");
const IMUSensor = @import("../components/IMUSensor.zig");
const FlightController = @import("../components/FlightController.zig");
const PathPlayback = @import("../components/PathPlayback.zig");
const SLAM = @import("../components/SLAMSystem.zig");
const DroneCamera = @import("DroneCamera.zig");
const SensorCamera = @import("SensorCamera.zig");

const glfw = gl.glfw;
const Vec3 = Math.Vec3;

pub fn spawn(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
    scene_width: u32,
    scene_height: u32,
) !Core.EntityID {
    const root_tf = Transform.TransformComponent.init(alloc);

    const drone_body_resource = try ecs.world.resource_manager.loadGLTFModelCached(
        alloc,
        "assets/drone/scene.gltf",
    );
    defer drone_body_resource.deinit();

    // Create visual model (no physics)
    var entities = try ecs.createEntitiesFromModel(drone_body_resource);
    const drone_body_entity = entities.root_entity;

    // Create convex hull collider from model
    const collider = try Collisions.ColliderComponent.initFromModel(
        alloc,
        drone_body_resource,
        .{ .ConvexHull = &[_]Collisions.ConvexHullShape{} },
        ecs.world.resource_manager,
    );

    // Create physics body with the collider's shape
    var rigid_body = Collisions.RigidBodyComponent.init(1.5, collider.bullet_shape.?);

    // Set realistic moment of inertia for a quadrotor drone
    // Based on typical quadrotor with 0.3m arm length and 1kg mass
    // Ixx ≈ Iyy (roll/pitch inertia), Izz (yaw inertia) is smaller
    const arm_length: f32 = 0.3; // meters
    const motor_mass: f32 = 0.1; // kg per motor
    const body_mass: f32 = rigid_body.mass - 4 * motor_mass;

    // Simplified calculation: 4 point masses at motor positions + central body
    const Ixx = 2 * motor_mass * arm_length * arm_length + body_mass * 0.02; // kg⋅m²
    const Iyy = Ixx; // Symmetric for quad-X
    const Izz = 4 * motor_mass * arm_length * arm_length + body_mass * 0.01; // Higher yaw inertia

    rigid_body.setInertia(Ixx, Iyy, Izz);
    std.debug.print("Drone inertia: Ixx={d:.6}, Iyy={d:.6}, Izz={d:.6} kg⋅m²\n", .{ Ixx, Iyy, Izz });

    // Set initial position offset from collision shape center of mass
    rigid_body.translate(.{ 0, 2, 0 });
    rigid_body.rotate(.{ 0, 0, 0, 1 });

    // Create IMU sensor component with realistic parameters
    var imu_sensor = IMUSensor.IMUSensorComponent.init();
    imu_sensor.pos_body = Vec3.init(0.0, 0.0, 0.0);
    imu_sensor.rot_body = Math.Quaternion.identity();

    // Create flight controller with OpenGL to NED coordinate adapter
    const coord_adapter = FlightController.OpenGLAdapter.create();
    const flight_controller = FlightController.FlightControllerComponent.init(
        .{},
        rigid_body.mass,
        .Attitude,
        coord_adapter,
    );
    // Create drone controller for flight input
    const drone_input_controller = FlightController.DroneInputController.createComponent();

    // Create path playback component for simulation
    const path_playback = PathPlayback.PathPlaybackComponent{};

    // Spawn drone with all flight control components
    const root_eid = try ecs.spawn(.{
        root_tf,
        collider,
        rigid_body,
        imu_sensor,
        flight_controller,
        drone_input_controller,
        path_playback,
    });

    // Set this drone as the selected entity for control
    ecs.control_system.setSelectedEntity(root_eid);

    // Clean up the entity map since we no longer need it
    entities.entity_map.deinit();

    // Stereo camera configuration
    const baseline = 0.075; // 75mm stereo baseline
    const sensor_config = SensorCamera.Config{};

    // Spawn cameras (each prefab spawns entity + frustum child)
    const drone_cam_eid = try DroneCamera.spawn(alloc, ecs, .{}, scene_width, scene_height);
    const sensor_cam_left_eid = try SensorCamera.spawn(alloc, ecs, "sensor_cam_left", .{ .pos = .{ -baseline / 2.0, 0.0, 0.15 } });
    const sensor_cam_right_eid = try SensorCamera.spawn(alloc, ecs, "sensor_cam_right", .{ .pos = .{ baseline / 2.0, 0.0, 0.15 } });

    // Parent cameras to drone
    try ecs.transform_system.addChild(root_eid, drone_body_entity);
    try ecs.transform_system.addChild(root_eid, drone_cam_eid);
    try ecs.transform_system.addChild(root_eid, sensor_cam_left_eid);
    try ecs.transform_system.addChild(root_eid, sensor_cam_right_eid);

    // Create SLAM component with stereo camera viewports
    const left_vp = ecs.viewport_components.get(sensor_cam_left_eid).?;
    const right_vp = ecs.viewport_components.get(sensor_cam_right_eid).?;
    const drone_tf = ecs.transform_components.get(root_eid).?;

    var slam_component = try SLAM.SLAMComponent.init(alloc, .{
        .config = SLAM.SLAMConfig.initWithIntrinsics(
            sensor_config.resolution_width,
            sensor_config.resolution_height,
            sensor_config.module.fx(sensor_config.resolution_width),
            sensor_config.module.fy(sensor_config.resolution_height),
            baseline,
        ),
        .left_viewport = left_vp,
        .right_viewport = right_vp,
        .ground_truth_transform = drone_tf,
    });
    try slam_component.attach(ecs, root_eid);

    return root_eid;
}
