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
const FlightInput = @import("../components/FlightInput.zig");
const DroneCamera = @import("DroneCamera.zig");
const SensorCamera = @import("SensorCamera.zig");
const Frustum = @import("Frustum.zig");

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
    var rigid_body = Collisions.RigidBodyComponent.init(1.0, collider.bullet_shape.?);

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
    const flight_input = FlightInput.FlightInputComponent.init();

    const drone_cam = try DroneCamera.generate(alloc, .{}, scene_width, scene_height);

    const disparity = 0.075; // 75mm
    const sensor_cam_left = try SensorCamera.generate(alloc, "sensor_cam_left", .{ .pos = .{ -disparity / 2.0, 0.0, 0.15 } });
    const sensor_cam_right = try SensorCamera.generate(alloc, "sensor_cam_right", .{ .pos = .{ disparity / 2.0, 0.0, 0.15 } });

    const drone_cam_frustum = try Frustum.generate(
        alloc,
        ecs,
        "drone_cam_frustum",
        drone_cam.cam.fov,
        drone_cam.cam.aspect,
        1.0,
        0.1,
    );
    const sensor_cam_frustum_left = try Frustum.generate(
        alloc,
        ecs,
        "sensor_cam_frustum",
        sensor_cam_left.cam.fov,
        sensor_cam_left.cam.aspect,
        1.0,
        0.1,
    );

    const sensor_cam_frustum_right = try Frustum.generate(
        alloc,
        ecs,
        "sensor_cam_frustum",
        sensor_cam_right.cam.fov,
        sensor_cam_right.cam.aspect,
        1.0,
        0.1,
    );

    // Spawn drone with all flight control components
    const root_eid = try ecs.spawn(.{
        root_tf,
        collider,
        rigid_body,
        imu_sensor,
        flight_controller,
        flight_input,
    });

    // Clean up the entity map since we no longer need it
    entities.entity_map.deinit();

    const drone_cam_eid = try ecs.spawn(drone_cam);
    const sensor_cam_left_eid = try ecs.spawn(sensor_cam_left);
    const sensor_cam_right_eid = try ecs.spawn(sensor_cam_right);

    const drone_cam_frustum_eid = try ecs.spawn(drone_cam_frustum);
    const sensor_cam_frustum_left_eid = try ecs.spawn(sensor_cam_frustum_left);
    const sensor_cam_frustum_right_eid = try ecs.spawn(sensor_cam_frustum_right);

    try ecs.transform_system.addChild(root_eid, drone_body_entity);
    try ecs.transform_system.addChild(root_eid, drone_cam_eid);
    try ecs.transform_system.addChild(drone_cam_eid, drone_cam_frustum_eid);

    try ecs.transform_system.addChild(root_eid, sensor_cam_left_eid);
    try ecs.transform_system.addChild(sensor_cam_left_eid, sensor_cam_frustum_left_eid);

    try ecs.transform_system.addChild(root_eid, sensor_cam_right_eid);
    try ecs.transform_system.addChild(sensor_cam_right_eid, sensor_cam_frustum_right_eid);

    return root_eid;
}
