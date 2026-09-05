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
const ChassisManifest = @import("../../ChassisManifest.zig");
const Sensors = @import("Sensors.zig");

/// CAD glTF frame (+X fwd, +Z up) -> sim scene frame (+Y up). Rotation +90deg about X.
/// NOTE: assumes the drone scene model nose is +X. VERIFY VISUALLY on first
/// manifest-driven desktop run (see docs/MANIFEST_SIM.md).
fn glbPosToSim(p: [3]f32) [3]f32 {
    return .{ p[0], p[2], -p[1] };
}

/// Same basis swap applied to a quaternion (x, y, z, w) -> (x, z, -y, w).
fn glbQuatToSim(q: [4]f32) Math.Quaternion {
    return Math.Quaternion.init(q[0], q[2], -q[1], q[3]);
}

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

    // Optional CAD chassis manifest (dronestudio.chassis/1.2), env-gated via
    // DRONE_CHASSIS_MANIFEST. When unset or unloadable, the hardcoded values
    // below are used unchanged.
    var manifest: ?std.json.Parsed(ChassisManifest.ChassisManifest) = null;
    defer if (manifest) |*m| m.deinit();
    if (std.posix.getenv("DRONE_CHASSIS_MANIFEST")) |mpath| {
        manifest = ChassisManifest.ChassisManifest.load(alloc, mpath) catch |err| blk: {
            std.debug.print("ChassisManifest: failed to load {s} ({}), using hardcoded drone params\n", .{ mpath, err });
            break :blk null;
        };
        if (manifest) |*m| std.debug.print("ChassisManifest: loaded {s} (schema {s}, name {s})\n", .{ mpath, m.value.schema, m.value.name });
    }

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
    const drone_mass: f32 = if (manifest) |*m| m.value.totalMassKg() else 1.5;
    var rigid_body = Collisions.RigidBodyComponent.init(drone_mass, collider.bullet_shape.?);

    // Moment of inertia: manifest diagonal inertia about CoM when available,
    // else the simplified 4-point-mass estimate (0.3m arm, 1kg class).
    // Ixx ≈ Iyy (roll/pitch inertia), Izz (yaw inertia) is smaller
    var Ixx: f32 = undefined;
    var Iyy: f32 = undefined;
    var Izz: f32 = undefined;
    if (manifest) |*m| {
        const inertia = m.value.diagonalInertia();
        Ixx = inertia[0];
        Iyy = inertia[1];
        Izz = inertia[2];
    } else {
        const arm_length: f32 = 0.3; // meters
        const motor_mass: f32 = 0.1; // kg per motor
        const body_mass: f32 = rigid_body.mass - 4 * motor_mass;
        Ixx = 2 * motor_mass * arm_length * arm_length + body_mass * 0.02; // kg⋅m²
        Iyy = Ixx; // Symmetric for quad-X
        Izz = 4 * motor_mass * arm_length * arm_length + body_mass * 0.01; // Higher yaw inertia
    }

    rigid_body.setInertia(Ixx, Iyy, Izz);
    std.debug.print("Drone mass={d:.3}kg inertia: Ixx={d:.6}, Iyy={d:.6}, Izz={d:.6} kg⋅m²\n", .{ drone_mass, Ixx, Iyy, Izz });

    // Set initial position offset from collision shape center of mass
    rigid_body.translate(.{ 0, 2, 0 });
    rigid_body.rotate(.{ 0, 0, 0, 1 });

    // Create IMU sensor component with realistic parameters.
    // Manifest (1.2): mount pose from imu.offset_from_com_m / rotation_quat_xyzw
    // (accel/gyro die frame; AK8963 mag remap happens in the estimator, not here).
    var imu_sensor = IMUSensor.IMUSensorComponent.init();
    if (manifest) |*m| {
        const off = glbPosToSim(m.value.imuOffsetFromComM());
        imu_sensor.pos_body = Vec3.init(off[0], off[1], off[2]);
        imu_sensor.rot_body = glbQuatToSim(m.value.imuRotationQuatXyzw());
        std.debug.print("IMU mount (sim frame): pos=({d:.4},{d:.4},{d:.4}) lever-arm on\n", .{ off[0], off[1], off[2] });
    } else {
        imu_sensor.pos_body = Vec3.init(0.0, 0.0, 0.0);
        imu_sensor.rot_body = Math.Quaternion.identity();
    }

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

    // Stereo camera configuration. Manifest (1.2) cameras[] drives poses +
    // module FOV (Pi Cam 3 Standard sensor/lens); else legacy 75mm baseline.
    var baseline: f32 = 0.075; // 75mm stereo baseline
    var cam_module = Sensors.Default;
    var cam_left_pos: [3]f32 = .{ -baseline / 2.0, 0.0, 0.15 };
    var cam_right_pos: [3]f32 = .{ baseline / 2.0, 0.0, 0.15 };
    if (manifest) |*m| {
        const cams = m.value.cameras;
        if (cams.len >= 1) {
            const c0 = cams[0];
            cam_module = Sensors.CameraModule{
                .name = c0.id,
                .sensor = Sensors.Module.PiCam3_Standard.sensor,
                .focal_length_mm = Sensors.Module.PiCam3_Standard.focal_length_mm,
                .hfov_degrees = @floatCast(c0.hfov_deg),
            };
            cam_left_pos = glbPosToSim(.{ @floatCast(c0.lens_origin_m[0]), @floatCast(c0.lens_origin_m[1]), @floatCast(c0.lens_origin_m[2]) });
            if (cams.len >= 2) {
                const c1 = cams[1];
                cam_right_pos = glbPosToSim(.{ @floatCast(c1.lens_origin_m[0]), @floatCast(c1.lens_origin_m[1]), @floatCast(c1.lens_origin_m[2]) });
                const dx = cam_right_pos[0] - cam_left_pos[0];
                const dy = cam_right_pos[1] - cam_left_pos[1];
                const dz = cam_right_pos[2] - cam_left_pos[2];
                baseline = @sqrt(dx * dx + dy * dy + dz * dz);
            }
            std.debug.print("Cameras from manifest: L=({d:.4},{d:.4},{d:.4}) R=({d:.4},{d:.4},{d:.4}) baseline={d:.4}m hfov={d:.1}\n", .{ cam_left_pos[0], cam_left_pos[1], cam_left_pos[2], cam_right_pos[0], cam_right_pos[1], cam_right_pos[2], baseline, cam_module.hfov_degrees });
        }
    }
    const sensor_config = SensorCamera.Config{ .module = cam_module };

    // Spawn cameras (each prefab spawns entity + frustum child)
    const drone_cam_eid = try DroneCamera.spawn(alloc, ecs, .{}, scene_width, scene_height);
    const sensor_cam_left_eid = try SensorCamera.spawn(alloc, ecs, "sensor_cam_left", .{ .pos = cam_left_pos, .module = cam_module });
    const sensor_cam_right_eid = try SensorCamera.spawn(alloc, ecs, "sensor_cam_right", .{ .pos = cam_right_pos, .module = cam_module });

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
