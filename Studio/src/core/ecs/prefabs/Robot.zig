const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const gl = @import("../../bindings/gl.zig");
const ECSManager = @import("../ECSManager.zig");
const Controller = @import("../components/Controller.zig");
const Transform = @import("../components/Transform.zig");
const Collisions = @import("../components/Collisions.zig");
const Renderer = @import("../components/Renderer.zig");
const Camera = @import("../components/Camera.zig");
const DroneCamera = @import("DroneCamera.zig");
const Frustum = @import("Frustum.zig");
const ResourceManager = @import("../ResourceManager.zig");
const GLTF = @import("../../GLTF.zig");
const Mesh = @import("../../Mesh.zig");

const glfw = gl.glfw;
const Vec3 = Math.Vec3;
const CollisionSystem = Collisions.CollisionSystem;

/// Robot configuration options
pub const RobotConfig = struct {
    position: [3]f32 = .{ 0, 0, 0 },
    scale: f32 = 1.0,
    mass: f32 = 15.0,
    body_color: [4]f32 = .{ 0.2, 0.3, 0.8, 1.0 }, // Blue body
    arm_color: [4]f32 = .{ 0.8, 0.2, 0.2, 1.0 }, // Red arms
    head_color: [4]f32 = .{ 0.1, 0.8, 0.1, 1.0 }, // Green head
};

/// Component structure for the robot
pub const RobotComponents = struct {
    root: Core.EntityID,
    model_entities: []Core.EntityID,
    camera: Core.EntityID,
    frustum: Core.EntityID,

    pub fn deinit(self: *RobotComponents, alloc: std.mem.Allocator) void {
        alloc.free(self.model_entities);
    }
};

/// Physics-aware movement for robot
inline fn moveRobot(
    eid: Core.EntityID,
    tf: *Transform.TransformComponent,
    rigid_body: ?*Collisions.RigidBodyComponent,
    collision_system: ?*CollisionSystem,
    dir: Vec3,
    speed: f32,
    dt: f32,
) void {
    if (rigid_body != null and collision_system != null and rigid_body.?.mass > 0.0) {
        const force = dir.scale(speed * rigid_body.?.mass);
        std.debug.print("Robot applying force to eid {d}: [{d:.2}, {d:.2}, {d:.2}]\n", .{ eid.id, force.data[0], force.data[1], force.data[2] });
        collision_system.?.applyCentralForce(eid, force.data);
    } else {
        tf.translate(dir.scale(speed * dt).data);
    }
}

/// Physics-aware rotation for robot
inline fn rotateRobot(
    eid: Core.EntityID,
    tf: *Transform.TransformComponent,
    rigid_body: ?*Collisions.RigidBodyComponent,
    collision_system: ?*CollisionSystem,
    axis: Vec3,
    degrees: f32,
    dt: f32,
) void {
    if (rigid_body != null and collision_system != null and rigid_body.?.mass > 0.0) {
        const torque = axis.scale(Math.radians(degrees) * rigid_body.?.mass);
        collision_system.?.applyTorque(eid, torque.data);
    } else {
        const q = Math.Quaternion.from_axis_angle(axis, degrees * dt);
        tf.rotate(q);
    }
}

/// Robot control callbacks
fn moveForward(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, dt: f32) void {
    moveRobot(eid, tf, rigid_body, collision_system, tf.local_transform.get_forward(), 15.0, dt);
}
fn moveBackward(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, dt: f32) void {
    moveRobot(eid, tf, rigid_body, collision_system, tf.local_transform.get_forward().scale(-1), 15.0, dt);
}
fn strafeLeft(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, dt: f32) void {
    moveRobot(eid, tf, rigid_body, collision_system, tf.local_transform.get_right().scale(-1), 10.0, dt);
}
fn strafeRight(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, dt: f32) void {
    moveRobot(eid, tf, rigid_body, collision_system, tf.local_transform.get_right(), 10.0, dt);
}
fn jumpUp(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, dt: f32) void {
    moveRobot(eid, tf, rigid_body, collision_system, Vec3.init(0, 1, 0), 25.0, dt);
}
fn turnLeft(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, dt: f32) void {
    rotateRobot(eid, tf, rigid_body, collision_system, Vec3.init(0, 1, 0), -90.0, dt);
}
fn turnRight(eid: Core.EntityID, tf: *Transform.TransformComponent, rigid_body: ?*Collisions.RigidBodyComponent, collision_system: ?*CollisionSystem, dt: f32) void {
    rotateRobot(eid, tf, rigid_body, collision_system, Vec3.init(0, 1, 0), 90.0, dt);
}

/// Create robot controller with enhanced controls
pub fn makeRobotController(alloc: std.mem.Allocator) !Controller.ControllerComponent {
    var c = Controller.ControllerComponent.init(alloc);

    // WASD movement + QE turning + Space jump
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_W, .onPressed = moveForward });
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_S, .onPressed = moveBackward });
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_A, .onPressed = strafeLeft });
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_D, .onPressed = strafeRight });
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_SPACE, .onPressed = jumpUp });
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_Q, .onPressed = turnLeft });
    try c.key_bindings.append(.{ .key = glfw.GLFW_KEY_E, .onPressed = turnRight });

    return c;
}

/// Spawn a complex robot with programmatically created model entities and compound collision shapes
pub fn spawn(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
    config: RobotConfig,
    scene_width: u32,
    scene_height: u32,
) !RobotComponents {
    std.debug.print("Creating robot with programmatic model entities...\n", .{});

    // Create the robot model entities programmatically
    const robot_model_resource = try createRobotModelEntities(alloc, ecs.world.resource_manager, config);

    std.debug.print("Loaded robot model with {d} entities\n", .{robot_model_resource.entities.len});

    // Create root entity with physics and controller
    var root_transform = Transform.TransformComponent.init(alloc);
    root_transform.setPosition(config.position[0], config.position[1], config.position[2]);
    root_transform.setScale(config.scale, config.scale, config.scale);

    const root_controller = try makeRobotController(alloc);

    // Create compound collision shape from the GLTF model
    std.debug.print("Creating compound collision shape from GLTF model...\n", .{});
    const compound_collider = Collisions.ColliderComponent.initFromModel(
        alloc,
        robot_model_resource,
        .{ .ConvexHull = &[_]Collisions.ConvexHullShape{} },
        ecs.world.resource_manager,
    ) catch |err| {
        std.debug.print("Failed to create collision shape from model: {any}\n", .{err});
        return err;
    };

    // Create physics body for the compound shape
    var robot_rigid_body = Collisions.RigidBodyComponent.init(config.mass, compound_collider.bullet_shape.?);
    robot_rigid_body.translate(config.position);

    std.debug.print("Created collision shape from robot model (type: {s})\n", .{@tagName(compound_collider.shape)});

    // Spawn the root entity with physics
    const root_eid = try ecs.spawn(.{ root_transform, root_controller, compound_collider, robot_rigid_body });

    // Create entities for each part of the GLTF model
    var model_entities = try alloc.alloc(Core.EntityID, robot_model_resource.entities.len);

    for (robot_model_resource.entities, 0..) |model_entity, i| {
        var entity_transform = Transform.TransformComponent.init(alloc);

        // Apply the model entity's transform
        if (model_entity.local_transformation) |transform_matrix| {
            // Convert the transform matrix to position, rotation, scale
            const trs = transform_matrix.decomposeTRS();

            entity_transform.setPosition(trs.translation[0], trs.translation[1], trs.translation[2]);
            entity_transform.setRotation(trs.rotation);
            entity_transform.setScale(trs.scale[0], trs.scale[1], trs.scale[2]);
            
            std.debug.print("Visual entity[{d}]: {s} positioned at [{d:.3}, {d:.3}, {d:.3}]\n", .{
                i, model_entity.name orelse "unnamed", trs.translation[0], trs.translation[1], trs.translation[2]
            });
        }

        // Create renderer if this entity has a mesh
        if (model_entity.mesh_name) |mesh_name| {
            var entity_renderer = try Renderer.Renderable.init(alloc, mesh_name);

            // Set material if available
            if (model_entity.material_name) |material_name| {
                try entity_renderer.setMaterial(alloc, material_name);
            }

            model_entities[i] = try ecs.spawn(.{ entity_transform, entity_renderer });
        } else {
            // Entity without mesh (just transform node)
            model_entities[i] = try ecs.spawn(.{entity_transform});
        }

        // Set up parent-child relationship with root
        try ecs.transform_system.addChild(root_eid, model_entities[i]);

        std.debug.print("Created model entity[{d}]: {d}, mesh='{s}', material='{s}'\n", .{
            i,                                    model_entities[i].id,
            model_entity.mesh_name orelse "none", model_entity.material_name orelse "none",
        });
    }

    // Create camera and frustum
    const robot_camera = try DroneCamera.generate(alloc, .{}, scene_width, scene_height);
    const robot_frustum = try Frustum.generate(
        alloc,
        ecs,
        "robot_cam_frustum",
        robot_camera.cam.fov,
        robot_camera.cam.aspect,
        10.0, // far
        0.1, // near
    );

    const camera_eid = try ecs.spawn(robot_camera);
    const frustum_eid = try ecs.spawn(robot_frustum);

    // Position camera above the robot
    if (ecs.transform_components.get(camera_eid)) |cam_transform| {
        cam_transform.setPosition(0, 2.0 * config.scale, 3.0 * config.scale);
        // Camera will look forward by default
    }

    // Set up camera hierarchy: root -> camera -> frustum
    try ecs.transform_system.addChild(root_eid, camera_eid);
    try ecs.transform_system.addChild(camera_eid, frustum_eid);

    std.debug.print("Created robot with compound collision: root={d}, model_parts={d}, camera={d}\n", .{
        root_eid.id, model_entities.len, camera_eid.id,
    });

    return RobotComponents{
        .root = root_eid,
        .model_entities = model_entities,
        .camera = camera_eid,
        .frustum = frustum_eid,
    };
}

/// Create robot model entities programmatically (simulating GLTF structure)
fn createRobotModelEntities(alloc: std.mem.Allocator, resource_manager: *ResourceManager, config: RobotConfig) !*GLTF.ModelResource {
    std.debug.print("Creating robot model entities programmatically with scale {d:.2}\n", .{config.scale});

    // Create meshes for different robot parts
    const torso_mesh = try createTorsoMesh(alloc, config.scale);
    const head_mesh = try createHeadMesh(alloc, config.scale);
    const arm_mesh = try createArmMesh(alloc, config.scale);
    const hand_mesh = try createHandMesh(alloc, config.scale);

    // Register meshes in resource manager
    try resource_manager.meshes.put(try alloc.dupe(u8, "robot_torso"), .{ .mesh = torso_mesh, .instance_count = 1 });
    try resource_manager.meshes.put(try alloc.dupe(u8, "robot_head"), .{ .mesh = head_mesh, .instance_count = 1 });
    try resource_manager.meshes.put(try alloc.dupe(u8, "robot_arm"), .{ .mesh = arm_mesh, .instance_count = 1 });
    try resource_manager.meshes.put(try alloc.dupe(u8, "robot_hand"), .{ .mesh = hand_mesh, .instance_count = 1 });

    // Create materials
    const torso_material = createMaterial(config.body_color);
    const head_material = createMaterial(config.head_color);
    const arm_material = createMaterial(config.arm_color);

    try resource_manager.loadMaterial(try alloc.dupeZ(u8, "robot_torso_material"), torso_material, null);
    try resource_manager.loadMaterial(try alloc.dupeZ(u8, "robot_head_material"), head_material, null);
    try resource_manager.loadMaterial(try alloc.dupeZ(u8, "robot_arm_material"), arm_material, null);

    // Create model entities array (simulating GLTF node structure)
    var entities = try alloc.alloc(GLTF.ModelResource.EntityInfo, 5);

    // Torso (root/body)
    entities[0] = GLTF.ModelResource.EntityInfo{
        .name = try alloc.dupeZ(u8, "Torso"),
        .mesh_name = try alloc.dupeZ(u8, "robot_torso"),
        .material_name = try alloc.dupeZ(u8, "robot_torso_material"),
        .local_transformation = Math.Mat4.identity(), // At origin
        .translation = .{ 0, 0, 0 },
        .rotation = null,
        .scale = .{ config.scale, config.scale, config.scale },
        .parent_idx = null,
        .children = try alloc.alloc(usize, 0),
    };

    // Head (above torso)
    const head_transform = Math.Mat4.translation(0, 1.5 * config.scale, 0);
    entities[1] = GLTF.ModelResource.EntityInfo{
        .name = try alloc.dupeZ(u8, "Head"),
        .mesh_name = try alloc.dupeZ(u8, "robot_head"),
        .material_name = try alloc.dupeZ(u8, "robot_head_material"),
        .local_transformation = head_transform,
        .translation = .{ 0, 1.5 * config.scale, 0 },
        .rotation = null,
        .scale = .{ config.scale, config.scale, config.scale },
        .parent_idx = 0,
        .children = try alloc.alloc(usize, 0),
    };

    // Left arm
    const left_arm_transform = Math.Mat4.translation(-1.2 * config.scale, 0.5 * config.scale, 0);
    entities[2] = GLTF.ModelResource.EntityInfo{
        .name = try alloc.dupeZ(u8, "LeftArm"),
        .mesh_name = try alloc.dupeZ(u8, "robot_arm"),
        .material_name = try alloc.dupeZ(u8, "robot_arm_material"),
        .local_transformation = left_arm_transform,
        .translation = .{ -1.2 * config.scale, 0.5 * config.scale, 0 },
        .rotation = null,
        .scale = .{ config.scale, config.scale, config.scale },
        .parent_idx = 0,
        .children = try alloc.dupe(usize, &[_]usize{4}),
    };

    // Right arm
    const right_arm_transform = Math.Mat4.translation(1.2 * config.scale, 0.5 * config.scale, 0);
    entities[3] = GLTF.ModelResource.EntityInfo{
        .name = try alloc.dupeZ(u8, "RightArm"),
        .mesh_name = try alloc.dupeZ(u8, "robot_arm"),
        .material_name = try alloc.dupeZ(u8, "robot_arm_material"),
        .local_transformation = right_arm_transform,
        .translation = .{ 1.2 * config.scale, 0.5 * config.scale, 0 },
        .rotation = null,
        .scale = .{ config.scale, config.scale, config.scale },
        .parent_idx = 0,
        .children = try alloc.alloc(usize, 0),
    };

    // Left hand (relative to left arm)
    const left_hand_transform = Math.Mat4.translation(0, -0.8 * config.scale, 0); // Relative to arm
    entities[4] = GLTF.ModelResource.EntityInfo{
        .name = try alloc.dupeZ(u8, "LeftHand"),
        .mesh_name = try alloc.dupeZ(u8, "robot_hand"),
        .material_name = try alloc.dupeZ(u8, "robot_arm_material"),
        .local_transformation = left_hand_transform,
        .translation = .{ 0, -0.8 * config.scale, 0 }, // Relative to arm
        .rotation = null,
        .scale = .{ config.scale, config.scale, config.scale },
        .parent_idx = 2,
        .children = try alloc.alloc(usize, 0),
    };

    std.debug.print("Created {d} robot model entities with compound structure\n", .{entities.len});

    const model_resource = try alloc.create(GLTF.ModelResource);
    model_resource.* = GLTF.ModelResource{
        .model_id = try alloc.dupeZ(u8, "programmatic_robot"),
        .entities = entities,
        .allocator = alloc,
    };

    return model_resource;
}

/// Create PBR material with given color
fn createMaterial(color: [4]f32) ResourceManager.MaterialVariant {
    return ResourceManager.MaterialVariant{
        .PBR = ResourceManager.Material(.PBR){
            .data = .{
                .baseColorFactor = color,
                .metallicFactor = 0.1,
                .roughnessFactor = 0.8,
            },
        },
    };
}

/// Create torso mesh (rectangular prism)
fn createTorsoMesh(alloc: std.mem.Allocator, scale: f32) !*Mesh {
    const width = 1.0 * scale;
    const height = 1.5 * scale;
    const depth = 0.6 * scale;

    return createBoxMesh(alloc, width, height, depth, .{ 0.2, 0.3, 0.8 });
}

/// Create head mesh (smaller cube)
fn createHeadMesh(alloc: std.mem.Allocator, scale: f32) !*Mesh {
    const size = 0.6 * scale;
    return createBoxMesh(alloc, size, size, size, .{ 0.1, 0.8, 0.1 });
}

/// Create arm mesh (elongated rectangular prism)
fn createArmMesh(alloc: std.mem.Allocator, scale: f32) !*Mesh {
    const width = 0.3 * scale;
    const height = 1.0 * scale;
    const depth = 0.3 * scale;

    return createBoxMesh(alloc, width, height, depth, .{ 0.8, 0.2, 0.2 });
}

/// Create hand mesh (small cube)
fn createHandMesh(alloc: std.mem.Allocator, scale: f32) !*Mesh {
    const size = 0.4 * scale;
    return createBoxMesh(alloc, size, size, size, .{ 0.8, 0.2, 0.2 });
}

/// Generic box mesh creation helper
fn createBoxMesh(alloc: std.mem.Allocator, width: f32, height: f32, depth: f32, color: [3]f32) !*Mesh {
    const hw = width / 2.0;
    const hh = height / 2.0;
    const hd = depth / 2.0;

    // Define 8 vertices of a box
    const vertices = [_]Mesh.Vertex{
        // Front face
        .{ .position = .{ -hw, -hh, hd }, .color = color, .normal = .{ 0, 0, 1 }, .texture = .{ 0, 0 } },
        .{ .position = .{ hw, -hh, hd }, .color = color, .normal = .{ 0, 0, 1 }, .texture = .{ 1, 0 } },
        .{ .position = .{ hw, hh, hd }, .color = color, .normal = .{ 0, 0, 1 }, .texture = .{ 1, 1 } },
        .{ .position = .{ -hw, hh, hd }, .color = color, .normal = .{ 0, 0, 1 }, .texture = .{ 0, 1 } },

        // Back face
        .{ .position = .{ -hw, -hh, -hd }, .color = color, .normal = .{ 0, 0, -1 }, .texture = .{ 1, 0 } },
        .{ .position = .{ -hw, hh, -hd }, .color = color, .normal = .{ 0, 0, -1 }, .texture = .{ 1, 1 } },
        .{ .position = .{ hw, hh, -hd }, .color = color, .normal = .{ 0, 0, -1 }, .texture = .{ 0, 1 } },
        .{ .position = .{ hw, -hh, -hd }, .color = color, .normal = .{ 0, 0, -1 }, .texture = .{ 0, 0 } },

        // Left face
        .{ .position = .{ -hw, -hh, -hd }, .color = color, .normal = .{ -1, 0, 0 }, .texture = .{ 0, 0 } },
        .{ .position = .{ -hw, -hh, hd }, .color = color, .normal = .{ -1, 0, 0 }, .texture = .{ 1, 0 } },
        .{ .position = .{ -hw, hh, hd }, .color = color, .normal = .{ -1, 0, 0 }, .texture = .{ 1, 1 } },
        .{ .position = .{ -hw, hh, -hd }, .color = color, .normal = .{ -1, 0, 0 }, .texture = .{ 0, 1 } },

        // Right face
        .{ .position = .{ hw, -hh, -hd }, .color = color, .normal = .{ 1, 0, 0 }, .texture = .{ 1, 0 } },
        .{ .position = .{ hw, hh, -hd }, .color = color, .normal = .{ 1, 0, 0 }, .texture = .{ 1, 1 } },
        .{ .position = .{ hw, hh, hd }, .color = color, .normal = .{ 1, 0, 0 }, .texture = .{ 0, 1 } },
        .{ .position = .{ hw, -hh, hd }, .color = color, .normal = .{ 1, 0, 0 }, .texture = .{ 0, 0 } },

        // Top face
        .{ .position = .{ -hw, hh, -hd }, .color = color, .normal = .{ 0, 1, 0 }, .texture = .{ 0, 1 } },
        .{ .position = .{ -hw, hh, hd }, .color = color, .normal = .{ 0, 1, 0 }, .texture = .{ 0, 0 } },
        .{ .position = .{ hw, hh, hd }, .color = color, .normal = .{ 0, 1, 0 }, .texture = .{ 1, 0 } },
        .{ .position = .{ hw, hh, -hd }, .color = color, .normal = .{ 0, 1, 0 }, .texture = .{ 1, 1 } },

        // Bottom face
        .{ .position = .{ -hw, -hh, -hd }, .color = color, .normal = .{ 0, -1, 0 }, .texture = .{ 0, 0 } },
        .{ .position = .{ hw, -hh, -hd }, .color = color, .normal = .{ 0, -1, 0 }, .texture = .{ 1, 0 } },
        .{ .position = .{ hw, -hh, hd }, .color = color, .normal = .{ 0, -1, 0 }, .texture = .{ 1, 1 } },
        .{ .position = .{ -hw, -hh, hd }, .color = color, .normal = .{ 0, -1, 0 }, .texture = .{ 0, 1 } },
    };

    // Define indices for triangles (2 triangles per face, 6 faces)
    const indices = [_]u32{
        // Front face
        0,  1,  2,  2,  3,  0,
        // Back face
        4,  5,  6,  6,  7,  4,
        // Left face
        8,  9,  10, 10, 11, 8,
        // Right face
        12, 13, 14, 14, 15, 12,
        // Top face
        16, 17, 18, 18, 19, 16,
        // Bottom face
        20, 21, 22, 22, 23, 20,
    };

    // Allocate and copy data
    const vertex_data = try alloc.dupe(Mesh.Vertex, &vertices);
    const index_data = try alloc.dupe(u32, &indices);

    return Mesh.init(alloc, vertex_data, index_data, Mesh.gen_draw(.triangles));
}
