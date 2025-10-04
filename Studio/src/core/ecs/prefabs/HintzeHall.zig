const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const ECSManager = @import("../ECSManager.zig");
const Transform = @import("../components/Transform.zig");
const Collisions = @import("../components/Collisions.zig");

pub fn spawn(
    alloc: std.mem.Allocator,
    ecs: *ECSManager,
) !Core.EntityID {
    const resource_manager = ecs.world.resource_manager;

    // Load the GLTF model
    const hintze_hall_resource = try resource_manager.loadGLTFModelCached(
        alloc,
        "assets/hintze_hall/scene.gltf",
    );

    // Create visual model entities
    var hintze_hall_result = try ecs.createEntitiesFromModel(hintze_hall_resource);
    const hintze_hall_entity = hintze_hall_result.root_entity;
    defer hintze_hall_result.entity_map.deinit();

    // Create static collision for the hall using triangle mesh
    const hall_collider = try Collisions.ColliderComponent.initFromModel(
        alloc,
        hintze_hall_resource,
        .{ .TriangleMesh = .{} },
        ecs.world.resource_manager,
    );

    // Create static physics body (mass = 0.0 for static)
    const hall_body = Collisions.RigidBodyComponent.init(0.0, hall_collider.bullet_shape.?);

    // Create root entity with physics components
    const root_tf = Transform.TransformComponent.init(alloc);
    const root_eid = ecs.spawn(.{ root_tf, hall_body, hall_collider }) catch |err| {
        std.debug.print("Error spawning HintzeHall... => {}\n", .{err});
        @panic("Failed to spawn Hintze Hall");
    };

    // Attach the visual model as a child
    ecs.transform_system.addChild(root_eid, hintze_hall_entity) catch |err| {
        std.debug.print("Failed to add child to tf in Hintze Hall... => {}\n", .{err});
        @panic("Failed to add child to transform component in hintze hall");
    };

    return root_eid;
}
