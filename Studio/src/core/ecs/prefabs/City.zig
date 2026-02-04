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
    const city_resource = try resource_manager.loadGLTFModelCached(
        alloc,
        "assets/cyberpunk_city/scene.gltf",
    );

    // Create visual model entities
    var hintze_hall_result = try ecs.createEntitiesFromModel(city_resource);
    const hintze_hall_entity = hintze_hall_result.root_entity;
    defer hintze_hall_result.entity_map.deinit();

    // Create static collision for the hall using triangle mesh
    const hall_collider = try Collisions.ColliderComponent.initFromModel(
        alloc,
        city_resource,
        .{ .TriangleMesh = .{} },
        ecs.world.resource_manager,
    );

    // Create static physics body (mass = 0.0 for static)
    const hall_body = Collisions.RigidBodyComponent.init(0.0, hall_collider.bullet_shape.?);

    // Create root entity with physics components
    const root_tf = Transform.TransformComponent.init(alloc);
    const root_eid = try ecs.spawn(.{ root_tf, hall_body, hall_collider });

    // Attach the visual model as a child
    try ecs.transform_system.addChild(root_eid, hintze_hall_entity);

    return root_eid;
}
