const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const ECSManager = @import("../ECSManager.zig");
const Transform = @import("../components/Transform.zig");
const Renderer = @import("../components/Renderer.zig");
const Collisions = @import("../components/Collisions.zig");
const Mesh = @import("../../Mesh.zig");
const gl = @import("../../bindings/gl.zig");

const glad = gl.glad;

const Vec3 = Math.Vec3;

pub const GroundConfig = struct {
    size: f32 = 100.0, // Size of the ground plane (100x100 units)
    position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    color: [3]f32 = .{ 0.3, 0.7, 0.3 }, // Green ground
};

pub fn spawn(
    allocator: std.mem.Allocator,
    ecs: *ECSManager,
    config: GroundConfig,
) !Core.EntityID {
    // Create transform component
    var transform = Transform.TransformComponent.init(allocator);
    transform.setPosition(config.position[0], config.position[1], config.position[2]);

    // Generate plane mesh
    const plane_mesh = try generatePlaneMesh(allocator, config);

    // Create mesh name and register it with resource manager
    const mesh_name = try std.fmt.allocPrint(allocator, "ground_plane_{d}", .{config.size});
    defer allocator.free(mesh_name);

    const mesh_name_owned = try allocator.dupe(u8, mesh_name);
    try ecs.world.resource_manager.meshes.put(mesh_name_owned, .{ .mesh = plane_mesh, .instance_count = 0 });

    // Create renderer component
    var renderer = try Renderer.Renderable.init(allocator, mesh_name_owned);
    
    // Create a double-sided PBR material for the ground plane
    const ResourceManager = @import("../ResourceManager.zig");
    const ground_material = ResourceManager.MaterialVariant{ 
        .PBR = ResourceManager.Material(.PBR){
            .data = .{
                .doubleSided = true,
                .baseColorFactor = .{ config.color[0], config.color[1], config.color[2], 1.0 },
            },
        },
    };
    
    // Create unique material name based on config
    const material_name = try std.fmt.allocPrint(allocator, "ground_material_{d}_{d}_{d}", .{ @as(u32, @intFromFloat(config.color[0] * 1000)), @as(u32, @intFromFloat(config.color[1] * 1000)), @as(u32, @intFromFloat(config.color[2] * 1000)) });
    defer allocator.free(material_name);
    
    const material_name_owned = try allocator.dupeZ(u8, material_name);
    
    // Load the material into the resource manager
    try ecs.world.resource_manager.loadMaterial(material_name_owned, ground_material, null);
    
    try renderer.setMaterial(allocator, material_name_owned);

    // Create physics collider (box shape for ground plane) - made thicker to prevent tunneling
    const half_size = config.size / 2.0;
    const collider = try Collisions.ColliderComponent.init(allocator, .{ .Box = .{ .half_extents = .{ half_size, 1.0, half_size } } }, null);

    // Create static rigid body (mass = 0)
    const rigid_body = Collisions.RigidBodyComponent.init(0.0, collider.bullet_shape.?);

    // Spawn entity with all components
    const ground_entity = try ecs.spawn(.{ transform, renderer, collider, rigid_body });

    // Physics body creation now handles collision properties automatically in threaded physics

    std.debug.print("Created ground plane: size={d}, entity={d}\n", .{ config.size, ground_entity.id });

    return ground_entity;
}

fn generatePlaneMesh(allocator: std.mem.Allocator, config: GroundConfig) !*Mesh {
    const half_size = config.size / 2.0;

    // Create vertices for a plane (2 triangles forming a quad)
    var vertices = try allocator.alloc(Mesh.Vertex, 4);
    var indices = try allocator.alloc(u32, 6);

    // Define the 4 corners of the plane
    vertices[0] = Mesh.Vertex{
        .position = .{ -half_size, 0.0, -half_size },
        .color = config.color,
        .texture = .{ 0.0, 0.0 },
        .normal = .{ 0.0, 1.0, 0.0 },
        .tangent = .{ 1.0, 0.0, 0.0 },
    };
    vertices[1] = Mesh.Vertex{
        .position = .{ half_size, 0.0, -half_size },
        .color = config.color,
        .texture = .{ 1.0, 0.0 },
        .normal = .{ 0.0, 1.0, 0.0 },
        .tangent = .{ 1.0, 0.0, 0.0 },
    };
    vertices[2] = Mesh.Vertex{
        .position = .{ half_size, 0.0, half_size },
        .color = config.color,
        .texture = .{ 1.0, 1.0 },
        .normal = .{ 0.0, 1.0, 0.0 },
        .tangent = .{ 1.0, 0.0, 0.0 },
    };
    vertices[3] = Mesh.Vertex{
        .position = .{ -half_size, 0.0, half_size },
        .color = config.color,
        .texture = .{ 0.0, 1.0 },
        .normal = .{ 0.0, 1.0, 0.0 },
        .tangent = .{ 1.0, 0.0, 0.0 },
    };

    // Define indices for 2 triangles (counter-clockwise winding)
    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2; // First triangle
    indices[3] = 0;
    indices[4] = 2;
    indices[5] = 3; // Second triangle

    // Create mesh using the proper init function with triangle draw mode
    const mesh = try Mesh.init(allocator, vertices, indices, Mesh.gen_draw(glad.GL_TRIANGLES));
    
    return mesh;
}
