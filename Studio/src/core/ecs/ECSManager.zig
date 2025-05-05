// src/ecs/ECSManager.zig
const std = @import("std");
const Core = @import("Core.zig");
const SparseSet = @import("SparseSet.zig").SparseSet;
const ResourceManager = @import("ResourceManager.zig");
const Math = @import("../Math.zig");
const GLTFPaser = @import("../GLTF.zig");
const meta = std.meta;
const GLTF = GLTFPaser.GLTF;
const Quaternion = Math.Quaternion;

const Transform = @import("components/Transform.zig");
const Renderer = @import("components/Renderer.zig");
const Physics = @import("components/Physics.zig");
const Controller = @import("components/Controller.zig");
const Camera = @import("components/Camera.zig");
const Globals = @import("components/Globals.zig");
const Viewports = @import("components/Viewports.zig");

// Components
const ControllerComponent = Controller.ControllerComponent;
const TransformComponent = Transform.TransformComponent;
const PhysicsComponent = Physics.PhysicsComponent;
const Renderable = Renderer.Renderable;
const CameraComponent = Camera.CameraComponent;
const GlobalsComponent = Globals.GlobalsComponent;
const ViewportComponent = Viewports.ViewportComponent;

// Systems
const ControllerSytem = Controller.ControlSystem;
const TransformSystem = Transform.TransformSystem;
const PhysicsSystem = Physics.PhysicsSystem;
const RenderSystem = Renderer.RenderSystem;
const CameraSystem = Camera.CameraSystem;
const GlobalsSystem = Globals.GlobalsSystem;
const ViewportSystem = Viewports.ViewportSystem;

const Self = @This();

allocator: std.mem.Allocator,
world: *Core.World,

// Component storage
globals: *GlobalsComponent,
transform_components: SparseSet(TransformComponent),
camera_components: SparseSet(CameraComponent),
renderer_components: SparseSet(Renderable),
physics_components: SparseSet(PhysicsComponent),
controller_components: SparseSet(ControllerComponent),
viewport_components: SparseSet(ViewportComponent),

// Systems
globals_system: *GlobalsSystem,
transform_system: TransformSystem,
camera_system: CameraSystem,
render_system: RenderSystem,
physics_system: PhysicsSystem,
control_system: ControllerSytem,
viewport_system: ViewportSystem,

pub fn init(allocator: std.mem.Allocator) !*Self {
    const global_system = try GlobalsSystem.init(allocator, .{});

    const resource_manager = try ResourceManager.init(allocator);
    const world = try Core.World.init(allocator, resource_manager);

    var manager = try allocator.create(Self);

    manager.* = .{
        .allocator = allocator,
        .world = world,
        .globals = &global_system.globals,
        .transform_components = SparseSet(TransformComponent).init(allocator),
        .viewport_components = SparseSet(ViewportComponent).init(allocator),
        .camera_components = SparseSet(CameraComponent).init(allocator),
        .renderer_components = SparseSet(Renderable).init(allocator),
        .physics_components = SparseSet(PhysicsComponent).init(allocator),
        .controller_components = SparseSet(ControllerComponent).init(allocator),

        // Initialize systems
        .globals_system = global_system,
        .transform_system = TransformSystem.init(
            world,
            &manager.transform_components,
        ),
        .viewport_system = ViewportSystem.init(
            allocator,
            manager.globals,
            &manager.viewport_components,
        ),
        .camera_system = CameraSystem.init(
            world,
            &manager.camera_components,
            &manager.transform_components,
        ),
        .physics_system = PhysicsSystem.init(
            world,
            &manager.transform_components,
            &manager.physics_components,
        ),
        .control_system = ControllerSytem.init(
            world,
            manager.globals,
            &manager.transform_components,
            &manager.physics_components,
            &manager.controller_components,
        ),
        .render_system = try RenderSystem.init(
            allocator,
            world,
            manager.globals,
            &manager.transform_components,
            &manager.camera_components,
            &manager.viewport_components,
            &manager.renderer_components,
        ),
    };

    global_system.camera_system = &manager.camera_system;
    global_system.control_system = &manager.control_system;
    global_system.viewport_system = &manager.viewport_system;
    global_system.transform_system = &manager.transform_system;

    return manager;
}

pub fn deinit(self: *Self) void {
    // Deinit all components
    var transform_iter = self.transform_components.iterator();
    while (transform_iter.next()) |tuple| {
        tuple.component.deinit();
    }

    var controller_iter = self.controller_components.iterator();
    while (controller_iter.next()) |tuple| {
        tuple.component.deinit();
    }

    // Deinit component storage
    self.transform_components.deinit();
    self.renderer_components.deinit();
    self.physics_components.deinit();
    self.controller_components.deinit();

    // Deinit world and resource manager
    self.world.resource_manager.deinit();
    self.allocator.destroy(self.world.resource_manager);
    self.world.deinit();
    self.allocator.destroy(self.world);

    // Free self
    self.allocator.destroy(self);
}

pub fn update(self: *Self, time: f64) !void {
    if (self.globals.last_frame_time == 0) {
        self.globals.last_frame_time = time;
        self.globals.last_fps_time = time;
    }

    const dt = time - self.globals.last_frame_time;
    const dt_fps = time - self.globals.last_fps_time;
    self.globals.last_frame_time = time;
    self.globals.frame_count += 1;

    if (dt_fps >= 1) {
        self.globals.last_fps_time = time;
        self.globals.avg_fps = @as(f32, @floatFromInt(self.globals.frame_count));
        self.globals.frame_count = 0;

        std.debug.print("Avg FPS: {d:.2}\n", .{self.globals.avg_fps});
    }

    self.transform_system.update();
    self.control_system.update(dt);
    // self.physics_system.update(dt);
    try self.viewport_system.update();
    try self.render_system.update();
}

// Entity management methods
pub fn createEntity(self: *Self) !Core.EntityID {
    return self.world.createEntity();
}

pub fn destroyEntity(self: *Self, entity_id: Core.EntityID) void {
    // Remove all components first
    _ = self.transform_components.remove(entity_id);
    _ = self.renderer_components.remove(entity_id);
    _ = self.physics_components.remove(entity_id);
    _ = self.controller_components.remove(entity_id);

    // Then remove the entity from the world
    self.world.destroyEntity(entity_id);
}

// Component methods
pub fn addTransform(self: *Self, entity_id: Core.EntityID) !*TransformComponent {
    const transform = TransformComponent.init(self.allocator);
    try self.transform_components.add(entity_id, transform);
    return self.transform_components.get(entity_id).?;
}

pub fn addRenderer(self: *Self, entity_id: Core.EntityID, mesh_name: []const u8) !*Renderable {
    const renderer = try Renderable.init(self.allocator, mesh_name);
    try self.renderer_components.add(entity_id, renderer);
    return self.renderer_components.get(entity_id).?;
}

pub fn addPhysicsBody(self: *Self, entity_id: Core.EntityID, body_type: Physics.BodyType) !*PhysicsComponent {
    const physics = PhysicsComponent.init(body_type);
    try self.physics_components.add(entity_id, physics);
    return self.physics_components.get(entity_id).?;
}

pub fn addController(self: *Self, entity_id: Core.EntityID) !*ControllerComponent {
    const controller = ControllerComponent.init(self.allocator);
    try self.controller_components.add(entity_id, controller);
    return self.controller_components.get(entity_id).?;
}

// Transform methods
pub fn setParent(self: *Self, child_id: Core.EntityID, parent_id: Core.EntityID) !void {
    try self.transform_system.addChild(parent_id, child_id);
}

pub fn createEntitiesFromModel(self: *Self, model_resource: *GLTFPaser.ModelResource) !Core.EntityID {
    var entity_map = std.AutoHashMap(usize, Core.EntityID).init(self.allocator);
    defer entity_map.deinit();

    // Create ECS entity for every ModelResource.EntityInfo
    for (model_resource.entities, 0..) |node, idx| {
        const e_id = try self.createEntity();
        try entity_map.put(idx, e_id);

        // Add transform
        const transform = try self.addTransform(e_id);

        if (node.local_transformation) |local_transformation| {
            const trs = local_transformation.decomposeTRS();

            std.debug.print("Entity_Id: {d}\n", .{e_id.id});
            std.debug.print("Found Local Transform: {any}\n", .{local_transformation});
            std.debug.print("TRS: {any}\n", .{trs});

            transform.setPosition(trs.translation[0], trs.translation[1], trs.translation[2]);
            transform.setRotation(trs.rotation);
            transform.setScale(trs.scale[0], trs.scale[1], trs.scale[2]);
        } else {
            // If node has TRS
            if (node.translation) |t| {
                transform.setPosition(t[0], t[1], t[2]);
            }
            if (node.rotation) |r| {
                transform.setRotation(Quaternion.init(r[0], r[1], r[2], r[3]));
            }
            if (node.scale) |s| {
                transform.setScale(s[0], s[1], s[2]);
            }
        }

        // If there is a mesh_name, add a renderer
        if (node.mesh_name) |mesh_str| {
            const renderer = try self.addRenderer(e_id, mesh_str);
            // If material_name, bind material
            try renderer.setMaterial(self.allocator, node.material_name);
        }
    }

    // Hook up parent-child relationships <= TODO: Verify this works correctly
    for (model_resource.entities, 0..) |node, idx| {
        if (node.parent_idx) |p_idx| {
            if (entity_map.get(idx)) |child_eid| {
                if (entity_map.get(p_idx)) |parent_eid| {
                    try self.setParent(child_eid, parent_eid);
                }
            }
        }
    }

    // Return the ECS entity ID of the first node (root)
    return entity_map.get(0).?;
}

pub fn findControllerAncestor(self: *Self, start: Core.EntityID) ?Core.EntityID {
    var current = start;
    while (true) {
        if (self.controller_components.has(current))
            return current;

        const tf_opt = self.transform_components.get(current);
        if (tf_opt == null or tf_opt.?.parent == null)
            break;

        current = tf_opt.?.parent.?; // climb one level
    }

    return null;
}

pub fn spawn(self: *Self, bundle: anytype) !Core.EntityID {
    const T = @TypeOf(bundle);
    comptime switch (@typeInfo(T)) {
        .@"struct" => {}, // supported
        else => @compileError("spawn() expects a struct or tuple of components"),
    };

    const eid = try self.createEntity();

    inline for (meta.fields(T)) |fld| {
        const field_val = @field(bundle, fld.name);
        try attachComponent(self, eid, field_val);
    }
    return eid;
}

fn attachComponent(self: *Self, eid: Core.EntityID, comp_any: anytype) !void {
    // Pointer => use directly
    if (@typeInfo(@TypeOf(comp_any)) == .pointer) {
        try @field(@TypeOf(comp_any.*), "attach")(comp_any, self, eid);
        return;
    }

    // By value => make a copy on stack then attach
    var tmp = comp_any;
    try @field(@TypeOf(tmp), "attach")(&tmp, self, eid);
}
