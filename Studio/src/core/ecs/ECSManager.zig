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
const Recorder = @import("components/Recorder.zig");
const Collisions = @import("components/Collisions.zig");
const SharedMem = @import("components/SharedMem.zig");

// Components
const ControllerComponent = Controller.ControllerComponent;
const TransformComponent = Transform.TransformComponent;
const PhysicsComponent = Physics.PhysicsComponent;
const Renderable = Renderer.Renderable;
const CameraComponent = Camera.CameraComponent;
const GlobalsComponent = Globals.GlobalsComponent;
const ViewportComponent = Viewports.ViewportComponent;
const ColliderComponent = Collisions.ColliderComponent;
const RigidBodyComponent = Collisions.RigidBodyComponent;

// Systems
const ControllerSytem = Controller.ControlSystem;
const TransformSystem = Transform.TransformSystem;
const RenderSystem = Renderer.RenderSystem;
const CameraSystem = Camera.CameraSystem;
const GlobalsSystem = Globals.GlobalsSystem;
const ViewportSystem = Viewports.ViewportSystem;
const RecorderSystem = Recorder.RecorderSystem;
const CollisionSystem = Collisions.CollisionSystem;
const SharedMemSystem = SharedMem.SharedMemSystem;

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
collider_components: SparseSet(ColliderComponent),
rigid_body_components: SparseSet(RigidBodyComponent),

// Systems
globals_system: *GlobalsSystem,
transform_system: TransformSystem,
camera_system: CameraSystem,
render_system: RenderSystem,
control_system: ControllerSytem,
viewport_system: ViewportSystem,
recorder_system: *RecorderSystem,
collision_system: CollisionSystem,
shared_mem_system: SharedMemSystem,

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
        .collider_components = SparseSet(ColliderComponent).init(allocator),
        .rigid_body_components = SparseSet(RigidBodyComponent).init(allocator),

        // Initialize systems
        .globals_system = global_system,
        .recorder_system = try RecorderSystem.init(allocator),
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
        .control_system = ControllerSytem.init(
            world,
            manager.globals,
            &manager.transform_components,
            &manager.rigid_body_components,
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
        .collision_system = try CollisionSystem.init(
            allocator,
            world,
            &manager.transform_components,
            &manager.rigid_body_components,
            &manager.collider_components,
        ),
        .shared_mem_system = try SharedMemSystem.init(
            allocator,
            manager.globals,
            &manager.viewport_components,
        ),
    };

    global_system.camera_system = &manager.camera_system;
    global_system.control_system = &manager.control_system;
    global_system.viewport_system = &manager.viewport_system;
    global_system.transform_system = &manager.transform_system;

    // Link collision system to control system for physics-based movement
    manager.control_system.collision_system = &manager.collision_system;

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

    // Deinit systems
    self.collision_system.deinit();
    self.shared_mem_system.deinit();

    // Deinit world and resource manager
    self.world.resource_manager.deinit();
    self.allocator.destroy(self.world.resource_manager);
    self.world.deinit();
    self.allocator.destroy(self.world);

    // Free self
    self.allocator.destroy(self);
}

pub fn update(self: *Self, time: f64) !void {
    // Check for reset request first
    if (self.globals.reset_requested) {
        try self.resetToInitialState();
        self.globals.reset_requested = false;
        return;
    }

    if (self.globals.last_frame_time == 0) {
        self.globals.last_frame_time = time;
        self.globals.last_fps_time = time;
    }

    const dt = time - self.globals.last_frame_time;
    const dt_fps = time - self.globals.last_fps_time;
    self.globals.dt = dt;
    self.globals.last_frame_time = time;
    self.globals.frame_count += 1;

    if (dt_fps >= 1) {
        self.globals.last_fps_time = time;
        self.globals.avg_fps = @as(f32, @floatFromInt(self.globals.frame_count));
        self.globals.frame_count = 0;

        std.debug.print("Avg FPS: {d:.2}\n", .{self.globals.avg_fps});
    }

    // Add timing for system updates (only for first 10 frames to avoid spam)
    // const should_time = self.globals.frame_count <= 10;
    const should_time = false;
    const timer = if (should_time) std.time.Timer.start() catch unreachable else undefined;

    if (should_time) std.debug.print("Frame {d} system timings:\n", .{self.globals.frame_count});

    if (should_time) _ = timer.lap(); // Reset timer
    self.control_system.update(dt);
    if (should_time) std.debug.print("  Control system: {d:.2}ms\n", .{@as(f64, @floatFromInt(timer.lap())) / 1e6});

    self.transform_system.update();
    if (should_time) std.debug.print("  Transform system: {d:.2}ms\n", .{@as(f64, @floatFromInt(timer.lap())) / 1e6});

    try self.collision_system.update(@floatCast(dt));
    if (should_time) std.debug.print("  Collision system: {d:.2}ms\n", .{@as(f64, @floatFromInt(timer.lap())) / 1e6});

    try self.viewport_system.update();
    if (should_time) std.debug.print("  Viewport system: {d:.2}ms\n", .{@as(f64, @floatFromInt(timer.lap())) / 1e6});

    try self.render_system.update();
    if (should_time) std.debug.print("  Render system: {d:.2}ms\n", .{@as(f64, @floatFromInt(timer.lap())) / 1e6});

    try self.recorder_system.update(self);
    if (should_time) std.debug.print("  Recorder system: {d:.2}ms\n", .{@as(f64, @floatFromInt(timer.lap())) / 1e6});

    self.shared_mem_system.update();
    if (should_time) std.debug.print("  SharedMem system: {d:.2}ms\n", .{@as(f64, @floatFromInt(timer.lap())) / 1e6});
}

pub fn resetToInitialState(self: *Self) !void {
    std.debug.print("Resetting ECS to initial state...\n", .{});
    self.collision_system.resetAllDynamicBodies();
    std.debug.print("Reset complete!\n", .{});
}

// Entity management methods
pub fn createEntity(self: *Self) !Core.EntityID {
    return self.world.createEntity();
}

pub fn destroyEntity(self: *Self, entity_id: Core.EntityID) void {
    // Remove all components first
    _ = self.transform_components.remove(entity_id) catch false;
    _ = self.renderer_components.remove(entity_id) catch false;
    _ = self.physics_components.remove(entity_id) catch false;
    _ = self.controller_components.remove(entity_id) catch false;

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

pub fn createEntitiesFromModel(self: *Self, model_resource: *GLTFPaser.ModelResource) !struct { root_entity: Core.EntityID, entity_map: std.AutoHashMap(usize, Core.EntityID) } {
    var entity_map = std.AutoHashMap(usize, Core.EntityID).init(self.allocator);
    // Don't defer deinit - we're returning this


    // Create ECS entity for every ModelResource.EntityInfo
    for (model_resource.entities, 0..) |node, idx| {
        const e_id = try self.createEntity();
        try entity_map.put(idx, e_id);


        const transform = try self.addTransform(e_id);

        if (node.local_transformation) |local_transformation| {
            const trs = local_transformation.decomposeTRS();

            transform.setPosition(trs.translation[0], trs.translation[1], trs.translation[2]);
            transform.setRotation(trs.rotation);
            transform.setScale(trs.scale[0], trs.scale[1], trs.scale[2]);
        } else {
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
            if (node.material_name) |mat_str| {
                try renderer.setMaterial(self.allocator, mat_str);
            }
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

    return .{
        .root_entity = entity_map.get(0).?,
        .entity_map = entity_map,
    };
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
