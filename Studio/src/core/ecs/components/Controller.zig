// src/ecs/components/Controller.zig
const std = @import("std");
const Core = @import("../Core.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const Globals = @import("Globals.zig");
const Transform = @import("../components/Transform.zig");
const Physics = @import("../components/Physics.zig");
const Collisions = @import("Collisions.zig");
const ECSManager = @import("../ECSManager.zig");
const GlobalsComponent = Globals.GlobalsComponent;
const PhysicsComponent = Physics.PhysicsComponent;
const TransformComponent = Transform.TransformComponent;

// Platform-independent key codes
pub const Key = enum(c_int) {
    // Letters
    A = 65,
    B = 66,
    C = 67,
    D = 68,
    E = 69,
    F = 70,
    G = 71,
    H = 72,
    I = 73,
    J = 74,
    K = 75,
    L = 76,
    M = 77,
    N = 78,
    O = 79,
    P = 80,
    Q = 81,
    R = 82,
    S = 83,
    T = 84,
    U = 85,
    V = 86,
    W = 87,
    X = 88,
    Y = 89,
    Z = 90,

    // Numbers
    @"0" = 48,
    @"1" = 49,
    @"2" = 50,
    @"3" = 51,
    @"4" = 52,
    @"5" = 53,
    @"6" = 54,
    @"7" = 55,
    @"8" = 56,
    @"9" = 57,

    // Special keys
    Space = 32,
    Escape = 256,
    Enter = 257,
    Tab = 258,
    Backspace = 259,
    Insert = 260,
    Delete = 261,
    Right = 262,
    Left = 263,
    Down = 264,
    Up = 265,
    LeftShift = 340,
    LeftControl = 341,
    LeftAlt = 342,
    RightBracket = 93,

    // Special value for unsupported keys
    Unregistered = -1,

    pub fn fromGLFW(glfw_key: c_int) Key {
        return std.meta.intToEnum(Key, glfw_key) catch return .Unregistered;
    }

    pub fn toGLFW(self: Key) c_int {
        return @intFromEnum(self);
    }
};

pub const KeyAction = enum(c_int) {
    Release = 0,
    Press = 1,
    Repeat = 2,
};

pub const KeyMod = struct {
    pub const Shift: c_int = 0x0001;
    pub const Control: c_int = 0x0002;
    pub const Alt: c_int = 0x0004;
    pub const Super: c_int = 0x0008;
};

// Input event passed through layers
pub const InputEvent = struct {
    key: Key,
    scancode: c_int,
    action: KeyAction,
    mods: c_int,
    dt: f32,
    consumed: bool = false,

    pub fn consume(self: *InputEvent) void {
        self.consumed = true;
    }

    pub fn processEvent(self: *InputEvent, controller: *const ControllerComponent) void {
        for (0..controller.binding_count) |i| {
            const binding = &controller.bindings[i];
            if (!binding.enabled) continue;

            if (binding.key == self.key and
                (binding.mods == 0 or (self.mods & binding.mods) == binding.mods))
            {
                binding.handler(self, binding.context);
                if (self.consumed) return;
            }
        }

        // Opaque layers consume all input even if no handler matched
        if (controller._opaque) {
            self.consume();
        }
    }
};

pub const MouseButton = enum(c_int) {
    left = 0,
    right = 1,
    middle = 2,
};

pub const MouseAction = enum(c_int) {
    release = 0,
    press = 1,
};

pub const MouseEvent = struct {
    x: f32, // Absolute position
    y: f32,
    dx: f32, // Delta movement
    dy: f32,
    button: ?MouseButton = null, // For button events
    action: ?MouseAction = null, // For button events
    mods: c_int = 0, // Modifier keys (GLFW_MOD_SHIFT, GLFW_MOD_CONTROL, etc.)
    scroll_x: f32 = 0, // Horizontal scroll
    scroll_y: f32 = 0, // Vertical scroll
    dt: f32,
    consumed: bool = false,

    pub fn consume(self: *MouseEvent) void {
        self.consumed = true;
    }

    pub fn processEvent(self: *MouseEvent, controller: *const ControllerComponent) void {
        if (controller.mouse_handler) |handler| {
            handler(self, controller.mouse_context);
        }

        if (controller._opaque and !self.consumed) {
            self.consume();
        }
    }
};

pub const Event = union(enum) {
    key: InputEvent,
    mouse: MouseEvent,

    pub fn isConsumed(self: *const Event) bool {
        return switch (self.*) {
            .key => |*event| event.consumed,
            .mouse => |*event| event.consumed,
        };
    }

    pub fn processEvent(self: *Event, controller: *const ControllerComponent) void {
        switch (self.*) {
            .key => |*key_event| key_event.processEvent(controller),
            .mouse => |*mouse_event| mouse_event.processEvent(controller),
        }
    }
};

// Handler signatures with context
pub const KeyHandler = *const fn (event: *InputEvent, context: ?*anyopaque) void;
pub const MouseHandler = *const fn (event: *MouseEvent, context: ?*anyopaque) void;

// Binding types for different input handling
pub const BindingType = enum {
    Discrete, // Called only on press/release events (toggles, single actions)
    Continuous, // Called every frame while held (movement, camera look)
};

// Individual key binding
pub const KeyBinding = struct {
    key: Key,
    mods: c_int = 0,
    handler: KeyHandler,
    context: ?*anyopaque = null,
    enabled: bool = true,
    binding_type: BindingType = .Discrete, // Default to discrete for backward compatibility
};

// Layer types for categorization
pub const ControllerType = enum(u8) {
    Entity = 0, // Highest priority (0-63)
    Modal = 1, // Dialogs, menus (64-127)
    Tab = 2, // Tab-specific (128-191)
    Application = 3, // Global bindings (192-255)

    pub fn basePriority(self: ControllerType) u8 {
        return switch (self) {
            .Entity => 0,
            .Modal => 64,
            .Tab => 128,
            .Application => 192,
        };
    }
};

pub const ControllerComponent = struct {
    const MAX_LAYERS = 16;
    const MAX_BINDINGS_PER_LAYER = 64;
    const KEY_COUNT = 1024;

    id: u32,
    name: [32]u8,
    name_len: usize,
    layer_type: ControllerType,
    priority: u8, // Lower = higher priority
    enabled: bool = true,
    _opaque: bool = false, // If true, consumes all input

    // Bindings storage (no allocation)
    bindings: [MAX_BINDINGS_PER_LAYER]KeyBinding = undefined,
    binding_count: usize = 0,

    // Mouse handler
    mouse_handler: ?MouseHandler = null,
    mouse_context: ?*anyopaque = null,

    // Optional entity association
    entity_id: ?Core.EntityID = null,

    pub fn init(id: u32, name: []const u8, layer_type: ControllerType) ControllerComponent {
        var layer = ControllerComponent{
            .id = id,
            .name = undefined,
            .name_len = @min(name.len, 32),
            .layer_type = layer_type,
            .priority = layer_type.basePriority(),
        };
        @memcpy(layer.name[0..layer.name_len], name[0..layer.name_len]);
        return layer;
    }

    pub fn addBinding(self: *ControllerComponent, binding: KeyBinding) !void {
        if (self.binding_count >= MAX_BINDINGS_PER_LAYER) {
            return error.LayerBindingsFull;
        }
        self.bindings[self.binding_count] = binding;
        self.binding_count += 1;
    }

    pub fn removeBinding(self: *ControllerComponent, key: Key, mods: c_int) bool {
        var write_idx: usize = 0;
        var found = false;

        for (0..self.binding_count) |read_idx| {
            const binding = self.bindings[read_idx];
            if (binding.key != key or binding.mods != mods) {
                self.bindings[write_idx] = binding;
                write_idx += 1;
            } else {
                found = true;
            }
        }

        self.binding_count = write_idx;
        return found;
    }

    pub fn clearBindings(self: *ControllerComponent) void {
        self.binding_count = 0;
        self.mouse_handler = null;
        self.mouse_context = null;
    }

    pub fn setMouseHandler(self: *ControllerComponent, handler: MouseHandler, context: ?*anyopaque) void {
        self.mouse_handler = handler;
        self.mouse_context = context;
    }

    pub fn attach(self: *ControllerComponent, ecs: *ECSManager, entity_id: Core.EntityID) !void {
        self.entity_id = entity_id;

        // Set ECSManager as context for all bindings that don't have a specific context
        for (0..self.binding_count) |i| {
            if (self.bindings[i].context == null) {
                self.bindings[i].context = @ptrCast(ecs);
            }
        }

        // Set ECSManager as mouse context if not set
        if (self.mouse_handler != null and self.mouse_context == null) {
            self.mouse_context = @ptrCast(ecs);
        }

        // Add to the controller components sparse set
        try ecs.controller_components.add(entity_id, self.*);

        // Register with control system based on controller type
        ecs.control_system.setActiveLayer(self.layer_type, entity_id);
    }
};

pub const ControlSystem = struct {
    const Self = @This();

    world: *Core.World,
    globals: *GlobalsComponent,
    controller_components: *SparseSet(ControllerComponent),
    // Layer management - only one active controller per type
    active_layers: [4]?Core.EntityID = .{ null, null, null, null }, // Entity, Modal, Tab, Application

    // Entity selection state
    selected_entity: ?Core.EntityID = null,
    active_tab: enum { scene, paths } = .scene,

    // Key state tracking for continuous input
    key_states: [1024]bool = .{false} ** 1024,

    pub fn init(
        world: *Core.World,
        globals: *GlobalsComponent,
        controller_components: *SparseSet(ControllerComponent),
        ecs: *ECSManager,
    ) !Self {
        var self = Self{
            .world = world,
            .globals = globals,
            .controller_components = controller_components,
        };

        // Setup global controller
        var global_controller = Globals.GlobalController.createComponent();
        const global_entity_id = try world.createEntity();
        try global_controller.attach(ecs, global_entity_id);
        self.setActiveLayer(.Application, global_entity_id);

        // Setup scene controller
        var scene_controller = SceneController.createComponent();
        const scene_entity_id = try world.createEntity();
        try scene_controller.attach(ecs, scene_entity_id);
        self.setActiveLayer(.Tab, scene_entity_id);

        return self;
    }

    pub fn setActiveLayer(self: *Self, layer_type: ControllerType, entity_id: ?Core.EntityID) void {
        const type_index = @intFromEnum(layer_type);
        self.active_layers[type_index] = entity_id;
    }

    pub fn setSelectedEntity(self: *Self, entity_id: ?Core.EntityID) void {
        self.selected_entity = entity_id;

        // Automatically activate entity controller for selected entity
        const eid = entity_id orelse {
            self.setActiveLayer(.Entity, null);
            return;
        };

        const ctrl = self.controller_components.get(eid) orelse return;
        if (ctrl.layer_type != .Entity) return;

        self.setActiveLayer(.Entity, eid);
    }

    pub fn setActiveTab(self: *Self, tab: enum { scene, paths }) void {
        self.active_tab = tab;

        // Find and activate the appropriate tab controller
        var iterator = self.controller_components.iterator();
        while (iterator.next()) |entry| {
            const ctrl = entry.component;
            if (ctrl.layer_type != .Tab) continue;

            const controller_name = ctrl.name[0..ctrl.name_len];
            const should_activate = switch (tab) {
                .scene => std.mem.eql(u8, controller_name, "Scene"),
                .paths => std.mem.eql(u8, controller_name, "Paths"),
            };

            if (should_activate) {
                self.setActiveLayer(.Tab, entry.entity);
                return;
            }
        }

        // No matching tab controller found, deactivate tab layer
        self.setActiveLayer(.Tab, null);
    }

    fn processInputThroughLayers(self: *Self, event: *Event) void {
        // Process layers in priority order: Entity -> Modal -> Tab -> Application
        for (self.active_layers) |maybe_eid| {
            const eid = maybe_eid orelse continue;
            const ctrl = self.controller_components.get(eid) orelse continue;
            if (!ctrl.enabled) continue;

            event.processEvent(ctrl);
            if (event.isConsumed()) return;
        }
    }

    // Public method for GlobalsSystem to forward events (discrete actions only)
    pub fn handleEvent(self: *Self, event: *Event) void {
        self.processInputThroughLayers(event);
    }

    // Update key states from GLFW (called from GlobalsSystem key callback)
    pub fn updateKeyState(self: *Self, glfw_key: c_int, pressed: bool) void {
        if (glfw_key >= 0 and glfw_key < 1024) {
            self.key_states[@intCast(glfw_key)] = pressed;
        }
    }

    // Process continuous input every frame (called from ECS update loop)
    pub fn updateContinuous(self: *Self) void {
        // Check each active layer for continuous bindings
        const dt = self.globals.dt;
        // Create a synthetic input event for continuous processing
        var event = InputEvent{
            .key = undefined,
            .scancode = 0,
            .action = .Press, // Treat as continuous press
            .mods = 0,
            .dt = @floatCast(dt), // Use frame dt for smooth movement
            .consumed = false,
        };

        for (self.active_layers) |eid_opt| {
            const eid = eid_opt orelse continue;
            const ctrl = self.controller_components.get(eid) orelse continue;
            if (!ctrl.enabled) continue;

            // Process continuous bindings for this controller
            for (0..ctrl.binding_count) |i| {
                const binding = &ctrl.bindings[i];
                if (!binding.enabled or binding.binding_type != .Continuous) continue;

                // Check if key is currently held down
                const glfw_key = binding.key.toGLFW();
                if (glfw_key >= 0 and glfw_key < 1024 and self.key_states[@intCast(glfw_key)]) {

                    // Call the handler
                    event.key = binding.key;
                    binding.handler(&event, binding.context);

                    // If this binding consumed the input, don't process lower priority layers
                    // if (event.consumed) continue;
                }
            }
        }
    }
};

// Scene tab controller
pub const SceneController = struct {
    pub fn createComponent() ControllerComponent {
        var controller = ControllerComponent.init(1, "Scene", .Tab);

        // V - Toggle viewport
        controller.addBinding(.{
            .key = .V,
            .handler = handleToggleViewport,
            .context = null,
        }) catch unreachable;

        // P - Pause simulation
        controller.addBinding(.{
            .key = .P,
            .handler = handlePause,
            .context = null,
        }) catch unreachable;

        // R - Reset scene
        controller.addBinding(.{
            .key = .R,
            .handler = handleReset,
            .context = null,
        }) catch unreachable;

        return controller;
    }

    fn handleToggleViewport(event: *InputEvent, context: ?*anyopaque) void {
        if (event.action != .Press) return; // Only on PRESS

        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        toggleViewport(ecs);
        event.consume();
    }

    fn handlePause(event: *InputEvent, context: ?*anyopaque) void {
        if (event.action != .Press) return; // Only on PRESS

        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        togglePause(ecs);
        event.consume();
    }

    fn handleReset(event: *InputEvent, context: ?*anyopaque) void {
        if (event.action != .Press) return; // Only on PRESS

        const ecs = @as(*ECSManager, @ptrCast(@alignCast(context.?)));
        requestReset(ecs);
        event.consume();
    }

    /// Cycle through enabled viewports and switch both the active camera
    /// and the active controller.
    fn toggleViewport(ecs: *ECSManager) void {
        const globals_system = ecs.globals_system;
        const vps = globals_system.viewport_system.viewports;
        const cam_sys = globals_system.camera_system;
        const ctrl_sys = globals_system.control_system;

        var target_cam_eid: Core.EntityID = blk: {
            // No camera yet → just pick the first viewport entity
            if (cam_sys.active_camera_eid) |eid| break :blk eid;

            // Find first enabled viewport with a camera
            var vps_iter = vps.iterator();
            while (vps_iter.next()) |entry| {
                const vp_component = entry.component;
                if (!vp_component.vp.enabled) continue;
                if (ecs.camera_components.get(entry.entity_id)) |_| {
                    break :blk entry.entity_id;
                }
            }
            return; // No cameras found
        };

        // Cycle through viewports to find the next enabled camera
        var found_current = false;
        var vps_iter = vps.iterator();
        while (vps_iter.next()) |entry| {
            const vp_component = entry.component;
            if (!vp_component.vp.enabled) continue;
            if (ecs.camera_components.get(entry.entity_id) == null) continue;

            if (found_current) {
                target_cam_eid = entry.entity_id;
                break;
            }
            if (entry.entity_id.id == target_cam_eid.id) {
                found_current = true;
            }
        }

        // If we reached the end, wrap to the first camera
        if (found_current and cam_sys.active_camera_eid != null and cam_sys.active_camera_eid.?.id == target_cam_eid.id) {
            vps_iter = vps.iterator();
            while (vps_iter.next()) |entry| {
                const vp_component = entry.component;
                if (!vp_component.vp.enabled) continue;
                if (ecs.camera_components.get(entry.entity_id)) |_| {
                    target_cam_eid = entry.entity_id;
                    break;
                }
            }
        }

        // Switch camera and controller
        cam_sys.set_active(target_cam_eid);

        // Find associated controller - check camera entity first, then parent
        var target_ctrl_eid: ?Core.EntityID = null;

        // First check if camera entity itself has a controller
        if (ecs.controller_components.get(target_cam_eid)) |_| {
            target_ctrl_eid = target_cam_eid;
        } else {
            // Check if camera has a parent with a controller (drone case)
            if (ecs.transform_system.getParent(target_cam_eid)) |parent_eid| {
                if (ecs.controller_components.get(parent_eid)) |_| {
                    target_ctrl_eid = parent_eid;
                }
            }
        }

        std.debug.print("New active cam: {any}\n", .{target_cam_eid.id});
        std.debug.print("New active ctrl: {any}\n", .{if (target_ctrl_eid) |eid| eid.id else null});
        ctrl_sys.setSelectedEntity(target_ctrl_eid);
    }

    fn togglePause(ecs: *ECSManager) void {
        const globals_system = ecs.globals_system;
        globals_system.globals.paused = !globals_system.globals.paused;

        // Also pause/unpause physics simulation
        if (globals_system.collision_system) |collision_system| {
            collision_system.setPhysicsPaused(globals_system.globals.paused);
        }

        std.debug.print("Toggled pause: {}\n", .{globals_system.globals.paused});
    }

    fn requestReset(ecs: *ECSManager) void {
        ecs.globals_system.globals.reset_requested = true;
        std.debug.print("Reset requested!\n", .{});
    }
};
