// const std = @import("std");
// const Math = @import("../../Math.zig");
// const Core = @import("../Core.zig");
// const SparseSet = @import("../SparseSet.zig").SparseSet;
// const FlightController = @import("FlightController.zig");
// const gl = @import("../../bindings/gl.zig");

// const Vec3 = Math.Vec3;
// const glfw = gl.glfw;
// const InputState = FlightController.InputState;
// const InputParams = FlightController.InputParams;

// /// Flight input component that captures keyboard/mouse input
// pub const FlightInputComponent = struct {
//     const Self = @This();

//     // Configuration
//     params: InputParams = InputParams{},
//     entity_id: Core.EntityID = undefined,

//     // Current raw input state
//     input_state: InputState = InputState{},

//     pub fn init() Self {
//         return Self{
//             .params = InputParams{},
//             .input_state = InputState{},
//         };
//     }

//     pub fn attach(self: *Self, ecs: anytype, eid: Core.EntityID) !void {
//         self.entity_id = eid;
//         try ecs.flight_input_components.add(eid, self.*);
//     }

//     /// Update input state from keyboard/mouse (called by input system)
//     pub fn updateInputState(
//         self: *Self,
//         keys: []bool, // GLFW key state array
//         mouse_dx: f32, // Mouse movement since last frame
//         mouse_dy: f32,
//     ) void {
//         // Update key states
//         if (keys[glfw.GLFW_KEY_6]) {
//             self.input_state.arm = !self.input_state.arm;
//         }
//         self.input_state.throttle_up = keys[glfw.GLFW_KEY_W];
//         self.input_state.throttle_down = keys[glfw.GLFW_KEY_S];
//         self.input_state.yaw_left = keys[glfw.GLFW_KEY_A];
//         self.input_state.yaw_right = keys[glfw.GLFW_KEY_D];

//         // Update mouse state
//         self.input_state.mouse_dx = mouse_dx;
//         self.input_state.mouse_dy = mouse_dy;
//     }

//     /// Reset all inputs
//     pub fn reset(self: *Self) void {
//         self.input_state = InputState{};
//     }
// };

// /// Flight input system that processes inputs and sends commands to flight controllers
// pub const FlightInputSystem = struct {
//     const Self = @This();

//     allocator: std.mem.Allocator,

//     // References to other systems
//     flight_input_components: *SparseSet(FlightInputComponent),
//     flight_controller_system: ?*FlightController.FlightControllerSystem = null,

//     pub fn init(allocator: std.mem.Allocator, flight_input_components: *SparseSet(FlightInputComponent)) Self {
//         return Self{
//             .allocator = allocator,
//             .flight_input_components = flight_input_components,
//         };
//     }

//     pub fn deinit(self: *Self) void {
//         _ = self; // No cleanup needed for now
//     }

//     /// Set reference to flight controller system
//     pub fn setFlightControllerSystem(self: *Self, flight_controller_system: *FlightController.FlightControllerSystem) void {
//         self.flight_controller_system = flight_controller_system;
//     }

//     /// Update all flight input components
//     pub fn update(
//         self: *Self,
//         keys: []bool, // Global key state from GLFW
//         mouse_dx: f32, // Global mouse movement
//         mouse_dy: f32,
//         dt: f32,
//     ) void {
//         const fc_system = self.flight_controller_system orelse return;

//         // Process each flight input component
//         var input_iter = self.flight_input_components.iterator();
//         while (input_iter.next()) |entry| {
//             const input_component = entry.component;
//             const entity_id = entry.entity_id;

//             // Update raw input state
//             input_component.updateInputState(keys, mouse_dx, mouse_dy);

//             // Get the flight controller for this entity
//             if (fc_system.flight_controller_components.get(entity_id)) |flight_controller| {
//                 // Use the controller to process input and generate setpoints
//                 flight_controller.armed = input_component.input_state.arm;
//                 const setpoints = flight_controller.controller.processInput(
//                     input_component.input_state,
//                     input_component.params,
//                     dt,
//                 );

//                 // Send setpoints to the flight controller
//                 flight_controller.setControlSetpoints(setpoints);
//             }
//         }
//     }
// };
