const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const FlightController = @import("FlightController.zig");
const gl = @import("../../bindings/gl.zig");

const Vec3 = Math.Vec3;
const glfw = gl.glfw;

/// Flight input configuration parameters
pub const FlightInputParams = struct {
    // Input sensitivity
    throttle_sensitivity: f32 = 30.0, // N/s (thrust change rate) - increased for faster response
    yaw_sensitivity: f32 = 3.14, // rad/s per key press
    roll_pitch_sensitivity: f32 = 0.1, // rad/s per pixel of mouse movement - increased for better response

    // Input limits (safety bounds)
    max_roll_rate: f32 = 10.47, // rad/s (600 deg/s)
    max_pitch_rate: f32 = 10.47, // rad/s (600 deg/s)
    max_yaw_rate: f32 = 5.24, // rad/s (300 deg/s)
    max_thrust: f32 = 40.0, // Newtons (3g for 1kg drone) - increased for better performance

    // Low-pass filter parameters
    throttle_filter_tau: f32 = 0.05, // seconds (much faster throttle response)
    rate_filter_tau: f32 = 0.05, // seconds (smooth rate commands)
};

/// Current input state (raw from keyboard/mouse)
pub const InputState = struct {
    // Key states
    throttle_up: bool = false, // W key
    throttle_down: bool = false, // S key
    yaw_left: bool = false, // A key
    yaw_right: bool = false, // D key

    // Mouse movement (pixels per frame)
    mouse_dx: f32 = 0.0, // Roll input
    mouse_dy: f32 = 0.0, // Pitch input
};

/// Filtered control commands (output to flight controller)
pub const ControlCommands = struct {
    desired_thrust: f32 = 0.0, // Newtons
    desired_rates: [3]f32 = [3]f32{ 0, 0, 0 }, // [roll, pitch, yaw] rad/s
};

/// Flight input component that converts keyboard/mouse to flight setpoints
pub const FlightInputComponent = struct {
    const Self = @This();

    // Configuration
    params: FlightInputParams = FlightInputParams{},
    entity_id: Core.EntityID = undefined,

    // Current state
    input_state: InputState = InputState{},
    control_commands: ControlCommands = ControlCommands{},

    // Filter state (for smooth control)
    filtered_thrust: f32 = 0.0,
    filtered_rates: [3]f32 = [3]f32{ 0, 0, 0 },

    pub fn init() Self {
        return Self{};
    }

    pub fn attach(self: *Self, ecs: anytype, eid: Core.EntityID) !void {
        self.entity_id = eid;
        try ecs.flight_input_components.add(eid, self.*);
    }

    /// Update input state from keyboard/mouse (called by input system)
    pub fn updateInputState(
        self: *Self,
        keys: []bool, // GLFW key state array
        mouse_dx: f32, // Mouse movement since last frame
        mouse_dy: f32,
    ) void {

        // Update key states
        self.input_state.throttle_up = keys[glfw.GLFW_KEY_W];
        self.input_state.throttle_down = keys[glfw.GLFW_KEY_S];
        self.input_state.yaw_left = keys[glfw.GLFW_KEY_A];
        self.input_state.yaw_right = keys[glfw.GLFW_KEY_D];

        // Update mouse state
        self.input_state.mouse_dx = mouse_dx;
        self.input_state.mouse_dy = mouse_dy;
    }

    /// Process inputs and generate filtered control commands
    pub fn updateControlCommands(self: *Self, dt: f32) void {
        // Process throttle input (W/S keys with low-pass filter)
        var throttle_command: f32 = 0.0;
        if (self.input_state.throttle_up) {
            throttle_command += self.params.throttle_sensitivity * dt;
        }
        if (self.input_state.throttle_down) {
            throttle_command -= self.params.throttle_sensitivity * dt;
        }

        // Apply low-pass filter to throttle for smooth control
        const throttle_alpha = dt / (self.params.throttle_filter_tau + dt);
        const target_thrust = std.math.clamp(self.filtered_thrust + throttle_command, 0.0, self.params.max_thrust);
        self.filtered_thrust = self.filtered_thrust + throttle_alpha * (target_thrust - self.filtered_thrust);
        self.control_commands.desired_thrust = self.filtered_thrust;

        // Process rate commands (yaw from A/D keys, roll/pitch from mouse)
        var raw_rates = [3]f32{ 0, 0, 0 };

        // Roll from mouse X movement (right is positive roll)
        raw_rates[0] = self.input_state.mouse_dx * self.params.roll_pitch_sensitivity;

        // Pitch from mouse Y movement (up is negative pitch in NED)
        raw_rates[1] = -self.input_state.mouse_dy * self.params.roll_pitch_sensitivity;

        // Yaw from A/D keys
        if (self.input_state.yaw_left) {
            raw_rates[2] += self.params.yaw_sensitivity;
        }
        if (self.input_state.yaw_right) {
            raw_rates[2] -= self.params.yaw_sensitivity;
        }

        // Apply limits to rate commands
        raw_rates[0] = std.math.clamp(raw_rates[0], -self.params.max_roll_rate, self.params.max_roll_rate);
        raw_rates[1] = std.math.clamp(raw_rates[1], -self.params.max_pitch_rate, self.params.max_pitch_rate);
        raw_rates[2] = std.math.clamp(raw_rates[2], -self.params.max_yaw_rate, self.params.max_yaw_rate);

        // Apply low-pass filter to rate commands for smoothness
        const rate_alpha = dt / (self.params.rate_filter_tau + dt);
        for (0..3) |i| {
            self.filtered_rates[i] = self.filtered_rates[i] + rate_alpha * (raw_rates[i] - self.filtered_rates[i]);
        }
        self.control_commands.desired_rates = self.filtered_rates;
    }

    /// Get current control commands (used by flight controller)
    pub fn getControlCommands(self: *Self) ControlCommands {
        return self.control_commands;
    }

    /// Reset all inputs and filters (useful for mode changes)
    pub fn reset(self: *Self) void {
        self.input_state = InputState{};
        self.control_commands = ControlCommands{};
        self.filtered_thrust = 0.0;
        self.filtered_rates = [3]f32{ 0, 0, 0 };
    }
};

/// Flight input system that processes inputs and sends commands to flight controllers
pub const FlightInputSystem = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    flight_input_components: *SparseSet(FlightInputComponent),

    // References to other systems
    flight_controller_system: ?*FlightController.FlightControllerSystem = null,

    pub fn init(allocator: std.mem.Allocator, flight_input_components: *SparseSet(FlightInputComponent)) Self {
        return Self{
            .allocator = allocator,
            .flight_input_components = flight_input_components,
        };
    }

    pub fn deinit(self: *Self) void {
        _ = self; // No cleanup needed for now
    }

    /// Set reference to flight controller system
    pub fn setFlightControllerSystem(self: *Self, flight_controller_system: *FlightController.FlightControllerSystem) void {
        self.flight_controller_system = flight_controller_system;
    }

    /// Update all flight input components
    pub fn update(
        self: *Self,
        keys: []bool, // Global key state from GLFW
        mouse_dx: f32, // Global mouse movement
        mouse_dy: f32,
        dt: f32,
    ) void {

        // Process each flight input component
        var input_iter = self.flight_input_components.iterator();
        while (input_iter.next()) |entry| {
            const input_component = entry.component;
            const entity_id = entry.entity_id;

            input_component.updateInputState(keys, mouse_dx, mouse_dy);
            input_component.updateControlCommands(dt);

            self.sendCommandsToFlightController(entity_id, input_component.getControlCommands());
        }
    }

    /// Send control commands to the flight controller for a specific entity
    fn sendCommandsToFlightController(self: *Self, entity_id: Core.EntityID, commands: ControlCommands) void {
        if (self.flight_controller_system) |fc_system| {
            // Find the flight controller for this entity
            var fc_iter = fc_system.flight_controller_components.iterator();
            while (fc_iter.next()) |entry| {
                if (entry.entity_id.id == entity_id.id) {
                    // Convert our commands to flight controller setpoints
                    const setpoints = FlightController.ControlSetpoints{
                        .desired_rates = commands.desired_rates,
                        .desired_thrust = commands.desired_thrust,
                    };

                    entry.component.setControlSetpoints(setpoints);
                    return;
                }
            }
        }
    }
};
