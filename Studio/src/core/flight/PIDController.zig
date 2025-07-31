const std = @import("std");

/// Single-axis PID controller with anti-windup
pub const PIDController = struct {
    const Self = @This();
    
    // PID gains
    kp: f32,
    ki: f32,
    kd: f32,
    
    // State variables
    integrator: f32 = 0.0,
    derivative: f32 = 0.0,
    last_error: f32 = 0.0,
    
    // Anti-windup limits
    integrator_min: f32,
    integrator_max: f32,
    output_min: f32,
    output_max: f32,
    
    pub fn init(kp: f32, ki: f32, kd: f32, integrator_limit: f32, output_limit: f32) Self {
        return Self{
            .kp = kp,
            .ki = ki,
            .kd = kd,
            .integrator_min = -integrator_limit,
            .integrator_max = integrator_limit,
            .output_min = -output_limit,
            .output_max = output_limit,
        };
    }
    
    /// Update the PID controller with new error and time step
    pub fn step(self: *Self, err: f32, dt: f32) f32 {
        // Proportional term
        const p_term = self.kp * err;
        
        // Integral term with anti-windup
        self.integrator += err * dt;
        self.integrator = std.math.clamp(self.integrator, self.integrator_min, self.integrator_max);
        const i_term = self.ki * self.integrator;
        
        // Derivative term (derivative on measurement to avoid derivative kick)
        self.derivative = (err - self.last_error) / dt;
        const d_term = self.kd * self.derivative;
        self.last_error = err;
        
        // Combine terms and apply output limits
        const output = p_term + i_term + d_term;
        return std.math.clamp(output, self.output_min, self.output_max);
    }
    
    /// Reset the controller state (useful for mode changes)
    pub fn reset(self: *Self) void {
        self.integrator = 0.0;
        self.derivative = 0.0;
        self.last_error = 0.0;
    }
    
    /// Update gains (for runtime tuning)
    pub fn setGains(self: *Self, kp: f32, ki: f32, kd: f32) void {
        self.kp = kp;
        self.ki = ki;
        self.kd = kd;
    }
};

/// Three-axis PID controller for pose control (attitude or rates)
pub const PosePIDController = struct {
    const Self = @This();
    
    roll: PIDController,
    pitch: PIDController,
    yaw: PIDController,
    
    pub fn init(
        roll_gains: [3]f32,   // [kp, ki, kd]
        pitch_gains: [3]f32,  // [kp, ki, kd]
        yaw_gains: [3]f32,    // [kp, ki, kd]
        integrator_limit: f32,
        output_limit: f32,
    ) Self {
        return Self{
            .roll = PIDController.init(roll_gains[0], roll_gains[1], roll_gains[2], integrator_limit, output_limit),
            .pitch = PIDController.init(pitch_gains[0], pitch_gains[1], pitch_gains[2], integrator_limit, output_limit),
            .yaw = PIDController.init(yaw_gains[0], yaw_gains[1], yaw_gains[2], integrator_limit, output_limit),
        };
    }
    
    /// Update all three axes and return torque vector
    pub fn step(self: *Self, err_rpy: [3]f32, dt: f32) [3]f32 {
        return [3]f32{
            self.roll.step(err_rpy[0], dt),
            self.pitch.step(err_rpy[1], dt),
            self.yaw.step(err_rpy[2], dt),
        };
    }
    
    /// Reset all controllers
    pub fn reset(self: *Self) void {
        self.roll.reset();
        self.pitch.reset();
        self.yaw.reset();
    }
};