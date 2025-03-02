//MotorController.zig
const std = @import("std");
const Math = @import("core/Math.zig");
const Drone = @import("core/Drone.zig");

const time = std.time;
const fs = std.fs;
const net = std.net;
const json = std.json;
const heap = std.heap;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const Atomic = std.atomic;

const Protocol = Drone.Protocol;
const Battery = Drone.Battery;
const DSHOT = Drone.DSHOT;
const Timing = Drone.TimingUtils;

// DShot Constants
const GPIO_BASE: usize = 0x3F200000; // Raspberry Pi zero 2W GPIO base address

// GPIO Memory mapping structure
const GpioRegs = struct {
    gpfsel: [6]u32,
    reserved1: u32,
    gpset: [2]u32,
    reserved2: u32,
    gpclr: [2]u32,
};

pub const PidController = struct {
    const Self = @This();

    // PID constants
    kp: f32, // Proportional gain
    ki: f32, // Integral gain
    kd: f32, // Derivative gain

    // PID state
    setpoint: f32 = 0.0,
    last_error: f32 = 0.0,
    integral: f32 = 0.0,
    output: f32 = 0.0,

    // Limit values
    max_integral: f32,
    max_output: f32,

    // Time tracking
    last_time: i128 = 0,

    pub fn init(kp: f32, ki: f32, kd: f32, max_integral: f32, max_output: f32) Self {
        return Self{
            .kp = kp,
            .ki = ki,
            .kd = kd,
            .max_integral = max_integral,
            .max_output = max_output,
        };
    }

    pub fn reset(self: *Self) void {
        self.integral = 0.0;
        self.last_error = 0.0;
        self.output = 0.0;
        self.last_time = 0;
    }

    pub fn setSetpoint(self: *Self, setpoint: f32) void {
        self.setpoint = setpoint;
    }

    pub fn update(self: *Self, current_value: f32, current_time: i128) f32 {
        const err = self.setpoint - current_value;

        // Calculate dt in seconds (convert from ns to s)
        var dt: f32 = 0.01; // Default to 10ms if first run
        if (self.last_time > 0) {
            dt = @floatFromInt(current_time - self.last_time);
            dt /= @as(f32, 1_000_000_000.0); // Convert ns to seconds
        }
        self.last_time = current_time;

        // Guard against very small or negative dt values
        if (dt < 0.001) dt = 0.001;

        // Calculate integral term with anti-windup
        self.integral += err * dt;
        self.integral = @min(self.max_integral, @max(-self.max_integral, self.integral));

        // Calculate derivative term
        const derivative = if (self.last_time > 0) (err - self.last_error) / dt else 0.0;
        self.last_error = err;

        // Calculate PID output
        const output = self.kp * err + self.ki * self.integral + self.kd * derivative;

        // Clamp output to limits
        self.output = @min(self.max_output, @max(-self.max_output, output));

        return self.output;
    }
};

pub const QuadcopterController = struct {
    const Self = @This();

    // Physical configuration
    const MotorConfiguration = enum {
        X_Configuration, // Motors in X pattern (most common)
        Plus_Configuration, // Motors in + pattern
    };

    pub const Axis = enum {
        Roll,
        Pitch,
        Yaw,
        Any,
    };

    roll_pid: PidController,
    pitch_pid: PidController,
    yaw_pid: PidController,

    // Motor mixing values
    motor_outputs: [4]f32,

    motor_config: MotorConfiguration,
    base_throttle: f32,

    current_orientation: Math.Quaternion,
    target_orientation: Math.Quaternion,

    running: std.atomic.Value(bool),
    mutex: std.Thread.Mutex,
    pid_thread: ?std.Thread,

    pub fn init(roll_kp: f32, roll_ki: f32, roll_kd: f32, pitch_kp: f32, pitch_ki: f32, pitch_kd: f32, yaw_kp: f32, yaw_ki: f32, yaw_kd: f32, max_integral: f32, max_output: f32, config: MotorConfiguration, base_throttle: f32) Self {
        return Self{
            .roll_pid = PidController.init(roll_kp, roll_ki, roll_kd, max_integral, max_output),
            .pitch_pid = PidController.init(pitch_kp, pitch_ki, pitch_kd, max_integral, max_output),
            .yaw_pid = PidController.init(yaw_kp, yaw_ki, yaw_kd, max_integral, max_output),
            .motor_outputs = [_]f32{0.0} ** 4,
            .motor_config = config,
            .base_throttle = base_throttle,
            .current_orientation = Math.Quaternion.identity(),
            .target_orientation = Math.Quaternion.identity(),
            .running = std.atomic.Value(bool).init(false),
            .mutex = std.Thread.Mutex{},
            .pid_thread = null,
        };
    }

    pub fn start(self: *Self) !void {
        if (self.running.load(.acquire)) return;

        self.running.store(true, .release);
        self.pid_thread = try std.Thread.spawn(.{}, pidControlLoop, .{self});
    }

    pub fn stop(self: *Self) void {
        if (!self.running.load(.acquire)) return;

        self.running.store(false, .release);
        if (self.pid_thread) |thread| {
            thread.join();
            self.pid_thread = null;
        }
    }

    pub fn setTargetOrientation(self: *Self, quaternion: Math.Quaternion) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.target_orientation = quaternion;
    }

    pub fn updateCurrentOrientation(self: *Self, quaternion: Math.Quaternion) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.current_orientation = quaternion;
    }

    pub fn getMotorOutputs(self: *Self) [4]f32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.motor_outputs;
    }

    fn pidControlLoop(self: *Self) void {
        Timing.setRealtimePriority() catch |err| {
            std.debug.print("Failed to set realtime priority for PID thread: {any}\n", .{err});
        };
        Timing.pinToCore(1) catch |err| {
            std.debug.print("Failed to pin PID thread to core: {any}\n", .{err});
        };

        const update_frequency_hz: f32 = 1000.0; // 1kHz update rate
        const update_period_ns: u64 = @intFromFloat(1_000_000_000.0 / update_frequency_hz);

        var current_euler_cache: [3]f32 = undefined;
        var target_euler_cache: [3]f32 = undefined;

        var frame_count: u64 = 0;
        var overrun_count: u64 = 0;

        while (self.running.load(.acquire)) {
            const start_time = std.time.nanoTimestamp();
            frame_count += 1;

            // Get quaternions with minimal lock time
            var current_quat: Math.Quaternion = undefined;
            var target_quat: Math.Quaternion = undefined;

            {
                self.mutex.lock();
                current_quat = self.current_orientation;
                target_quat = self.target_orientation;
                self.mutex.unlock();
            }

            // Convert quaternions to Euler angles outside of lock
            // This is CPU intensive but now outside critical section
            current_euler_cache = current_quat.toEuler();
            target_euler_cache = target_quat.toEuler();

            // Update PID controllers (with lock)
            self.mutex.lock();

            // Set PID setpoints
            self.roll_pid.setSetpoint(target_euler_cache[0]);
            self.pitch_pid.setSetpoint(target_euler_cache[1]);
            self.yaw_pid.setSetpoint(target_euler_cache[2]);

            // Update PID controllers
            const current_time = std.time.nanoTimestamp();
            const roll_output = self.roll_pid.update(current_euler_cache[0], current_time);
            const pitch_output = self.pitch_pid.update(current_euler_cache[1], current_time);
            const yaw_output = self.yaw_pid.update(current_euler_cache[2], current_time);

            // Apply motor mixing algorithm
            self.applyMotorMixing(roll_output, pitch_output, yaw_output);

            self.mutex.unlock();

            // Sleep until next update period with overrun detection
            const elapsed_ns = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
            if (elapsed_ns < update_period_ns) {
                std.time.sleep(update_period_ns - elapsed_ns);
            } else {
                // We're overrunning our frame time budget
                overrun_count += 1;
                if (frame_count % 1000 == 0) {
                    std.debug.print("PID control overrun: {d}/{d} frames, avg time: {d}ns\n", .{ overrun_count, frame_count, elapsed_ns });
                }
            }
        }
    }

    fn applyMotorMixing(self: *Self, roll: f32, pitch: f32, yaw: f32) void {
        // Calculate motor outputs based on PID controller outputs
        // Motors are numbered:
        //   X Configuration:     Plus Configuration:
        //      0    1                    0
        //       \  /                     |
        //        \/                   3--+--1
        //        /\                      |
        //       /  \                     2
        //      2    3

        switch (self.motor_config) {
            .X_Configuration => {
                // Front-left, front-right, back-right, back-left
                self.motor_outputs[0] = self.base_throttle - roll - pitch - yaw;
                self.motor_outputs[1] = self.base_throttle + roll - pitch + yaw;
                self.motor_outputs[2] = self.base_throttle - roll + pitch + yaw;
                self.motor_outputs[3] = self.base_throttle + roll + pitch - yaw;
            },
            .Plus_Configuration => {
                // Front, right, back, left
                self.motor_outputs[0] = self.base_throttle - pitch - yaw;
                self.motor_outputs[1] = self.base_throttle + roll + yaw;
                self.motor_outputs[2] = self.base_throttle + pitch - yaw;
                self.motor_outputs[3] = self.base_throttle - roll + yaw;
            },
        }

        // Ensure all motor outputs are within valid range (0-100%)
        for (&self.motor_outputs) |*output| {
            output.* = @min(100.0, @max(0.0, output.*));
        }
    }
};

const PinRegisterCache = struct {
    reg_idx: usize,
    bit_mask: u32,
    pin: u8,
};

pub const Controller = struct {
    const Self = @This();

    motors: []Motor,
    pin_cache: ?[]PinRegisterCache = null,
    gpio_mem: *volatile GpioRegs,
    running: Atomic.Value(bool),
    thread: ?std.Thread = null,
    allocator: std.mem.Allocator,

    protocol: DSHOT,
    battery_type: Battery,
    battery_voltage: Atomic.Value(f32),
    battery_percentage: Atomic.Value(f32),
    low_battery_failsafe: Atomic.Value(bool),
    battery_mutex: Mutex,

    orientation_control: ?QuadcopterController = null,
    current_orientation: Math.Quaternion = Math.Quaternion.identity(),
    target_orientation: Math.Quaternion = Math.Quaternion.identity(),
    orientation_control_active: std.atomic.Value(bool),
    orientation_mutex: Mutex,

    // Motor Configuration
    const Motor = struct {
        pin: u8,
        direction: Atomic.Value(Drone.RotationDirection),
        throttle: Atomic.Value(u16),
        armed: Atomic.Value(bool),
        mutex: Mutex,

        pub fn init(pin: u8, direction: Drone.RotationDirection) Motor {
            return Motor{
                .pin = pin,
                .direction = Atomic.Value(Drone.RotationDirection).init(direction),
                .throttle = Atomic.Value(u16).init(DSHOT.MIN_THROTTLE),
                .armed = Atomic.Value(bool).init(false),
                .mutex = Mutex{},
            };
        }
    };

    pub const Config = struct {
        pin: u8,
        direction: Drone.RotationDirection,
    };

    pub fn init(allocator: std.mem.Allocator, config: []Config, battery_type: Battery, protocol: DSHOT) !*Self {
        // try setRealtimePriority();
        // try pinToCore(3);

        // Open /dev/mem to map GPIO registers
        const mem_file = try fs.openFileAbsolute("/dev/mem", .{ .mode = .read_write });
        defer mem_file.close();

        // Map GPIO memory region
        const gpio_mem = try std.posix.mmap(
            null,
            @sizeOf(GpioRegs),
            std.os.linux.PROT.READ | std.os.linux.PROT.WRITE,
            std.os.linux.MAP{ .TYPE = .SHARED },
            mem_file.handle,
            GPIO_BASE,
        );

        // Initialize motors
        var motors = try allocator.alloc(Motor, config.len);
        for (config, 0..) |motor_config, i| {
            motors[i] = Motor.init(motor_config.pin, motor_config.direction);
            // Configure GPIO as output
            const reg_idx = @divFloor(motor_config.pin, 10);
            const bit_idx = @mod(motor_config.pin, 10) * 3;
            const gpio_mem_ptr: *volatile GpioRegs = @ptrCast(@alignCast(gpio_mem));
            gpio_mem_ptr.gpfsel[reg_idx] &= ~(@as(u32, 0b111) << @as(u5, @intCast(bit_idx)));
            gpio_mem_ptr.gpfsel[reg_idx] |= @as(u32, 0b001) << @as(u5, @intCast(bit_idx));
        }

        const controller = try allocator.create(Self);
        controller.* = Self{
            .motors = motors,
            .gpio_mem = @ptrCast(@alignCast(gpio_mem)),
            .running = Atomic.Value(bool).init(true),
            .allocator = allocator,
            .protocol = protocol,
            .battery_type = battery_type,
            .battery_voltage = Atomic.Value(f32).init(0.0),
            .battery_percentage = Atomic.Value(f32).init(0.0),
            .low_battery_failsafe = Atomic.Value(bool).init(true),
            .battery_mutex = Mutex{},
            .orientation_control_active = std.atomic.Value(bool).init(false),
            .orientation_mutex = Mutex{},
        };
        try controller.initPinCache();

        std.debug.print(
            "Motor controller initialized with {d} Motors and battery type: {s} ({d} cells)\n",
            .{ config.len, @tagName(battery_type), battery_type.cellCount() },
        );

        controller.orientation_control = QuadcopterController.init(
            0.5,
            0.01,
            0.1,
            0.5,
            0.01,
            0.1,
            0.2,
            0.0,
            0.05,
            20.0,
            30.0,
            .X_Configuration,
            20.0,
        );

        controller.thread = try Thread.spawn(.{}, motorControlThread, .{controller});
        return controller;
    }

    fn initPinCache(self: *Self) !void {
        self.pin_cache = try self.allocator.alloc(PinRegisterCache, self.motors.len);
        var pin_cache = self.pin_cache.?;
        for (self.motors, 0..) |motor, i| {
            const pin = motor.pin;
            pin_cache[i] = PinRegisterCache{
                .reg_idx = @divFloor(pin, 32),
                .bit_mask = @as(u32, 1) << @as(u5, @intCast(@mod(pin, 32))),
                .pin = pin, // Store original pin
            };
        }
    }

    pub fn updateBatteryVoltage(self: *Self, voltage: f32) void {
        self.battery_mutex.lock();
        defer self.battery_mutex.unlock();

        self.battery_voltage.store(voltage, .release);

        // Calculate battery percentage
        const percentage = self.battery_type.calculatePercentage(voltage);
        self.battery_percentage.store(percentage, .release);

        // Check for low battery condition
        if (voltage <= self.battery_type.minVoltage() and !self.low_battery_failsafe.load(.acquire)) {
            self.low_battery_failsafe.store(true, .release);

            std.debug.print("LOW BATTERY ALERT! {d:.2}V / {d:.1}% - Triggering Failsafe\n", .{ voltage, percentage });

            // Disarm all motors as failsafe
            for (0..self.motors.len) |i| {
                self.disarmMotor(i) catch |err| {
                    std.debug.print("Failed to disarm motor {d} during low battery failsafe: {any}\n", .{ i, err });
                };
            }
        }

        if (voltage > self.battery_type.minVoltage() and self.low_battery_failsafe.load(.acquire)) {
            std.debug.print("Turning off Failsafe - Battery raised above threshold! {d}V\n", .{voltage});
            self.low_battery_failsafe.store(false, .release);
        }

        // Log battery status periodically
        if (@mod(@as(u64, @intFromFloat(voltage * 100)), 25) < 2) { // Log roughly every 0.5V change
            std.debug.print("Battery: {d:.2}V / {d:.1}%\n", .{ voltage, percentage });
        }
    }

    pub fn deinit(self: *Self) void {
        self.running.store(false, .release);
        if (self.thread) |thread| {
            thread.join();
        }

        self.allocator.free(self.motors);
        // Unmap GPIO memory
        _ = std.os.linux.munmap(@as([*]u8, @ptrCast(@volatileCast(self.gpio_mem))), @sizeOf(GpioRegs));
    }

    fn sendDshotPacket(self: *Self, pin: u8, packet: u16) void {
        var reg_idx: usize = undefined;
        var bit_mask: u32 = undefined;

        // Quick lookup from cache if possible
        var found = false;
        if (self.pin_cache) |pin_cache| {
            for (pin_cache) |cache| {
                if (cache.pin == pin) {
                    reg_idx = cache.reg_idx;
                    bit_mask = cache.bit_mask;
                    found = true;
                    break;
                }
            }
        }

        // // Fall back to computation if not found
        if (!found) {
            reg_idx = @divFloor(pin, 32);
            bit_mask = @as(u32, 1) << @as(u5, @intCast(@mod(pin, 32)));
        }
        // Disable interrupts for precise timing
        var old_mask: std.os.linux.sigset_t = undefined;
        _ = std.os.linux.sigprocmask(std.os.linux.SIG.BLOCK, null, &old_mask);
        defer _ = std.os.linux.sigprocmask(std.os.linux.SIG.SETMASK, &old_mask, null);

        const protocol = self.protocol;
        const t0h = protocol.t0h_time();
        const t1h = protocol.t1h_time();
        const bit_time = protocol.bit_time();

        var start_time = Timing.getNanoTime();
        var bit_time_ns: u64 = undefined;

        // Send 16 bits
        var i: u8 = 0;
        while (i < 16) : (i += 1) {
            const bit = (packet >> (15 - @as(u4, @intCast(i)))) & 1;
            bit_time_ns = start_time;
            self.gpio_mem.gpset[reg_idx] = bit_mask;

            if (bit == 1) {
                Timing.busyWaitUntil(bit_time_ns + t1h);
            } else {
                Timing.busyWaitUntil(bit_time_ns + t0h);
            }

            self.gpio_mem.gpclr[reg_idx] = bit_mask;
            start_time += bit_time;
            Timing.busyWaitUntil(start_time);
        }

        Timing.busyWaitUntil(start_time + protocol.frame_reset_time());
    }

    pub fn reverseMotorDirection(self: *Self, motor_idx: usize) !void {
        if (motor_idx >= self.motors.len) return error.InvalidMotorIndex;

        var motor = &self.motors[motor_idx];
        motor.mutex.lock();
        defer motor.mutex.unlock();

        if (motor.armed.load(.acquire)) {
            std.debug.print("Cannot reverse direction while motor {d} is armed\n", .{motor_idx});
            return error.MotorArmed;
        }

        var start_time = Timing.getNanoTime();

        // Send a solid burst of zero throttle commands first
        var i: u32 = 0;
        while (i < 100) : (i += 1) {
            self.sendDshotPacket(motor.pin, DSHOT.create_packet(DSHOT.CMD.MOTOR_STOP, false));
            start_time += 100 * time.ns_per_us;
            Timing.busyWaitUntil(start_time);
        }

        // Long delay to ensure ESC is ready
        start_time += 10 * time.ns_per_ms;
        Timing.busyWaitUntil(start_time);

        const current_direction = motor.direction.load(.acquire);
        const direction_cmd = switch (current_direction) {
            .Clockwise => DSHOT.CMD.SPIN_DIRECTION_2,
            .Counterclockwise => DSHOT.CMD.SPIN_DIRECTION_1,
        };

        // Send multiple bursts of direction commands with delays between bursts
        var burst: u32 = 0;
        while (burst < 7) : (burst += 1) {
            // Send a burst of direction commands
            i = 0;
            while (i < 100) : (i += 1) {
                self.sendDshotPacket(motor.pin, DSHOT.create_packet(direction_cmd, true));
                start_time += 100 * time.ns_per_us;
                Timing.busyWaitUntil(start_time);
            }

            // Send save settings after each burst
            i = 0;
            while (i < 20) : (i += 1) {
                self.sendDshotPacket(motor.pin, DSHOT.create_packet(DSHOT.CMD.SAVE_SETTINGS, true));
                start_time += 100 * time.ns_per_us;
                Timing.busyWaitUntil(start_time);
            }

            // Delay between bursts
            start_time += 50 * time.ns_per_ms;
            Timing.busyWaitUntil(start_time);
        }

        // Final delay to ensure all commands are processed
        start_time += 100 * time.ns_per_ms;
        Timing.busyWaitUntil(start_time);

        const new_direction: Drone.RotationDirection = if (current_direction == .Clockwise)
            .Counterclockwise
        else
            .Clockwise;

        motor.direction.store(new_direction, .release);
        std.debug.print("Motor {d} direction reversed from {s} to {s}\n", .{ motor_idx, @tagName(current_direction), @tagName(new_direction) });
    }

    pub fn armMotor(self: *Self, motor_idx: usize) !void {
        if (motor_idx >= self.motors.len) return error.InvalidMotorIndex;

        var motor = &self.motors[motor_idx];
        motor.mutex.lock();
        defer motor.mutex.unlock();

        std.debug.print("Arming motor {d}...\n", .{motor_idx});

        // Send arming sequence
        var i: u32 = 0;
        while (i < 1000) : (i += 1) {
            const packet = DSHOT.create_packet(DSHOT.CMD.MOTOR_STOP, false);
            self.sendDshotPacket(motor.pin, packet);
            std.time.sleep(1 * time.ns_per_ms); // 1ms delay
        }
        std.time.sleep(150 * std.time.ns_per_ms);

        motor.armed.store(true, .release);
        std.debug.print("Motor {d} armed\n", .{motor_idx});
    }

    pub fn disarmMotor(self: *Self, motor_idx: usize) !void {
        if (motor_idx >= self.motors.len) return error.InvalidMotorIndex;

        var motor = &self.motors[motor_idx];
        motor.mutex.lock();
        defer motor.mutex.unlock();

        // Send zero throttle one last time
        const packet = DSHOT.create_packet(DSHOT.CMD.MOTOR_STOP, false);
        self.sendDshotPacket(motor.pin, packet);

        motor.armed.store(false, .release);
        motor.throttle.store(DSHOT.MIN_THROTTLE, .release);
        std.debug.print("Motor {d} disarmed\n", .{motor_idx});
    }

    pub fn setMotorSpeed(self: *Self, motor_idx: usize, speed_percent: f32) !void {
        if (motor_idx >= self.motors.len) return error.InvalidMotorIndex;

        var motor = &self.motors[motor_idx];
        if (!motor.armed.load(.acquire)) return error.MotorNotArmed;

        const throttle = DSHOT.percentage_to_throttle(speed_percent);
        motor.throttle.store(throttle, .release);

        std.debug.print("Motor {d} speed set to {d:.1}%\n", .{ motor_idx, speed_percent });
    }

    pub fn setTargetOrientation(self: *Self, quaternion: Math.Quaternion) void {
        self.orientation_mutex.lock();
        defer self.orientation_mutex.unlock();

        self.target_orientation = quaternion;

        if (self.orientation_control) |*control| {
            control.setTargetOrientation(quaternion);
        }

        std.debug.print("Target orientation set: W={d:.3} X={d:.3} Y={d:.3} Z={d:.3}\n", .{ quaternion.w, quaternion.x, quaternion.y, quaternion.z });
    }

    pub fn updateCurrentOrientation(self: *Self, quaternion: Math.Quaternion) void {
        self.orientation_mutex.lock();
        defer self.orientation_mutex.unlock();

        self.current_orientation = quaternion;

        if (self.orientation_control) |*control| {
            control.updateCurrentOrientation(quaternion);
        }
    }

    pub fn startOrientationControl(self: *Self) !void {
        if (self.orientation_control_active.load(.acquire)) return;

        if (self.orientation_control) |*control| {
            try control.start();
            self.orientation_control_active.store(true, .release);
            std.debug.print("Orientation control started\n", .{});
        } else {
            return error.OrientationControllerNotInitialized;
        }
    }

    pub fn stopOrientationControl(self: *Self) void {
        if (!self.orientation_control_active.load(.acquire)) return;

        if (self.orientation_control) |*control| {
            control.stop();
            self.orientation_control_active.store(false, .release);
            std.debug.print("Orientation control stopped\n", .{});
        }
    }

    fn motorControlThread(self: *Self) void {
        Timing.setRealtimePriority() catch |err| {
            std.debug.print("Failed to set realtime priority for motor thread: {any}\n", .{err});
        };
        Timing.pinToCore(0) catch |err| {
            std.debug.print("Failed to pin motor thread to core: {any}\n", .{err});
        };

        var motor_outputs_cache: [4]f32 = undefined;
        var last_update_time = std.time.nanoTimestamp();
        const update_interval_ns: i128 = 2000 * std.time.ns_per_us; // 1ms (1kHz update rate)

        while (self.running.load(.acquire)) {
            const current_time = std.time.nanoTimestamp();

            // Use precise timing with a minimum interval to prevent CPU hogging
            if (current_time - last_update_time < update_interval_ns) {
                std.time.sleep(10 * time.ns_per_us); //
                continue;
            }

            last_update_time = current_time;

            if (self.low_battery_failsafe.load(.acquire)) {
                for (self.motors) |*motor| {
                    if (motor.armed.load(.acquire)) {
                        motor.armed.store(false, .release);
                    }
                }
                continue;
            }

            if (self.orientation_control_active.load(.acquire) and self.orientation_control != null) {
                motor_outputs_cache = self.orientation_control.?.getMotorOutputs();

                for (0..@min(self.motors.len, motor_outputs_cache.len)) |i| {
                    if (self.motors[i].armed.load(.acquire)) {
                        const throttle = DSHOT.percentage_to_throttle(motor_outputs_cache[i]);
                        self.motors[i].throttle.store(throttle, .release);

                        const packet = DSHOT.create_packet(throttle, false);
                        self.sendDshotPacket(self.motors[i].pin, packet);
                    }
                }
            } else {
                // Regular direct motor control when orientation control is not active
                for (self.motors) |*motor| {
                    if (motor.armed.load(.acquire)) {
                        const throttle = motor.throttle.load(.acquire);
                        const packet = DSHOT.create_packet(throttle, false);
                        self.sendDshotPacket(motor.pin, packet);
                    }
                }
            }
        }
    }
};

const Server = struct {
    const Self = @This();
    const HEARTBEAT_TIMEOUT_SEC = 2;

    state: Atomic.Value(Protocol.ServerState),
    socket: ?std.posix.socket_t,
    address: std.net.Address,
    client_addr: ?std.net.Address,
    controller: ?*Controller,
    last_heartbeat: Atomic.Value(i64),
    allocator: std.mem.Allocator,
    running: bool,
    mutex: Mutex,

    pub fn init(allocator: std.mem.Allocator, port: u16) !*Self {
        const self = try allocator.create(Self);

        self.* = .{
            .state = Atomic.Value(Protocol.ServerState).init(.WaitingForClient),
            .socket = null,
            .address = try std.net.Address.parseIp("0.0.0.0", port),
            .client_addr = null,
            .controller = null,
            .last_heartbeat = Atomic.Value(i64).init(0),
            .allocator = allocator,
            .running = false,
            .mutex = .{},
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.controller) |controller| {
            controller.deinit();
        }
        if (self.socket) |sock| {
            std.posix.close(sock);
        }
        self.allocator.destroy(self);
    }

    pub fn start(self: *Self) !void {
        if (self.running) return;

        try self.initializeSocket();
        self.running = true;

        // Start heartbeat monitoring thread
        _ = try Thread.spawn(.{}, heartbeatMonitor, .{self});

        try Timing.pinToCore(3);

        // Start main server loop
        try self.serverLoop();
    }

    fn initializeSocket(self: *Self) !void {
        const sock = try std.posix.socket(
            std.posix.AF.INET,
            std.posix.SOCK.DGRAM,
            std.posix.IPPROTO.UDP,
        );
        errdefer std.posix.close(sock);

        // Set socket options
        var reuse: c_int = 1;
        try std.posix.setsockopt(
            sock,
            std.posix.SOL.SOCKET,
            std.posix.SO.REUSEADDR,
            std.mem.asBytes(&reuse),
        );

        const timeout = std.posix.timeval{
            .sec = 1, // 1 second timeout
            .usec = 0,
        };
        try std.posix.setsockopt(
            sock,
            std.posix.SOL.SOCKET,
            std.posix.SO.RCVTIMEO,
            std.mem.asBytes(&timeout),
        );

        try std.posix.bind(
            sock,
            &self.address.any,
            self.address.getOsSockLen(),
        );

        self.socket = sock;
    }

    fn serverLoop(self: *Self) !void {
        var buf: [4096]u8 = undefined;
        var client_addr: std.posix.sockaddr = undefined;
        var client_addr_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);

        while (self.running) {
            const len = std.posix.recvfrom(
                self.socket.?,
                &buf,
                0,
                &client_addr,
                &client_addr_len,
            ) catch |err| {
                if (err == error.WouldBlock or err == error.TimedOut) {
                    continue;
                }

                std.debug.print("Receive error: {any}\n", .{err});
                continue;
            };

            if (len == 0) continue;

            const msg = buf[0..len];
            self.handleMessage(msg, client_addr, client_addr_len) catch |err| {
                std.debug.print("Failed to handle message! Err => {any}\n", .{err});
            };
        }
    }

    fn createStatusResponse(self: *Self) ![]u8 {
        if (self.controller) |controller| {
            var status_buffer: [8192]u8 = undefined;
            var fbs = std.io.fixedBufferStream(&status_buffer);
            var writer = fbs.writer();

            try writer.writeAll("STATUS ");

            for (controller.motors, 0..) |motor, i| {
                const armed = motor.armed.load(.acquire);
                const throttle = motor.throttle.load(.acquire);

                try std.fmt.format(writer, "{d} {d} {d} ", .{
                    i,
                    @intFromBool(armed),
                    throttle,
                });
            }

            const voltage = controller.battery_voltage.load(.acquire);
            const percentage = controller.battery_percentage.load(.acquire);
            const failsafe = controller.low_battery_failsafe.load(.acquire);

            try std.fmt.format(writer, "BATTERY {d:.2} {d:.1} {d} ", .{
                voltage,
                percentage,
                @intFromBool(failsafe),
            });

            // Add PID controller state information
            const orientation_active = controller.orientation_control_active.load(.acquire);
            try std.fmt.format(writer, "PID_ACTIVE {d} ", .{@intFromBool(orientation_active)});

            if (orientation_active and controller.orientation_control != null) {
                var pid_control = controller.orientation_control.?;

                try std.fmt.format(writer, "PID_PARAMS " ++
                    "R_KP {d:.3} R_KI {d:.3} R_KD {d:.3} " ++
                    "P_KP {d:.3} P_KI {d:.3} P_KD {d:.3} " ++
                    "Y_KP {d:.3} Y_KI {d:.3} Y_KD {d:.3} ", .{
                    pid_control.roll_pid.kp,  pid_control.roll_pid.ki,  pid_control.roll_pid.kd,
                    pid_control.pitch_pid.kp, pid_control.pitch_pid.ki, pid_control.pitch_pid.kd,
                    pid_control.yaw_pid.kp,   pid_control.yaw_pid.ki,   pid_control.yaw_pid.kd,
                });

                const current_quat = controller.current_orientation;
                try std.fmt.format(writer, "CURR_QUAT {d:.4} {d:.4} {d:.4} {d:.4} ", .{
                    current_quat.w, current_quat.x, current_quat.y, current_quat.z,
                });

                const target_quat = controller.target_orientation;
                try std.fmt.format(writer, "TARGET_QUAT {d:.4} {d:.4} {d:.4} {d:.4} ", .{
                    target_quat.w, target_quat.x, target_quat.y, target_quat.z,
                });

                const motor_outputs = pid_control.getMotorOutputs();
                try writer.writeAll("PID_OUTPUTS ");
                for (motor_outputs, 0..) |output, i| {
                    try std.fmt.format(writer, "{d}:{d:.1} ", .{ i, output });
                }

                try std.fmt.format(writer, "PID_ERRORS {d:.2} {d:.2} {d:.2} ", .{
                    pid_control.roll_pid.last_error,
                    pid_control.pitch_pid.last_error,
                    pid_control.yaw_pid.last_error,
                });

                const current_euler = current_quat.toEuler();
                const target_euler = target_quat.toEuler();

                const rad_to_deg = 180.0 / std.math.pi;
                try std.fmt.format(writer, "CURR_EULER {d:.1} {d:.1} {d:.1} TARGET_EULER {d:.1} {d:.1} {d:.1} ", .{
                    current_euler[0] * rad_to_deg, // Roll in degrees
                    current_euler[1] * rad_to_deg, // Pitch in degrees
                    current_euler[2] * rad_to_deg, // Yaw in degrees
                    target_euler[0] * rad_to_deg, // Target roll in degrees
                    target_euler[1] * rad_to_deg, // Target pitch in degrees
                    target_euler[2] * rad_to_deg, // Target yaw in degrees
                });

                // PID integrator values (useful for debugging wind-up issues)
                try std.fmt.format(writer, "PID_INTEGRAL {d:.2} {d:.2} {d:.2}", .{
                    pid_control.roll_pid.integral,
                    pid_control.pitch_pid.integral,
                    pid_control.yaw_pid.integral,
                });
            }

            try writer.writeAll("\n");
            const written_size = fbs.getPos() catch 0;
            return self.allocator.dupe(u8, status_buffer[0..written_size]);
        } else {
            return try self.allocator.dupe(u8, "ACK\n");
        }
    }

    fn handleMessage(self: *Self, msg: []const u8, client_addr: std.posix.sockaddr, addr_len: std.posix.socklen_t) !void {
        const trimmed_msg = std.mem.trim(u8, msg, &std.ascii.whitespace);

        switch (self.state.load(.acquire)) {
            .WaitingForClient => {
                if (std.mem.eql(u8, trimmed_msg, "CONNECT")) {
                    std.debug.print("Connecting...\n", .{});
                    self.client_addr = std.net.Address.initPosix(@alignCast(&client_addr));
                    try self.sendResponse("ACK\n", client_addr, addr_len);
                    self.state.store(.Connected, .release);
                }
            },
            .Connected, .ConfigSync => {
                if (std.mem.startsWith(u8, trimmed_msg, "{")) {
                    // Attempt to parse config JSON
                    std.debug.print("Configuring...\n", .{});
                    try self.handleConfigMessage(trimmed_msg);
                    try self.sendResponse("CONFIG_ACK\n", client_addr, addr_len);
                    self.state.store(.Ready, .release);
                }
            },
            .Ready, .Running => {
                if (std.mem.eql(u8, trimmed_msg, "HEARTBEAT")) {
                    self.last_heartbeat.store(std.time.timestamp(), .release);

                    // Create status response
                    const status_msg = try self.createStatusResponse();
                    defer self.allocator.free(status_msg);

                    try self.sendResponse(status_msg, client_addr, addr_len);
                } else {
                    try self.handleCommand(trimmed_msg);
                }
            },
            .Failed => {
                std.debug.print("Connection Failed. Waiting for Client to Reconnect...\n", .{});

                if (self.controller) |controller| {
                    controller.deinit();
                    self.controller = null;
                }

                if (self.socket) |socket| {
                    std.posix.close(socket);
                    self.socket = null;
                }

                try self.initializeSocket();
                self.state.store(.WaitingForClient, .release);
                std.debug.print("Socket recreated, waiting for client...\n", .{});
            },
        }
    }

    fn handleConfigMessage(self: *Self, config_json: []const u8) !void {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            config_json,
            .{},
        );
        defer parsed.deinit();

        const protocol: DSHOT = @enumFromInt(parsed.value.object.get("dshot_protocol").?.integer);

        const motors_array = parsed.value.object.get("motors").?.array;
        var config = try self.allocator.alloc(Controller.Config, motors_array.items.len);
        defer self.allocator.free(config);

        for (motors_array.items, 0..) |motor, i| {
            config[i] = .{
                .pin = @intCast(motor.object.get("pin").?.integer),
                .direction = @enumFromInt(motor.object.get("direction").?.integer),
            };
        }

        var battery_type = Battery.Lipo_4S; // Default to 4S if not specified
        if (parsed.value.object.get("battery")) |battery_config| {
            const cells = @as(u8, @intCast(battery_config.object.get("cells").?.integer));
            battery_type = @as(Battery, @enumFromInt(cells));
        }

        if (self.controller) |controller| {
            controller.deinit();
            self.controller = null;
        }

        self.controller = try Controller.init(self.allocator, config, battery_type, protocol);
    }

    fn handleCommand(self: *Self, cmd_str: []const u8) !void {
        var iterator = std.mem.splitSequence(u8, cmd_str, " ");
        const parsed_cmd = iterator.next() orelse return error.NoCommandReceived;

        const cmd_opt: ?Protocol.CommandType = std.meta.stringToEnum(Protocol.CommandType, parsed_cmd);
        if (cmd_opt == null) {
            std.debug.print("{s} is not a valid command!\n", .{parsed_cmd});
            return error.InvalidCommand;
        }

        const cmd = cmd_opt.?;
        switch (cmd) {
            .Arm => {
                const motor_idx = try std.fmt.parseInt(
                    usize,
                    iterator.next() orelse return error.InvalidArmMotor,
                    10,
                );

                try self.controller.?.armMotor(motor_idx);
            },
            .Disarm => {
                const motor_idx = try std.fmt.parseInt(
                    usize,
                    iterator.next() orelse return error.InvalidDisarmMotor,
                    10,
                );

                try self.controller.?.disarmMotor(motor_idx);
            },
            .SetSpeed => {
                const motor_idx = try std.fmt.parseInt(
                    usize,
                    iterator.next() orelse return error.InvalidSpeedMotor,
                    10,
                );

                const speed = try std.fmt.parseFloat(
                    f32,
                    iterator.next() orelse return error.InvalidSpeedCommand,
                );
                try self.controller.?.setMotorSpeed(motor_idx, speed);
            },
            .ReverseDirection => {
                const motor_idx = try std.fmt.parseInt(
                    usize,
                    iterator.next() orelse return error.InvalidReverseMotor,
                    10,
                );

                try self.controller.?.reverseMotorDirection(motor_idx);
            },
            .Battery => {
                const voltage_str = iterator.next() orelse return error.InvalidVoltageStr;
                const voltage = std.fmt.parseFloat(f32, voltage_str) catch |err| {
                    std.debug.print("Failed to parse voltage str: {s}. Err => {any}\n", .{ voltage_str, err });
                    return;
                };

                if (self.controller) |controller| {
                    controller.updateBatteryVoltage(voltage);
                }
                return;
            },
            .UpdateOrientation => {
                const w = try std.fmt.parseFloat(f32, iterator.next() orelse return error.InvalidQuaternion);
                const x = try std.fmt.parseFloat(f32, iterator.next() orelse return error.InvalidQuaternion);
                const y = try std.fmt.parseFloat(f32, iterator.next() orelse return error.InvalidQuaternion);
                const z = try std.fmt.parseFloat(f32, iterator.next() orelse return error.InvalidQuaternion);

                const quaternion = Math.Quaternion{
                    .w = w,
                    .x = x,
                    .y = y,
                    .z = z,
                };

                if (self.controller) |controller| {
                    controller.updateCurrentOrientation(quaternion);
                }
            },
            .SetOrientation => {
                const w = try std.fmt.parseFloat(f32, iterator.next() orelse return error.InvalidQuaternion);
                const x = try std.fmt.parseFloat(f32, iterator.next() orelse return error.InvalidQuaternion);
                const y = try std.fmt.parseFloat(f32, iterator.next() orelse return error.InvalidQuaternion);
                const z = try std.fmt.parseFloat(f32, iterator.next() orelse return error.InvalidQuaternion);

                const quaternion = Math.Quaternion{
                    .w = w,
                    .x = x,
                    .y = y,
                    .z = z,
                };

                if (self.controller) |controller| {
                    controller.setTargetOrientation(quaternion);

                    // Start orientation control if not already running
                    if (!controller.orientation_control_active.load(.acquire)) {
                        try controller.startOrientationControl();
                    }
                }
            },
            .StopOrientation => {
                if (self.controller) |controller| {
                    controller.stopOrientationControl();
                }
            },
            .UpdatePidParams => {
                if (self.controller == null or self.controller.?.orientation_control == null) {
                    return error.OrientationControllerNotInitialized;
                }

                const axis_str = iterator.next() orelse return error.InvalidPidParamCommand;

                const kp = try std.fmt.parseFloat(f32, iterator.next() orelse return error.InvalidPidParam);
                const ki = try std.fmt.parseFloat(f32, iterator.next() orelse return error.InvalidPidParam);
                const kd = try std.fmt.parseFloat(f32, iterator.next() orelse return error.InvalidPidParam);

                var control = &self.controller.?.orientation_control.?;

                const axis_opt = std.meta.stringToEnum(QuadcopterController.Axis, axis_str);
                if (axis_opt) |axis| {
                    switch (axis) {
                        .Roll => {
                            control.roll_pid.kp = kp;
                            control.roll_pid.ki = ki;
                            control.roll_pid.kd = kd;
                            control.roll_pid.reset();
                            std.debug.print("Updated Roll PID: kp={d:.3} ki={d:.3} kd={d:.3}\n", .{ kp, ki, kd });
                        },
                        .Pitch => {
                            control.pitch_pid.kp = kp;
                            control.pitch_pid.ki = ki;
                            control.pitch_pid.kd = kd;
                            control.pitch_pid.reset();
                            std.debug.print("Updated Pitch PID: kp={d:.3} ki={d:.3} kd={d:.3}\n", .{ kp, ki, kd });
                        },
                        .Yaw => {
                            control.yaw_pid.ki = ki;
                            control.yaw_pid.kd = kd;
                            control.yaw_pid.reset();
                            std.debug.print("Updated Yaw PID: kp={d:.3} ki={d:.3} kd={d:.3}\n", .{ kp, ki, kd });
                        },
                        .Any => {
                            // Update all PIDs with the same values
                            control.roll_pid.kp = kp;
                            control.roll_pid.ki = ki;
                            control.roll_pid.kd = kd;
                            control.roll_pid.reset();

                            control.pitch_pid.kp = kp;
                            control.pitch_pid.ki = ki;
                            control.pitch_pid.kd = kd;
                            control.pitch_pid.reset();

                            control.yaw_pid.kp = kp;
                            control.yaw_pid.ki = ki;
                            control.yaw_pid.kd = kd;
                            control.yaw_pid.reset();

                            std.debug.print("Updated All PIDs: kp={d:.3} ki={d:.3} kd={d:.3}\n", .{ kp, ki, kd });
                        },
                    }
                }
            },

            // Add a command to update base throttle
            .UpdateBaseThrottle => {
                if (self.controller == null or self.controller.?.orientation_control == null) {
                    return error.OrientationControllerNotInitialized;
                }

                const base_throttle = try std.fmt.parseFloat(
                    f32,
                    iterator.next() orelse return error.InvalidBaseThrottleCommand,
                );

                // Clamp to valid range
                const throttle = @min(90.0, @max(10.0, base_throttle));

                self.controller.?.orientation_control.?.base_throttle = throttle;
                std.debug.print("Updated base throttle to {d:.1}%\n", .{throttle});
            },
        }
    }

    fn sendResponse(self: *Self, response: []const u8, client_addr: std.posix.sockaddr, addr_len: std.posix.socklen_t) !void {
        _ = std.posix.sendto(
            self.socket.?,
            response,
            0,
            &client_addr,
            addr_len,
        ) catch |err| {
            std.debug.print("Error in sending response: {s} to client! Err => {any}\n", .{ response, err });
        };
    }

    fn heartbeatMonitor(self: *Self) void {
        Timing.pinToCore(3) catch |err| {
            std.debug.print("Failed to pin heartbeatmonitor to pin 3! Err => {any}\n", .{err});
        };

        while (self.running) {
            const state = self.state.load(.acquire);
            if (state == .Ready or state == .Running) {
                const last_heartbeat = self.last_heartbeat.load(.acquire);
                const current_time = std.time.timestamp();

                if (last_heartbeat != 0 and current_time - last_heartbeat > HEARTBEAT_TIMEOUT_SEC) {
                    std.debug.print("Client heartbeat timeout. Curr: {d} Last: {d}  => Diff: {d}\n", .{ current_time, last_heartbeat, current_time - last_heartbeat });
                    self.state.store(.Failed, .release);
                    if (self.controller) |controller| {
                        // Safely disarm all motors
                        for (0..controller.motors.len) |i| {
                            controller.disarmMotor(i) catch |err| {
                                std.debug.print("Error disarming motor: {d}. Err => {any}\n", .{ i, err });
                            };
                        }
                    }
                }
            }
            std.time.sleep(100 * std.time.ns_per_ms);
        }
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var server = try Server.init(allocator, 5000);
    defer server.deinit();

    try server.start();
}
