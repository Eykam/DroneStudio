//MotorController.zig
const std = @import("std");
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

pub const MotorController = struct {
    const Self = @This();
    motors: []Motor,
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
        };

        std.debug.print(
            "Motor controller initialized with {d} Motors and battery type: {s} ({d} cells)\n",
            .{ config.len, @tagName(battery_type), battery_type.cellCount() },
        );

        controller.thread = try Thread.spawn(.{}, motorControlThread, .{controller});
        return controller;
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
        const reg_idx = @divFloor(pin, 32);
        const bit_mask = @as(u32, 1) << @as(u5, @intCast(@mod(pin, 32)));

        // Disable interrupts for precise timing
        var old_mask: std.os.linux.sigset_t = undefined;
        _ = std.os.linux.sigprocmask(std.os.linux.SIG.BLOCK, null, &old_mask);
        defer _ = std.os.linux.sigprocmask(std.os.linux.SIG.SETMASK, &old_mask, null);

        var start_time = Timing.getNanoTime();
        var bit_time: u64 = undefined;

        // Send 16 bits
        var i: u8 = 0;
        while (i < 16) : (i += 1) {
            const bit = (packet >> (15 - @as(u4, @intCast(i)))) & 1;
            bit_time = start_time;
            self.gpio_mem.gpset[reg_idx] = bit_mask;

            if (bit == 1) {
                Timing.busyWaitUntil(bit_time + self.protocol.t1h_time());
            } else {
                Timing.busyWaitUntil(bit_time + self.protocol.t0h_time());
            }

            self.gpio_mem.gpclr[reg_idx] = bit_mask;
            start_time += self.protocol.bit_time();
            Timing.busyWaitUntil(start_time);
        }

        Timing.busyWaitUntil(start_time + self.protocol.frame_reset_time());
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
            // For arming, send zero throttle with no telemetry bit
            const packet = DSHOT.create_packet(DSHOT.CMD.MOTOR_STOP, false);
            self.sendDshotPacket(motor.pin, packet);
            std.time.sleep(1 * time.ns_per_ms); // 1ms delay
        }

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

        // Clamp speed percentage between 0 and 100
        const clamped_speed = @min(100.0, @max(0.0, speed_percent));

        const throttle_range = DSHOT.MAX_THROTTLE - DSHOT.MIN_THROTTLE;
        const throttle_increment = @as(u16, @intFromFloat((clamped_speed / 100.0) * @as(f32, @floatFromInt(throttle_range))));
        const throttle = DSHOT.MIN_THROTTLE + throttle_increment;

        const final_throttle = @min(DSHOT.MAX_THROTTLE, @max(DSHOT.MIN_THROTTLE, throttle));

        motor.throttle.store(final_throttle, .release);
        std.debug.print("Motor {d} speed set to {d:.1}%\n", .{ motor_idx, speed_percent });
    }

    fn motorControlThread(self: *Self) void {
        while (self.running.load(.acquire)) {
            if (self.low_battery_failsafe.load(.acquire)) {
                // In failsafe mode, ensure all motors remain disarmed
                for (self.motors) |*motor| {
                    if (motor.armed.load(.acquire)) {
                        motor.armed.store(false, .release);
                    }
                }
                // Still sleep to prevent CPU hogging
                std.time.sleep(10 * time.ns_per_ms);
                continue;
            }

            for (self.motors) |*motor| {
                if (motor.armed.load(.acquire)) {
                    const throttle = motor.throttle.load(.acquire);
                    const packet = DSHOT.create_packet(throttle, false);
                    self.sendDshotPacket(motor.pin, packet);
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
    motor_controller: ?*MotorController,
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
            .motor_controller = null,
            .last_heartbeat = Atomic.Value(i64).init(0),
            .allocator = allocator,
            .running = false,
            .mutex = .{},
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.motor_controller) |controller| {
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
        if (self.motor_controller) |controller| {
            var status = std.ArrayList(u8).init(self.allocator);
            defer status.deinit();

            try status.writer().writeAll("STATUS ");

            // Add status for each motor
            for (controller.motors, 0..) |motor, i| {
                const armed = motor.armed.load(.acquire);
                const throttle = motor.throttle.load(.acquire);

                // Format: "STATUS motor_idx armed throttle"
                try std.fmt.format(status.writer(), "{d} {d} {d} ", .{
                    i,
                    @intFromBool(armed),
                    throttle,
                });
            }

            const voltage = controller.battery_voltage.load(.acquire);
            const percentage = controller.battery_percentage.load(.acquire);
            const failsafe = controller.low_battery_failsafe.load(.acquire);

            try std.fmt.format(status.writer(), "BATTERY {d:.2} {d:.1} {d}", .{
                voltage,
                percentage,
                @intFromBool(failsafe),
            });

            try status.appendSlice("\n");
            return try status.toOwnedSlice();
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

                if (self.motor_controller) |motor_controller| {
                    motor_controller.deinit();
                    self.motor_controller = null;
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

        // Extract motor pins from config
        const motors_array = parsed.value.object.get("motors").?.array;
        var config = try self.allocator.alloc(MotorController.Config, motors_array.items.len);
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

        // Initialize motor controller
        if (self.motor_controller) |controller| {
            controller.deinit();
            self.motor_controller = null;
        }

        self.motor_controller = try MotorController.init(self.allocator, config, battery_type, protocol);
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

                try self.motor_controller.?.armMotor(motor_idx);
            },
            .Disarm => {
                const motor_idx = try std.fmt.parseInt(
                    usize,
                    iterator.next() orelse return error.InvalidDisarmMotor,
                    10,
                );

                try self.motor_controller.?.disarmMotor(motor_idx);
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
                try self.motor_controller.?.setMotorSpeed(motor_idx, speed);
            },
            .ReverseDirection => {
                const motor_idx = try std.fmt.parseInt(
                    usize,
                    iterator.next() orelse return error.InvalidReverseMotor,
                    10,
                );

                try self.motor_controller.?.reverseMotorDirection(motor_idx);
            },
            .Battery => {
                const voltage_str = iterator.next() orelse return error.InvalidVoltageStr;
                const voltage = std.fmt.parseFloat(f32, voltage_str) catch |err| {
                    std.debug.print("Failed to parse voltage str: {s}. Err => {any}\n", .{ voltage_str, err });
                    return;
                };

                if (self.motor_controller) |controller| {
                    controller.updateBatteryVoltage(voltage);
                }
                return;
            },
            .UpdateOrientation => {},
            .SetOrientation => {},
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
        while (self.running) {
            const state = self.state.load(.acquire);
            if (state == .Ready or state == .Running) {
                const last_heartbeat = self.last_heartbeat.load(.acquire);
                const current_time = std.time.timestamp();

                if (last_heartbeat != 0 and current_time - last_heartbeat > HEARTBEAT_TIMEOUT_SEC) {
                    std.debug.print("Client heartbeat timeout. Curr: {d} Last: {d}  => Diff: {d}\n", .{ current_time, last_heartbeat, current_time - last_heartbeat });
                    self.state.store(.Failed, .release);
                    if (self.motor_controller) |controller| {
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
