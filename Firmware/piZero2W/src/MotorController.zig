const std = @import("std");
const time = std.time;
const fs = std.fs;
const net = std.net;
const json = std.json;
const heap = std.heap;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const Atomic = std.atomic;

// DShot600 Constants
const DSHOT_MIN_THROTTLE: u16 = 48;
const DSHOT_MAX_THROTTLE: u16 = 2047;
const GPIO_BASE: usize = 0x3F200000; // Raspberry Pi zero 2W GPIO base address

// DShot300 Timing (in nanoseconds)
const DSHOT_BIT_TIME: u64 = 2700; // Overall bit time for DShot300
const T0H_TIME: u64 = DSHOT_BIT_TIME * 3 / 9; // High time for 0 bit
const FRAME_RESET_TIME: u64 = 30000; // Time between frames

// Precise timing sleep function using nanosleep
fn getNanoTime() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC_RAW, &ts);
    return @as(u64, @intCast(ts.sec)) * 1000000000 + @as(u64, @intCast(ts.nsec));
}

fn busyWaitUntil(target_time: u64) void {
    while (getNanoTime() < target_time) {
        // Tight loop for precise timing
        std.atomic.spinLoopHint();
    }
}

// Set real-time priority
fn setRealtimePriority() !void {
    var sched = std.os.linux.sched_param{ .priority = 99 };
    const ret = std.os.linux.sched_setscheduler(0, std.os.linux.SCHED{ .mode = .FIFO }, &sched);

    if (ret != 0) {
        return error.SetSchedulerFailed;
    }
}

// Pin process to specific CPU core
fn pinToCore(core: u32) !void {
    var set: std.os.linux.cpu_set_t = [_]usize{0} ** (std.os.linux.CPU_SETSIZE / @sizeOf(usize));

    const word_index = core / @bitSizeOf(usize);
    const bit_index = core % @bitSizeOf(usize);
    set[word_index] |= @as(usize, 1) << @as(u6, @intCast(bit_index));

    try std.os.linux.sched_setaffinity(0, &set);
}

// Motor Configuration
const Motor = struct {
    pin: u8,
    throttle: Atomic.Value(u16),
    armed: Atomic.Value(bool),
    mutex: Mutex,

    pub fn init(pin: u8) Motor {
        return Motor{
            .pin = pin,
            .throttle = Atomic.Value(u16).init(DSHOT_MIN_THROTTLE),
            .armed = Atomic.Value(bool).init(false),
            .mutex = Mutex{},
        };
    }
};

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
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, pins: []const u8) !*Self {
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
        var motors = try allocator.alloc(Motor, pins.len);
        for (pins, 0..) |pin, i| {
            motors[i] = Motor.init(pin);
            // Configure GPIO as output
            const reg_idx = @divFloor(pin, 10);
            const bit_idx = @mod(pin, 10) * 3;
            const gpio_mem_ptr: *volatile GpioRegs = @ptrCast(@alignCast(gpio_mem));
            gpio_mem_ptr.gpfsel[reg_idx] &= ~(@as(u32, 0b111) << @as(u5, @intCast(bit_idx)));
            gpio_mem_ptr.gpfsel[reg_idx] |= @as(u32, 0b001) << @as(u5, @intCast(bit_idx)); // Set as output
        }

        const controller = try allocator.create(Self);
        controller.* = Self{
            .motors = motors,
            .gpio_mem = @ptrCast(@alignCast(gpio_mem)),
            .running = Atomic.Value(bool).init(true),
            .allocator = allocator,
        };

        return controller;
    }

    pub fn deinit(self: *Self) void {
        self.running.store(false, .release);
        self.allocator.free(self.motors);
        // Unmap GPIO memory
        _ = std.os.linux.munmap(@as([*]u8, @ptrCast(@volatileCast(self.gpio_mem))), @sizeOf(GpioRegs));
    }

    fn dshotChecksum(value: u16) u8 {
        return @truncate((value ^ (value >> 4) ^ (value >> 8)) & 0x0F);
    }

    fn createDshotPacket(throttle: u16) u16 {
        const clamped_throttle = std.math.clamp(throttle, DSHOT_MIN_THROTTLE, DSHOT_MAX_THROTTLE);
        const packet = (clamped_throttle << 1); // Append telemetry bit (0)
        return (packet << 4) | dshotChecksum(packet);
    }

    fn sendDshotCommand(self: *Self, pin: u8, value: u16) void {
        const packet = createDshotPacket(value);
        const reg_idx = @divFloor(pin, 32);
        const bit_mask = @as(u32, 1) << @as(u5, @intCast(@mod(pin, 32)));

        // Disable interrupts for precise timing
        var old_mask: std.os.linux.sigset_t = undefined;
        _ = std.os.linux.sigprocmask(std.os.linux.SIG.BLOCK, null, &old_mask);
        defer _ = std.os.linux.sigprocmask(std.os.linux.SIG.SETMASK, &old_mask, null);

        var start_time = getNanoTime();
        var bit_time: u64 = undefined;

        // Send 16 bits
        var i: u8 = 0;
        while (i < 16) : (i += 1) {
            const bit = (packet >> (15 - @as(u4, @intCast(i)))) & 1;

            bit_time = start_time;

            // Set bit high
            self.gpio_mem.gpset[reg_idx] = bit_mask;

            if (bit == 1) {
                // Logic 1: 1.25μs high, 0.42μs low
                busyWaitUntil(bit_time + (DSHOT_BIT_TIME * 3) / 4);
            } else {
                // Logic 0: ~37.5% duty cycle
                busyWaitUntil(bit_time + T0H_TIME);
            }

            // Set bit low
            self.gpio_mem.gpclr[reg_idx] = bit_mask;

            // Wait for the rest of the bit period
            start_time += DSHOT_BIT_TIME;
            busyWaitUntil(start_time);
        }

        // Frame reset time
        busyWaitUntil(start_time + FRAME_RESET_TIME);
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
            self.sendDshotCommand(motor.pin, 0);
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

        motor.armed.store(false, .release);
        motor.throttle.store(DSHOT_MIN_THROTTLE, .release);
        std.debug.print("Motor {d} disarmed\n", .{motor_idx});
    }

    pub fn setMotorSpeed(self: *Self, motor_idx: usize, speed_percent: f32) !void {
        if (motor_idx >= self.motors.len) return error.InvalidMotorIndex;

        var motor = &self.motors[motor_idx];
        if (!motor.armed.load(.acquire)) return error.MotorNotArmed;

        const throttle_range = DSHOT_MAX_THROTTLE - DSHOT_MIN_THROTTLE;
        const throttle_increment = @as(u16, @intFromFloat((speed_percent / 100.0) * @as(f32, @floatFromInt(throttle_range))));
        const throttle = DSHOT_MIN_THROTTLE + throttle_increment;

        motor.throttle.store(throttle, .release);
        std.debug.print("Motor {d} speed set to {d:.1}%\n", .{ motor_idx, speed_percent });
    }

    fn motorControlThread(self: *Self) void {
        while (self.running.load(.acquire)) {
            for (self.motors) |*motor| {
                if (motor.armed.load(.acquire)) {
                    const throttle = motor.throttle.load(.acquire);
                    self.sendDshotCommand(motor.pin, throttle);
                }
            }
        }
    }

    // pub fn testPinOutput(self: *MotorController, pin: u8, frequency: u32, duration_ms: u32) !void {
    //     _ = duration_ms;
    //     // Calculate one full period in nanoseconds (1s / frequency).
    //     const period_ns = 1_000_000_000 / frequency;
    //     const half_period_ns = period_ns / 2;

    //     // Determine when to stop toggling (monotonic time).

    //     const reg_idx = @divFloor(pin, 32);
    //     const bit_mask = @as(u32, 1) << @as(u5, @intCast(@mod(pin, 32)));

    //     // Toggle until we reach end_time
    //     while (true) {
    //         // Set GPIO pin high
    //         self.gpio_mem.gpset[reg_idx] = bit_mask;
    //         nanosleep(half_period_ns);

    //         // Set GPIO pin low
    //         self.gpio_mem.gpclr[reg_idx] = bit_mask;
    //         nanosleep(half_period_ns);
    //     }

    //     // Once done, ensure pin is cleared
    //     self.gpio_mem.gpclr[reg_idx] = bit_mask;
    //     std.debug.print("Finished toggling pin {d}\n", .{pin});
    // }
};

const Commands = enum {
    Arm,
    Disarm,
    SetSpeed,
    SetDirection,
};

// TCP Server for remote control
const Command = struct {
    motor_idx: usize,
    command_type: Commands,
    speed: ?f32 = null,
};

const MOTOR_0 = 6;
const MOTOR_1 = 5;
const MOTOR_2 = 22;
const MOTOR_3 = 27;

pub const ServerState = enum(u8) {
    Idle,
    WaitingForClient,
    Connected,
    ConfigSync,
    Ready,
    Running,
    Failed,
};

pub const ServerError = error{
    SocketCreationFailed,
    BindFailed,
    ConfigParseError,
    InvalidConfig,
    MotorInitFailed,
};

const Server = struct {
    const Self = @This();
    const HEARTBEAT_TIMEOUT_SEC = 2;

    state: Atomic.Value(ServerState),
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
            .state = Atomic.Value(ServerState).init(.Idle),
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
        self.state.store(.WaitingForClient, .release);

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
                std.debug.print("Receive error: {}\n", .{err});
                continue;
            };

            if (len == 0) continue;

            const msg = buf[0..len];
            try self.handleMessage(msg, client_addr, client_addr_len);
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
                    // try self.sendResponse("OK\n", client_addr, addr_len);
                }
            },
            else => {},
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

        // Extract motor pins from config
        const motors_array = parsed.value.object.get("motors").?.array;
        var pins = try self.allocator.alloc(u8, motors_array.items.len);
        defer self.allocator.free(pins);

        for (motors_array.items, 0..) |motor, i| {
            pins[i] = @intCast(motor.object.get("pin").?.integer);
        }

        // TODO: Implement different ability to select protocols
        // const protocol = parsed.value.object.get("").?.

        // Initialize motor controller
        if (self.motor_controller) |controller| {
            controller.deinit();
        }

        self.motor_controller = try MotorController.init(self.allocator, pins);
        // Start motor control thread
        _ = try Thread.spawn(.{}, MotorController.motorControlThread, .{self.motor_controller.?});
    }

    fn handleCommand(self: *Self, cmd_str: []const u8) !void {
        var iterator = std.mem.splitSequence(u8, cmd_str, " ");
        const parsed_cmd = iterator.next() orelse return error.NoCommandReceived;

        const cmd: ?Commands = std.meta.stringToEnum(Commands, parsed_cmd);
        if (cmd == null) return error.InvalidCommand;

        const motor_idx = try std.fmt.parseInt(
            usize,
            iterator.next() orelse return error.InvalidCommand,
            10,
        );

        // TODO: Implement SetDirection & conditions to only allow certain commands only during specific states
        switch (cmd.?) {
            .Arm => try self.motor_controller.?.armMotor(motor_idx),
            .Disarm => try self.motor_controller.?.disarmMotor(motor_idx),
            .SetSpeed => {
                const speed = try std.fmt.parseFloat(
                    f32,
                    iterator.next() orelse return error.InvalidCommand,
                );
                try self.motor_controller.?.setMotorSpeed(motor_idx, speed);
            },
            .SetDirection => @panic("Not implemented yet!"),
        }
    }

    fn sendResponse(self: *Self, response: []const u8, client_addr: std.posix.sockaddr, addr_len: std.posix.socklen_t) !void {
        _ = try std.posix.sendto(
            self.socket.?,
            response,
            0,
            &client_addr,
            addr_len,
        );
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
                            controller.disarmMotor(i) catch {};
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
