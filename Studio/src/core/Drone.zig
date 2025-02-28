//Drone.zig
const std = @import("std");
const Math = @import("Math.zig");
const Sensors = @import("Sensors.zig");
const Vec3 = Math.Vec3;

pub const Battery = enum(u8) {
    Lipo_2S = 2, // 2 cells in series (7.4V nominal)
    Lipo_3S = 3, // 3 cells in series (11.1V nominal)
    Lipo_4S = 4, // 4 cells in series (14.8V nominal)
    Lipo_5S = 5, // 5 cells in series (18.5V nominal)
    Lipo_6S = 6, // 6 cells in series (22.2V nominal)

    // Get cell count for voltage calculations
    pub fn cellCount(self: Battery) u8 {
        return @intFromEnum(self);
    }

    // Get minimum safe voltage (3.4V per cell)
    pub fn minVoltage(self: Battery) f32 {
        return 3.4 * @as(f32, @floatFromInt(self.cellCount()));
    }

    // Get maximum voltage (4.2V per cell when fully charged)
    pub fn maxVoltage(self: Battery) f32 {
        return 4.2 * @as(f32, @floatFromInt(self.cellCount()));
    }

    // Calculate battery percentage based on current voltage
    pub fn calculatePercentage(self: Battery, voltage: f32) f32 {
        const min_v = self.minVoltage();
        const max_v = self.maxVoltage();

        // Normalize between min and max voltages
        const percentage = (voltage - min_v) / (max_v - min_v) * 100.0;
        return @max(0.0, @min(100.0, percentage)); // Clamp between 0-100%
    }
};

pub const RotationDirection = enum(u8) {
    Clockwise = 0,
    Counterclockwise = 1,
};

pub const DSHOT = enum(u32) {
    DSHOT_150 = 150,
    DSHOT_300 = 300,
    DSHOT_600 = 600,

    pub const MIN_THROTTLE: u16 = 48;
    pub const MAX_THROTTLE: u16 = 2047;

    /// Special DShot commands
    pub const CMD = struct {
        pub const MOTOR_STOP: u16 = 0;
        pub const BEACON1: u16 = 1;
        pub const BEACON2: u16 = 2;
        pub const BEACON3: u16 = 3;
        pub const BEACON4: u16 = 4;
        pub const BEACON5: u16 = 5;
        pub const ESC_INFO: u16 = 6;
        pub const SPIN_DIRECTION_1: u16 = 7; // Set normal direction
        pub const SPIN_DIRECTION_2: u16 = 8; // Set reverse direction
        pub const @"3D_MODE_OFF": u16 = 9;
        pub const @"3D_MODE_ON": u16 = 10;
        pub const SETTINGS_REQUEST: u16 = 11;
        pub const SAVE_SETTINGS: u16 = 12;
        pub const SPIN_DIRECTION_NORMAL: u16 = 20;
        pub const SPIN_DIRECTION_REVERSED: u16 = 21;
        pub const LED0_ON: u16 = 22;
        pub const LED1_ON: u16 = 23;
        pub const LED2_ON: u16 = 24;
        pub const LED3_ON: u16 = 25;
        pub const LED0_OFF: u16 = 26;
        pub const LED1_OFF: u16 = 27;
        pub const LED2_OFF: u16 = 28;
        pub const LED3_OFF: u16 = 29;

        pub const REPEAT_COUNT: u32 = 10; // Number of times to repeat direction command
    };

    /// Get the total bit time in nanoseconds for this protocol
    pub fn bit_time(self: DSHOT) u64 {
        return switch (self) {
            .DSHOT_150 => 6670, // 150kbps -> ~6.67µs per bit
            .DSHOT_300 => 3330, // 300kbps -> ~3.33µs per bit
            .DSHOT_600 => 1670, // 600kbps -> ~1.67µs per bit
        };
    }

    /// Get the high time for a zero bit (T0H) in nanoseconds
    pub fn t0h_time(self: DSHOT) u64 {
        // T0H is typically ~37.5% of the bit time
        const t = self.bit_time();
        return (t * 3) / 8;
    }

    /// Get the high time for a one bit (T1H) in nanoseconds
    pub fn t1h_time(self: DSHOT) u64 {
        // T1H is typically ~75% of the bit time
        const t = self.bit_time();
        return (t * 3) / 4;
    }

    /// Get the frame reset time (gap between packets) in nanoseconds
    pub fn frame_reset_time(self: DSHOT) u64 {
        return switch (self) {
            .DSHOT_150 => 30000, // 30µs
            .DSHOT_300 => 25000, // 25µs
            .DSHOT_600 => 20000, // 20µs
        };
    }

    pub fn checksum(value: u16) u8 {
        return @truncate((value ^ (value >> 4) ^ (value >> 8)) & 0x0F);
    }

    pub fn create_packet(value: u16, telemetry: bool) u16 {
        // Bits 0-10: throttle/command value
        // Bit 11: telemetry request
        const payload = (value << 1) | @intFromBool(telemetry);
        // Bits 12-15: checksum
        return (payload << 4) | checksum(payload);
    }

    pub fn percentage_to_throttle(percentage: f32) u16 {
        const clamped = @min(100.0, @max(0.0, percentage));
        const throttle_range = MAX_THROTTLE - MIN_THROTTLE;
        const throttle_value = MIN_THROTTLE +
            @as(u16, @intFromFloat((clamped / 100.0) * @as(f32, @floatFromInt(throttle_range))));

        return @min(MAX_THROTTLE, @max(MIN_THROTTLE, throttle_value));
    }

    pub fn throttle_to_percentage(throttle: u16) f32 {
        const clamped = @min(MAX_THROTTLE, @max(MIN_THROTTLE, throttle));
        return @as(f32, @floatFromInt(clamped - MIN_THROTTLE)) /
            @as(f32, @floatFromInt(MAX_THROTTLE - MIN_THROTTLE)) * 100.0;
    }
};

/// Precise timing utilities for DShot protocol
pub const TimingUtils = struct {
    /// Get current time in nanoseconds with high precision
    pub fn getNanoTime() u64 {
        var ts: std.os.linux.timespec = undefined;
        _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC_RAW, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    /// Busy-wait until a specific target time, for precise timing
    pub fn busyWaitUntil(target_time: u64) void {
        while (getNanoTime() < target_time) {
            // Tight loop for precise timing
            std.atomic.spinLoopHint();
        }
    }

    /// Set real-time priority for the current process
    pub fn setRealtimePriority() !void {
        var sched = std.os.linux.sched_param{ .priority = 99 };
        const ret = std.os.linux.sched_setscheduler(0, std.os.linux.SCHED{ .mode = .FIFO }, &sched);

        if (ret != 0) {
            return error.SetSchedulerFailed;
        }
    }

    /// Pin process to a specific CPU core for better timing performance
    pub fn pinToCore(core: u32) !void {
        var set: std.os.linux.cpu_set_t = [_]usize{0} ** (std.os.linux.CPU_SETSIZE / @sizeOf(usize));

        const word_index = core / @bitSizeOf(usize);
        const bit_index = core % @bitSizeOf(usize);
        set[word_index] |= @as(usize, 1) << @as(u6, @intCast(bit_index));

        try std.os.linux.sched_setaffinity(0, &set);
    }
};

pub const Protocol = struct {
    pub const Motors = enum {
        Motor_1,
        Motor_2,
        Motor_3,
        Motor_4,
    };

    pub const CommandType = enum {
        Arm,
        Disarm,
        SetSpeed,
        ReverseDirection,
        Battery,
        UpdateOrientation,
        SetOrientation,
    };

    pub const Command = struct {
        const Self = @This();

        type: CommandType,
        motor: ?Motors = null,
        speed: ?f32 = null,
        pose: ?Math.Quaternion = null,

        pub fn is_valid(self: Self) bool {
            switch (self.type) {
                .SetSpeed => {
                    if (self.motor == null or self.speed == null) return false;

                    const speed = self.speed.?;
                    return speed >= 0 and speed <= 100.0;
                },
                .UpdateOrientation => {
                    if (self.pose == null) return false;

                    const pose = self.pose.?;
                    return !std.math.isNan(pose.x) and
                        !std.math.isNan(pose.y) and
                        !std.math.isNan(pose.z) and
                        !std.math.isNan(pose.w);
                },
                .SetOrientation => {
                    if (self.pose == null) return false;

                    const pose = self.pose.?;
                    return !std.math.isNan(pose.x) and
                        !std.math.isNan(pose.y) and
                        !std.math.isNan(pose.z) and
                        !std.math.isNan(pose.w);
                },
                .ReverseDirection => return self.motor != null,
                .Arm => return self.motor != null,
                .Disarm => return self.motor != null,
                .Battery => return true,
            }
        }

        pub fn generate_cmd_str(
            self: Self,
            allocator: std.mem.Allocator,
        ) ![]u8 {
            if (!self.is_valid()) return error.InvalidCommand;

            const cmd = switch (self.type) {
                .SetSpeed => blk: {
                    break :blk std.fmt.allocPrint(
                        allocator,
                        "SetSpeed {d} {d:.1}\n",
                        .{
                            @intFromEnum(self.motor.?),
                            self.speed.?,
                        },
                    ) catch |err| {
                        std.debug.print("Failed to format SetSpeed: {any}\n", .{err});
                        return error.FailedToGenerateCommand;
                    };
                },
                .ReverseDirection => blk: {
                    break :blk std.fmt.allocPrint(
                        allocator,
                        "ReverseDirection {d}\n",
                        .{@intFromEnum(self.motor.?)},
                    ) catch |err| {
                        std.debug.print("Failed to format ReverseDirection: {any}\n", .{err});
                        return error.FailedToGenerateCommand;
                    };
                },
                .Arm => blk: {
                    break :blk std.fmt.allocPrint(
                        allocator,
                        "Arm {d}\n",
                        .{@intFromEnum(self.motor.?)},
                    ) catch |err| {
                        std.debug.print("Failed to format Arm command: {any}\n", .{err});
                        return error.FailedToGenerateCommand;
                    };
                },
                .Disarm => blk: {
                    break :blk std.fmt.allocPrint(
                        allocator,
                        "Disarm {d}\n",
                        .{@intFromEnum(self.motor.?)},
                    ) catch |err| {
                        std.debug.print("Failed to format Disarm command: {any}\n", .{err});
                        return error.FailedToGenerateCommand;
                    };
                },
                .UpdateOrientation => blk: {
                    break :blk std.fmt.allocPrint(
                        allocator,
                        "UpdateOrientation {d:.3} {d:.3} {d:.3} {d:.3}\n",
                        .{
                            self.pose.?.w,
                            self.pose.?.x,
                            self.pose.?.y,
                            self.pose.?.z,
                        },
                    ) catch |err| {
                        std.debug.print("Failed to format Orientation command: {any}\n", .{err});
                        return error.FailedToGenerateCommand;
                    };
                },
                .SetOrientation => blk: {
                    break :blk std.fmt.allocPrint(
                        allocator,
                        "SetOrientation {d:.3} {d:.3} {d:.3} {d:.3}\n",
                        .{
                            self.pose.?.x,
                            self.pose.?.y,
                            self.pose.?.z,
                            self.pose.?.w,
                        },
                    ) catch |err| {
                        std.debug.print("Failed to format SetOrientation command: {any}\n", .{err});
                        return error.FailedToGenerateCommand;
                    };
                },
                .Battery => unreachable,
            };

            return cmd;
        }

        pub fn parse_cmd(cmd_str: []const u8) Self {
            _ = cmd_str;
            @panic("Not implemented yet!");
        }

        pub fn execute_cmd(self: Self) void {
            _ = self;
            @panic("Not implemented yet!");
        }
    };

    pub const MotorState = struct {
        throttle: f32 = 0.0,
        armed: bool = false,
        direction: RotationDirection,
    };

    pub const ConnectionError = error{
        SocketCreationFailed,
        BindFailed,
        SendFailed,
        ConfigSyncFailed,
        AckTimeout,
        InvalidState,
    };

    pub const ServerState = enum(u8) {
        WaitingForClient,
        Connected,
        ConfigSync,
        Ready,
        Running,
        Failed,

        pub fn toString(self: ClientState) []const u8 {
            return switch (self) {
                .WaitingForClient => "Waiting for client...",
                .Connected => "Connected",
                .ConfigSync => "Syncing Config...",
                .Ready => "Ready",
                .Running => "Running",
                .Failed => "Failed",
            };
        }

        pub fn action(self: ServerState) void {
            _ = self;
            @panic("Not Implemented yet!");
        }
    };

    pub const ClientState = enum(u8) {
        Disconnected,
        Connecting,
        Connected,
        ConfigSync,
        Ready,
        Running,
        Failed,

        pub fn toString(self: ClientState) []const u8 {
            return switch (self) {
                .Disconnected => "Disconnected",
                .Connecting => "Connecting...",
                .Connected => "Connected",
                .ConfigSync => "Syncing Config...",
                .Ready => "Ready",
                .Running => "Running",
                .Failed => "Failed",
            };
        }

        pub fn action(self: ServerState) void {
            _ = self;
            @panic("Not Implemented yet!");
        }
    };
};

pub const ConnectionHandler = struct {
    const Self = @This();

    const ACK_TIMEOUT_MS = 500;
    const MAX_RETRIES = 1;
    const MAX_FAILURES = 1;
    const SYNC_INTERVAL_MS = 100;

    pub const ConnectionError = error{
        SocketCreationFailed,
        BindFailed,
        SendFailed,
        ConfigSyncFailed,
        AckTimeout,
        InvalidState,
    };

    pub const BatteryInfo = struct {
        voltage: f32 = 0.0,
        percentage: f32 = 0.0,
        failsafe_active: bool = true,
        type: Battery,
    };

    socket: ?std.posix.socket_t,
    local_addr: std.net.Address,
    server_addr: std.net.Address,
    state: std.atomic.Value(Protocol.ClientState),
    motor_states: *[@typeInfo(Protocol.Motors).@"enum".fields.len]Protocol.MotorState,
    config: *DroneConfig,
    battery_info: BatteryInfo,
    running: bool,
    retries: u8,
    failures: u8,
    allocator: std.mem.Allocator,
    sync_thread: ?std.Thread,
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator, config: *DroneConfig, motor_states: *[@typeInfo(Protocol.Motors).@"enum".fields.len]Protocol.MotorState) !*Self {
        const self = try allocator.create(Self);

        self.* = .{
            .socket = null,
            .local_addr = try std.net.Address.parseIp4(config.local_ip, config.local_port),
            .server_addr = try std.net.Address.parseIp4(config.controller_ip, config.controller_port),
            .motor_states = motor_states,
            .state = std.atomic.Value(Protocol.ClientState).init(.Disconnected),
            .config = config,
            .battery_info = BatteryInfo{ .type = config.battery },
            .running = false,
            .retries = 0,
            .failures = 0,
            .allocator = allocator,
            .sync_thread = null,
            .mutex = .{},
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.stop();
        if (self.sync_thread) |thread| {
            thread.join();
        }
        self.closeSocket();
        self.allocator.destroy(self);
    }

    fn closeSocket(self: *Self) void {
        if (self.socket) |sock| {
            std.posix.close(sock);
            self.socket = null;
        }
    }

    pub fn start(self: *Self) !void {
        if (self.running) return;

        self.running = true;
        self.state.store(.Disconnected, .release);
        self.retries = 0;
        self.sync_thread = try std.Thread.spawn(.{}, syncThread, .{self});
    }

    pub fn stop(self: *Self) void {
        if (!self.running) return;

        self.running = false;
        self.closeSocket();
        self.state.store(.Disconnected, .release);
    }

    pub fn retry(self: *Self) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state.load(.acquire) != .Failed) return error.InvalidState;

        self.closeSocket();
        self.retries = 0;
        self.state.store(.Connecting, .release);
    }

    fn syncThread(self: *Self) void {
        while (self.running) {
            switch (self.state.load(.acquire)) {
                .Disconnected => {
                    self.initializeConnection() catch |err| {
                        std.debug.print("Connection initialization failed: {any}\n", .{err});
                        self.handleFailure("Socket initialization failed");
                        continue;
                    };
                    self.state.store(.Connecting, .release);
                },
                .Connecting => {
                    self.sendConnectionRequest() catch |err| {
                        std.debug.print("Connection request failed: {any}\n", .{err});

                        if (self.retries >= MAX_RETRIES) {
                            self.retries = 0;
                            self.handleFailure("Connection failed after max retries");
                        }

                        std.time.sleep(1 * std.time.ns_per_s);
                        continue;
                    };
                    self.state.store(.Connected, .release);
                },
                .Connected => {
                    self.sendConfig() catch |err| {
                        std.debug.print("Config sync failed: {any}\n", .{err});
                        self.handleFailure("Config sync failed");
                        continue;
                    };
                    self.state.store(.ConfigSync, .release);
                },
                .ConfigSync => {
                    self.waitForConfigAck() catch |err| {
                        std.debug.print("Config ack failed: {any}\n", .{err});
                        self.handleFailure("Config acknowledgment failed");
                        continue;
                    };
                    self.state.store(.Ready, .release);
                },
                .Ready, .Running => {
                    self.sendHeartbeat() catch |err| switch (err) {
                        error.WouldBlock => {
                            // Non-blocking failure, just continue
                            std.time.sleep(SYNC_INTERVAL_MS * std.time.ns_per_ms);
                            continue;
                        },
                        else => {
                            std.debug.print("Heartbeat failed: {any}\n", .{err});
                            self.handleFailure("Lost connection to server");
                            continue;
                        },
                    };
                    std.time.sleep(SYNC_INTERVAL_MS * std.time.ns_per_ms);
                },
                .Failed => {
                    std.time.sleep(1000 * std.time.ns_per_ms); // Wait before retry
                },
            }
        }
    }

    fn initializeConnection(self: *Self) !void {
        self.closeSocket();

        const sock = try std.posix.socket(
            std.posix.AF.INET,
            std.posix.SOCK.DGRAM,
            std.posix.IPPROTO.UDP,
        );
        errdefer std.posix.close(sock);

        // Set socket timeout for receiving
        const timeout = std.posix.timeval{
            .sec = 1,
            .usec = ACK_TIMEOUT_MS * 1000,
        };

        // Set socket timeout
        try std.posix.setsockopt(
            sock,
            std.posix.SOL.SOCKET,
            std.posix.SO.RCVTIMEO,
            &std.mem.toBytes(timeout),
        );

        // Set socket reuse address option
        try std.posix.setsockopt(
            sock,
            std.posix.SOL.SOCKET,
            std.posix.SO.REUSEADDR,
            &std.mem.toBytes(@as(c_int, 1)),
        );

        try std.posix.bind(
            sock,
            &self.local_addr.any,
            self.local_addr.getOsSockLen(),
        );

        self.socket = sock;
    }

    fn sendConnectionRequest(self: *Self) !void {
        self.retries += 1;

        if (self.socket == null) return error.SocketNotInitialized;

        const msg = "CONNECT\n";
        _ = try std.posix.sendto(
            self.socket.?,
            msg,
            0,
            &self.server_addr.any,
            self.server_addr.getOsSockLen(),
        );

        var buf: [128]u8 = undefined;
        const len = try std.posix.recvfrom(
            self.socket.?,
            &buf,
            0,
            null,
            null,
        );

        if (!std.mem.eql(u8, buf[0..len], "ACK\n")) {
            return error.AckTimeout;
        }
    }

    fn sendConfig(self: *Self) !void {
        if (self.socket == null) return error.SocketNotInitialized;

        // Serialize config to JSON
        const config_json = try self.config.toJson(self.allocator);
        defer self.allocator.free(config_json);

        _ = try std.posix.sendto(
            self.socket.?,
            config_json,
            0,
            &self.server_addr.any,
            self.server_addr.getOsSockLen(),
        );
    }

    fn waitForConfigAck(self: *Self) !void {
        if (self.socket == null) return error.SocketNotInitialized;

        var buf: [128]u8 = undefined;
        const len = try std.posix.recvfrom(
            self.socket.?,
            &buf,
            0,
            null,
            null,
        );

        if (!std.mem.eql(u8, buf[0..len], "CONFIG_ACK\n")) {
            return error.ConfigSyncFailed;
        }
    }

    fn updateMotorStates(self: *Self, status_msg: []const u8) void {
        var iter = std.mem.splitSequence(u8, status_msg, " ");
        _ = iter.next(); // Skip "STATUS"

        var i: usize = 0;
        while (i < self.motor_states.len) : (i += 1) {
            if (iter.next()) |motor_idx_str| {
                const motor_idx = std.fmt.parseInt(usize, motor_idx_str, 10) catch continue;
                const armed_str = iter.next() orelse break;
                const throttle_str = iter.next() orelse break;

                if (motor_idx < self.motor_states.len) {
                    self.motor_states[motor_idx].armed = (std.fmt.parseInt(u8, armed_str, 10) catch 0) != 0;
                    const raw_throttle = std.fmt.parseInt(u16, throttle_str, 10) catch 0;
                    // Convert from DShot value to percentage
                    self.motor_states[motor_idx].throttle = @floatFromInt(raw_throttle - DSHOT.MIN_THROTTLE);
                    self.motor_states[motor_idx].throttle = self.motor_states[motor_idx].throttle / @as(f32, DSHOT.MAX_THROTTLE - DSHOT.MIN_THROTTLE) * 100.0;
                }
            }
        }
    }

    fn parseBatteryStatus(self: *Self, status_msg: []const u8) void {
        if (std.mem.indexOf(u8, status_msg, "BATTERY")) |battery_idx| {
            const battery_part = status_msg[battery_idx..];

            var iter = std.mem.splitSequence(u8, battery_part, " ");
            _ = iter.next();

            if (iter.next()) |voltage_str| {
                self.battery_info.voltage = std.fmt.parseFloat(f32, voltage_str) catch |err| {
                    std.debug.print("Failed to parse voltage_str {s}! Err => {any}\n", .{ voltage_str, err });
                    return;
                };
            }

            // Parse percentage
            if (iter.next()) |percentage_str| {
                self.battery_info.percentage = std.fmt.parseFloat(f32, percentage_str) catch |err| {
                    std.debug.print("Failed to parse percentage_str {s}! Err => {any}\n", .{ percentage_str, err });
                    return;
                };
            }

            // Parse failsafe status
            if (iter.next()) |raw_failsafe_str| {
                const failsafe_str = std.mem.trim(u8, raw_failsafe_str, &std.ascii.whitespace);
                const failsafe_val = std.fmt.parseInt(u8, failsafe_str, 10) catch |err| {
                    std.debug.print("Failed to parse failsafe_val {s}! Err => {any}\n", .{ failsafe_str, err });
                    return;
                };

                self.battery_info.failsafe_active = switch (failsafe_val) {
                    1 => true,
                    else => false,
                };
            }

            // Log battery status if percentage is below critical threshold or failsafe is active
            if (self.battery_info.percentage < 15.0 or self.battery_info.failsafe_active) {
                std.debug.print("BATTERY ALERT: {d:.2}V / {d:.1}% - Failsafe: {any}\n", .{
                    self.battery_info.voltage,
                    self.battery_info.percentage,
                    self.battery_info.failsafe_active,
                });
            }
        }
    }

    fn handleHeartbeatResponse(self: *Self, response: []const u8) !void {
        std.debug.print("Heartbeat response: {s}\n", .{response});
        if (std.mem.startsWith(u8, response, "STATUS ")) {
            self.updateMotorStates(response);
            self.parseBatteryStatus(response);
        } else if (!std.mem.eql(u8, response, "ACK\n")) {
            return error.InvalidResponse;
        }
    }

    fn sendHeartbeat(self: *Self) !void {
        if (self.socket == null) return error.SocketNotInitialized;

        const msg = "HEARTBEAT\n";
        _ = try std.posix.sendto(
            self.socket.?,
            msg,
            0,
            &self.server_addr.any,
            self.server_addr.getOsSockLen(),
        );

        var buf: [512]u8 = undefined;
        const len = try std.posix.recvfrom(
            self.socket.?,
            &buf,
            0,
            null,
            null,
        );

        // Process the response using handleHeartbeatResponse
        try self.handleHeartbeatResponse(buf[0..len]);
    }

    fn handleFailure(self: *Self, msg: []const u8) void {
        self.failures += 1;
        std.debug.print("Connection error: {s}\n", .{msg});
        self.closeSocket();

        if (self.failures >= MAX_FAILURES) {
            std.debug.print("Reached Maximum number of failures. Disconnecting from server...\n", .{});
            self.state.store(.Disconnected, .release);
            self.failures = 0;
            return;
        }

        self.state.store(.Failed, .release);
    }
};

pub const MotorControllerClient = struct {
    const Self = @This();

    udp_thread: ?std.Thread = null,
    orientation_thread: ?std.Thread = null,
    config: *DroneConfig,
    motor_states: [@typeInfo(Protocol.Motors).@"enum".fields.len]Protocol.MotorState,
    command_queue: CommandQueue = CommandQueue.init(),
    prev_command: ?Protocol.Command = null,
    connection_handler: *ConnectionHandler,
    sensor_state: ?*Sensors.SensorState = null,
    allocator: std.mem.Allocator,
    orientation_interval_ms: u64 = 1, // Send orientation updates every 1ms (1khz)
    orientation_running: bool = false, // Flag to control orientation thread
    target_orientation: Math.Quaternion = Math.Quaternion.identity(),

    pub const CommandQueue = struct {

        // Ring buffer for commands
        buffer: [512]Protocol.Command = undefined,
        read_idx: usize = 0,
        write_idx: usize = 0,
        mutex: std.Thread.Mutex = .{},

        pub fn init() @This() {
            var queue = CommandQueue{};
            // Initialize buffer with known valid values
            for (&queue.buffer) |*cmd| {
                cmd.* = Protocol.Command{ .type = .SetSpeed, .motor = .Motor_1, .speed = 0.0 };
            }
            return queue;
        }

        pub fn push(self: *@This(), command: Protocol.Command) bool {
            if (!command.is_valid()) {
                std.debug.print("Invalid command detected: speed={?}\n", .{command.speed});
                return false;
            }

            self.mutex.lock();
            defer self.mutex.unlock();

            const next_write = (self.write_idx + 1) % self.buffer.len;
            if (next_write == self.read_idx) return false;

            self.buffer[self.write_idx] = command;
            self.write_idx = next_write;
            return true;
        }

        pub fn pop(self: *@This()) ?Protocol.Command {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.read_idx >= self.buffer.len) {
                std.debug.print("Read index out of bounds, resetting queue\n", .{});
                self.read_idx = 0;
                self.write_idx = 0;
                return null;
            }

            if (self.read_idx == self.write_idx) return null;
            const command = self.buffer[self.read_idx];

            if (!command.is_valid()) {
                std.debug.print("Invalid command in buffer: type={any}, speed={?}. Resetting queue\n", .{ command.type, command.speed });
                self.read_idx = 0;
                self.write_idx = 0;
                return null;
            }

            self.read_idx = (self.read_idx + 1) % self.buffer.len;
            return command;
        }

        pub fn clear(self: *@This()) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            self.read_idx = 0;
            self.write_idx = 0;
            // Reset all commands to known good state
            for (&self.buffer) |*cmd| {
                cmd.* = Protocol.Command{ .type = .SetSpeed, .motor = .Motor_1, .speed = 0.0 };
            }
        }

        pub fn size(self: *@This()) usize {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.write_idx >= self.read_idx) {
                return self.write_idx - self.read_idx;
            } else {
                return self.buffer.len - (self.read_idx - self.write_idx);
            }
        }
    };

    pub fn init(allocator: std.mem.Allocator, config_path: ?[]const u8) !*Self {
        const config = DroneConfig.loadFromFile(allocator, config_path orelse null) catch
            try DroneConfig.init(allocator);

        const motor_states = blk: {
            const num_motors = @typeInfo(Protocol.Motors).@"enum".fields.len;
            var states: [num_motors]Protocol.MotorState = undefined;

            for (0..num_motors) |i| {
                states[i] = Protocol.MotorState{ .direction = @enumFromInt(i % 2) };
            }

            break :blk states;
        };

        const self = try allocator.create(Self);
        self.* = Self{
            .command_queue = CommandQueue.init(),
            .config = config,
            .motor_states = motor_states,
            .connection_handler = try ConnectionHandler.init(allocator, config, &self.motor_states),
            .allocator = allocator,
        };

        return self;
    }

    pub fn deinit(self: *@This()) void {
        if (self.udp_thread) |thread| {
            self.connection_handler.stop();
            thread.join();
        }
        self.connection_handler.deinit();
        self.config.deinit();
        self.allocator.destroy(self);
    }

    pub fn connect(self: *Self) !void {
        if (self.udp_thread != null) return error.AlreadyConnected;

        try self.connection_handler.start();
        self.udp_thread = try std.Thread.spawn(.{}, udpThread, .{self});

        self.orientation_running = true;
        self.orientation_thread = try std.Thread.spawn(.{}, orientationThread, .{self});
    }

    pub fn disconnect(self: *Self) void {
        self.orientation_running = false;

        if (self.udp_thread) |_| {
            self.connection_handler.stop();
            self.udp_thread.?.join();
            self.udp_thread = null;
        }

        if (self.orientation_thread) |thread| {
            thread.join();
            self.orientation_thread = null;
        }
    }

    pub fn retryConnection(self: *Self) !void {
        try self.connection_handler.retry();
    }

    fn udpThread(self: *Self) void {
        while (self.connection_handler.running) {
            switch (self.connection_handler.state.load(.acquire)) {
                .Ready, .Running => {
                    // Process command queue only when in Ready or Running state
                    if (self.command_queue.pop()) |command| {
                        if (!self.processCommand(command)) {
                            std.debug.print("Failed to process command\n", .{});
                        }
                    }
                },
                .Failed => {
                    // Clear command queue in failed state
                    self.command_queue.clear();
                },
                else => {},
            }
            std.time.sleep(1 * std.time.ns_per_ms);
        }
    }

    fn orientationThread(self: *Self) void {
        var last_orientation: ?Math.Quaternion = null;
        const QUAT_CHANGE_THRESHOLD: f32 = 0.01; // Minimum change threshold to send update

        while (self.orientation_running) {
            switch (self.connection_handler.state.load(.acquire)) {
                .Ready, .Running => {
                    if (self.sensor_state == null) continue;
                    if (self.sensor_state.?.filter) |filter| {
                        const current_orientation = filter.q;

                        // Only send update if orientation has changed significantly or no previous update
                        var should_send = false;
                        if (last_orientation == null) {
                            should_send = true;
                        } else {
                            // Calculate quaternion difference (simplified approach)
                            const dx = @abs(current_orientation.x - last_orientation.?.x);
                            const dy = @abs(current_orientation.y - last_orientation.?.y);
                            const dz = @abs(current_orientation.z - last_orientation.?.z);
                            const dw = @abs(current_orientation.w - last_orientation.?.w);

                            if (dx > QUAT_CHANGE_THRESHOLD or dy > QUAT_CHANGE_THRESHOLD or
                                dz > QUAT_CHANGE_THRESHOLD or dw > QUAT_CHANGE_THRESHOLD)
                            {
                                should_send = true;
                            }
                        }

                        if (should_send) {
                            const command = Protocol.Command{
                                .type = .SetOrientation,
                                .pose = current_orientation,
                            };

                            if (self.command_queue.push(command)) {
                                last_orientation = current_orientation;
                            }
                        }
                    }
                },
                else => {},
            }

            std.time.sleep(self.orientation_interval_ms * std.time.ns_per_ms);
        }
    }

    fn processCommand(self: *Self, command: Protocol.Command) bool {
        std.debug.print("Processing Command => {s}\n", .{@tagName(command.type)});

        const cmd_str = command.generate_cmd_str(self.allocator) catch {
            return false;
        };
        defer self.allocator.free(cmd_str);

        _ = std.posix.sendto(
            self.connection_handler.socket.?,
            cmd_str,
            0,
            &self.connection_handler.server_addr.any,
            self.connection_handler.server_addr.getOsSockLen(),
        ) catch |err| {
            std.debug.print("Failed to send command: {any}\n", .{err});
            return false;
        };

        self.prev_command = command;
        return true;
    }

    pub fn getConnectionState(self: *Self) Protocol.ClientState {
        return self.connection_handler.state.load(.acquire);
    }

    // pub fn handleInput(self: *@This(), up_pressed: bool) void {
    //     std.debug.print("Handling input => Throttle on : {any}\n", .{up_pressed});
    //     const speed: f32 = if (up_pressed) 25.0 else 0.0;
    //     const command = CommandQueue.Command{ .type = .SetSpeed, .motor = .Motor_1, .speed = speed };
    //     _ = self.command_queue.push(command);
    // }
};

pub const DroneConfig = struct {
    const Self = @This();

    pub const default_config_folder = "config";
    pub const default_config_path = "drone_config.json";

    pub const default_local_ip = "192.168.1.0";
    pub const default_local_port = 5000;

    pub const default_controller_ip = "192.168.1.1";
    pub const default_controller_port = 5000;

    pub const SensorCalibration = struct {
        mag_hard_iron: Vec3,
        mag_soft_iron: Vec3,
        accel_offset: Vec3,
        gyro_offset: Vec3,
    };

    local_ip: []const u8,
    local_port: u16,

    controller_ip: []const u8,
    controller_port: u16,

    dshot_protocol: DSHOT,
    global_max_throttle: f32,
    motors: [@typeInfo(Protocol.Motors).@"enum".fields.len]MotorConfig,
    battery: Battery,
    sensor_calibration: SensorCalibration,
    allocator: std.mem.Allocator,

    pub const MotorConfig = struct {
        pin: u8,
        direction: RotationDirection,
        max_throttle: f32,
    };

    pub fn init(allocator: std.mem.Allocator) !*DroneConfig {
        const self = try allocator.create(DroneConfig);
        const global_max_throttle = 75.0;

        self.* = .{
            .local_ip = try allocator.dupe(u8, default_local_ip),
            .local_port = default_local_port,
            .controller_ip = try allocator.dupe(u8, default_controller_ip),
            .controller_port = default_controller_port,
            .dshot_protocol = DSHOT.DSHOT_300,
            .global_max_throttle = global_max_throttle,
            .motors = [_]MotorConfig{
                .{
                    .pin = 6,
                    .direction = RotationDirection.Clockwise,
                    .max_throttle = global_max_throttle,
                },
                .{
                    .pin = 5,
                    .direction = RotationDirection.Counterclockwise,
                    .max_throttle = global_max_throttle,
                },
                .{
                    .pin = 22,
                    .direction = RotationDirection.Clockwise,
                    .max_throttle = global_max_throttle,
                },
                .{
                    .pin = 27,
                    .direction = RotationDirection.Counterclockwise,
                    .max_throttle = global_max_throttle,
                },
            },
            .battery = Battery.Lipo_4S,
            .sensor_calibration = .{
                .mag_hard_iron = .{ .x = 0, .y = 0, .z = 0 },
                .mag_soft_iron = .{ .x = 1, .y = 1, .z = 1 },
                .accel_offset = .{ .x = 0, .y = 0, .z = 0 },
                .gyro_offset = .{ .x = 0, .y = 0, .z = 0 },
            },
            .allocator = allocator,
        };

        try self.saveToFile(default_config_path);
        return self;
    }

    pub fn deinit(self: *DroneConfig) void {
        self.allocator.free(self.ip);
        self.allocator.destroy(self);
    }

    pub fn toJson(self: *Self, allocator: std.mem.Allocator) ![]u8 {
        return try std.json.stringifyAlloc(allocator, .{
            .local_ip = self.local_ip,
            .local_port = self.local_port,
            .controller_ip = self.controller_ip,
            .controller_port = self.controller_port,
            .dshot_protocol = @intFromEnum(self.dshot_protocol),
            .global_max_throttle = self.global_max_throttle,
            .motors = blk: {
                var motors: [@typeInfo(Protocol.Motors).@"enum".fields.len]struct {
                    pin: u8,
                    direction: u8,
                    max_throttle: f32,
                } = undefined;

                for (self.motors, 0..) |motor, i| {
                    motors[i] = .{
                        .pin = motor.pin,
                        .direction = @intFromEnum(motor.direction),
                        .max_throttle = motor.max_throttle,
                    };
                }
                break :blk motors;
            },
            .battery = .{
                .cells = @intFromEnum(self.battery),
                .type = "LiPo",
            },
            .sensor_calibration = .{
                .mag_hard_iron = .{
                    .x = self.sensor_calibration.mag_hard_iron.x,
                    .y = self.sensor_calibration.mag_hard_iron.y,
                    .z = self.sensor_calibration.mag_hard_iron.z,
                },
                .mag_soft_iron = .{
                    .x = self.sensor_calibration.mag_soft_iron.x,
                    .y = self.sensor_calibration.mag_soft_iron.y,
                    .z = self.sensor_calibration.mag_soft_iron.z,
                },
                .accel_offset = .{
                    .x = self.sensor_calibration.accel_offset.x,
                    .y = self.sensor_calibration.accel_offset.y,
                    .z = self.sensor_calibration.accel_offset.z,
                },
                .gyro_offset = .{
                    .x = self.sensor_calibration.gyro_offset.x,
                    .y = self.sensor_calibration.gyro_offset.y,
                    .z = self.sensor_calibration.gyro_offset.z,
                },
            },
        }, .{});
    }

    pub fn saveToFile(self: *Self, path: ?[]const u8) !void {
        const real_path = try std.fmt.allocPrint(self.allocator, "./{s}/{s}", .{
            default_config_folder,
            path orelse default_config_path,
        });
        defer self.allocator.free(real_path);

        const file = try std.fs.cwd().createFile(real_path, .{});
        defer file.close();

        try std.json.stringify(.{
            .local_ip = self.local_ip,
            .local_port = self.local_port,
            .controller_ip = self.controller_ip,
            .controller_port = self.controller_port,
            .dshot_protocol = @intFromEnum(self.dshot_protocol),
            .global_max_throttle = self.global_max_throttle,
            .motors = blk: {
                var motors: [@typeInfo(Protocol.Motors).@"enum".fields.len]struct {
                    pin: u8,
                    direction: u8,
                    max_throttle: f32,
                } = undefined;

                for (self.motors, 0..) |motor, i| {
                    motors[i] = .{
                        .pin = motor.pin,
                        .direction = @intFromEnum(motor.direction),
                        .max_throttle = motor.max_throttle,
                    };
                }
                break :blk motors;
            },
            .battery = .{
                .cells = @intFromEnum(self.battery),
                .type = "LiPo",
            },
            .sensor_calibration = .{
                .mag_hard_iron = .{
                    .x = self.sensor_calibration.mag_hard_iron.x,
                    .y = self.sensor_calibration.mag_hard_iron.y,
                    .z = self.sensor_calibration.mag_hard_iron.z,
                },
                .mag_soft_iron = .{
                    .x = self.sensor_calibration.mag_soft_iron.x,
                    .y = self.sensor_calibration.mag_soft_iron.y,
                    .z = self.sensor_calibration.mag_soft_iron.z,
                },
                .accel_offset = .{
                    .x = self.sensor_calibration.accel_offset.x,
                    .y = self.sensor_calibration.accel_offset.y,
                    .z = self.sensor_calibration.accel_offset.z,
                },
                .gyro_offset = .{
                    .x = self.sensor_calibration.gyro_offset.x,
                    .y = self.sensor_calibration.gyro_offset.y,
                    .z = self.sensor_calibration.gyro_offset.z,
                },
            },
        }, .{}, file.writer());
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, path: ?[]const u8) !*Self {
        const real_path = try std.fmt.allocPrint(allocator, "./{s}/{s}", .{
            default_config_folder,
            path orelse default_config_path,
        });
        defer allocator.free(real_path);

        const file = try std.fs.cwd().openFile(real_path, .{});
        defer file.close();

        const max_size = 4096;
        const contents = try file.readToEndAlloc(allocator, max_size);
        defer allocator.free(contents);

        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            contents,
            .{},
        );
        defer parsed.deinit();

        const config = try allocator.create(Self);

        // Update config from parsed JSON
        config.local_ip = parsed.value.object.get("local_ip").?.string;
        config.local_port = @intCast(parsed.value.object.get("local_port").?.integer);
        config.controller_ip = parsed.value.object.get("controller_ip").?.string;
        config.controller_port = @intCast(parsed.value.object.get("controller_port").?.integer);
        config.dshot_protocol = @enumFromInt(parsed.value.object.get("dshot_protocol").?.integer);
        config.global_max_throttle = @floatCast(parsed.value.object.get("global_max_throttle").?.float);
        config.allocator = allocator;

        config.battery = Battery.Lipo_4S;

        // Parse battery configuration if present
        if (parsed.value.object.get("battery")) |battery_config| {
            if (battery_config.object.get("cells")) |cells_value| {
                const cells = @as(u8, @intCast(cells_value.integer));
                // Validate cell count (2-6 cells supported)
                if (cells >= 2 and cells <= 6) {
                    config.battery = @enumFromInt(cells);
                }
            }
        }

        const motors_array = parsed.value.object.get("motors").?.array;
        for (motors_array.items, 0..) |motor, i| {
            config.motors[i] = .{
                .pin = @intCast(motor.object.get("pin").?.integer),
                .direction = @enumFromInt(motor.object.get("direction").?.integer),
                .max_throttle = @floatCast(motor.object.get("max_throttle").?.float),
            };
        }

        if (parsed.value.object.get("sensor_calibration")) |calibration| {
            const mag_hard_iron = calibration.object.get("mag_hard_iron").?.object;
            const mag_soft_iron = calibration.object.get("mag_soft_iron").?.object;
            const accel_offset = calibration.object.get("accel_offset").?.object;
            const gyro_offset = calibration.object.get("gyro_offset").?.object;

            config.sensor_calibration = .{
                .mag_hard_iron = .{
                    .x = @floatCast(mag_hard_iron.get("x").?.float),
                    .y = @floatCast(mag_hard_iron.get("y").?.float),
                    .z = @floatCast(mag_hard_iron.get("z").?.float),
                },
                .mag_soft_iron = .{
                    .x = @floatCast(mag_soft_iron.get("x").?.float),
                    .y = @floatCast(mag_soft_iron.get("y").?.float),
                    .z = @floatCast(mag_soft_iron.get("z").?.float),
                },
                .accel_offset = .{
                    .x = @floatCast(accel_offset.get("x").?.float),
                    .y = @floatCast(accel_offset.get("y").?.float),
                    .z = @floatCast(accel_offset.get("z").?.float),
                },
                .gyro_offset = .{
                    .x = @floatCast(gyro_offset.get("x").?.float),
                    .y = @floatCast(gyro_offset.get("y").?.float),
                    .z = @floatCast(gyro_offset.get("z").?.float),
                },
            };
        } else {
            // Set default calibration values if not found in config
            config.sensor_calibration = .{
                .mag_hard_iron = .{ .x = 0, .y = 0, .z = 0 },
                .mag_soft_iron = .{ .x = 1, .y = 1, .z = 1 },
                .accel_offset = .{ .x = 0, .y = 0, .z = 0 },
                .gyro_offset = .{ .x = 0, .y = 0, .z = 0 },
            };
        }

        return config;
    }
};
