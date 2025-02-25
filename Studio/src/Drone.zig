const std = @import("std");
const Math = @import("Math.zig");
const Vec3 = Math.Vec3;

pub const DroneConfig = struct {
    const Self = @This();

    pub const default_config_folder = "configs";
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

    dshot_protocol: MotorController.Protocols,
    global_max_throttle: f32,
    motors: [@typeInfo(MotorController.Motors).@"enum".fields.len]MotorConfig,

    sensor_calibration: SensorCalibration,
    allocator: std.mem.Allocator,

    pub const MotorConfig = struct {
        pin: u8,
        direction: MotorController.RotationDirection,
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
            .dshot_protocol = MotorController.Protocols.@"300",
            .global_max_throttle = global_max_throttle,
            .motors = [_]MotorConfig{
                .{
                    .pin = 6,
                    .direction = MotorController.RotationDirection.Clockwise,
                    .max_throttle = global_max_throttle,
                },
                .{
                    .pin = 5,
                    .direction = MotorController.RotationDirection.Counterclockwise,
                    .max_throttle = global_max_throttle,
                },
                .{
                    .pin = 22,
                    .direction = MotorController.RotationDirection.Clockwise,
                    .max_throttle = global_max_throttle,
                },
                .{
                    .pin = 27,
                    .direction = MotorController.RotationDirection.Counterclockwise,
                    .max_throttle = global_max_throttle,
                },
            },
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
                var motors: [@typeInfo(MotorController.Motors).@"enum".fields.len]struct {
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
            "configs",
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
                var motors: [@typeInfo(MotorController.Motors).@"enum".fields.len]struct {
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
            "configs",
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

const DSHOT_MIN_THROTTLE: u16 = 48;
const DSHOT_MAX_THROTTLE: u16 = 2047;

pub const ConnectionHandler = struct {
    const Self = @This();

    const ACK_TIMEOUT_MS = 500;
    const MAX_RETRIES = 3;
    const MAX_FAILURES = 5;
    const SYNC_INTERVAL_MS = 100;

    pub const ConnectionState = enum(u8) {
        Disconnected,
        Connecting,
        Connected,
        ConfigSync,
        Ready,
        Running,
        Failed,

        pub fn toString(self: ConnectionState) []const u8 {
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
    };

    pub const ConnectionError = error{
        SocketCreationFailed,
        BindFailed,
        SendFailed,
        ConfigSyncFailed,
        AckTimeout,
        InvalidState,
    };

    socket: ?std.posix.socket_t,
    local_addr: std.net.Address,
    server_addr: std.net.Address,
    state: std.atomic.Value(ConnectionState),
    motor_states: *[@typeInfo(MotorController.Motors).@"enum".fields.len]MotorController.MotorState,
    config: *DroneConfig,
    running: bool,
    retries: u8,
    failures: u8,
    allocator: std.mem.Allocator,
    sync_thread: ?std.Thread,
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator, config: *DroneConfig, motor_states: *[@typeInfo(MotorController.Motors).@"enum".fields.len]MotorController.MotorState) !*Self {
        const self = try allocator.create(Self);

        self.* = .{
            .socket = null,
            .local_addr = try std.net.Address.parseIp4(config.local_ip, config.local_port),
            .server_addr = try std.net.Address.parseIp4(config.controller_ip, config.controller_port),
            .motor_states = motor_states,
            .state = std.atomic.Value(ConnectionState).init(.Disconnected),
            .config = config,
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
                        std.debug.print("Connection initialization failed: {}\n", .{err});
                        self.handleFailure("Socket initialization failed");
                        continue;
                    };
                    self.state.store(.Connecting, .release);
                },
                .Connecting => {
                    self.sendConnectionRequest() catch |err| {
                        std.debug.print("Connection request failed: {}\n", .{err});

                        if (self.retries >= MAX_RETRIES) {
                            self.handleFailure("Connection failed after max retries");
                        }

                        std.time.sleep(1 * std.time.ns_per_s);
                        continue;
                    };
                    self.state.store(.Connected, .release);
                },
                .Connected => {
                    self.sendConfig() catch |err| {
                        std.debug.print("Config sync failed: {}\n", .{err});
                        self.handleFailure("Config sync failed");
                        continue;
                    };
                    self.state.store(.ConfigSync, .release);
                },
                .ConfigSync => {
                    self.waitForConfigAck() catch |err| {
                        std.debug.print("Config ack failed: {}\n", .{err});
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
                            std.debug.print("Heartbeat failed: {}\n", .{err});
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
                    self.motor_states[motor_idx].throttle = @floatFromInt(raw_throttle - DSHOT_MIN_THROTTLE);
                    self.motor_states[motor_idx].throttle = self.motor_states[motor_idx].throttle / @as(f32, DSHOT_MAX_THROTTLE - DSHOT_MIN_THROTTLE) * 100.0;
                }
            }
        }
    }

    fn handleHeartbeatResponse(self: *Self, response: []const u8) !void {
        std.debug.print("Heartbeat response: {s}\n", .{response});
        if (std.mem.startsWith(u8, response, "STATUS ")) {
            self.updateMotorStates(response);
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

pub const MotorController = struct {
    const Self = @This();

    pub const Protocols = enum {
        @"150",
        @"300",
        @"600",
        @"1200",
    };

    pub const Motors = enum {
        Motor_1,
        Motor_2,
        Motor_3,
        Motor_4,
    };

    pub const RotationDirection = enum {
        Clockwise,
        Counterclockwise,
    };

    pub const Commands = enum {
        SetSpeed,
        ReverseDirection,
        Arm,
        Disarm,
    };

    pub const Command = struct {
        kind: Commands,
        motor: Motors,
        speed: f32,

        /// Basic validation depending on command kind.
        pub fn isValid(self: Command) bool {
            switch (self.kind) {
                .SetSpeed => {
                    // Check for NaN and a "reasonable" speed range
                    return self.speed >= 0 and self.speed <= 100.0;
                },
                .ReverseDirection => return true,
                .Arm => return true,
                .Disarm => return true,
            }
        }
    };

    pub const CommandQueue = struct {

        // Ring buffer for commands
        buffer: [128]Command = undefined,
        read_idx: usize = 0,
        write_idx: usize = 0,
        mutex: std.Thread.Mutex = .{},

        pub fn init() @This() {
            var queue = CommandQueue{};
            // Initialize buffer with known valid values
            for (&queue.buffer) |*cmd| {
                cmd.* = Command{ .kind = .SetSpeed, .motor = Motors.Motor_1, .speed = 0.0 };
            }
            return queue;
        }

        pub fn push(self: *@This(), command: Command) bool {
            if (!command.isValid()) {
                std.debug.print("Invalid command detected: speed={d}\n", .{command.speed});
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

        pub fn pop(self: *@This()) ?Command {
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

            if (!command.isValid()) {
                std.debug.print("Invalid command in buffer: kind={any}, speed={d}. Resetting queue\n", .{ command.kind, command.speed });
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
                cmd.* = Command{ .kind = .SetSpeed, .motor = Motors.Motor_1, .speed = 0.0 };
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

    pub const MotorState = struct {
        throttle: f32 = 0.0,
        armed: bool = false,
        direction: RotationDirection,
    };

    udp_thread: ?std.Thread = null,
    config: *DroneConfig,
    motor_states: [@typeInfo(Motors).@"enum".fields.len]MotorState,
    command_queue: CommandQueue = CommandQueue.init(),
    prev_command: ?Command = null,
    connection_handler: *ConnectionHandler,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config_path: ?[]const u8) !*Self {
        const config = DroneConfig.loadFromFile(allocator, config_path orelse null) catch
            try DroneConfig.init(allocator);

        const motor_states = blk: {
            const num_motors = @typeInfo(Motors).@"enum".fields.len;
            var states: [num_motors]MotorState = undefined;

            for (0..num_motors) |i| {
                states[i] = MotorState{ .direction = @enumFromInt(i % 2) };
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
    }

    pub fn disconnect(self: *Self) void {
        if (self.udp_thread) |_| {
            self.connection_handler.stop();
            self.udp_thread.?.join();
            self.udp_thread = null;
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

    fn processCommand(self: *Self, command: Command) bool {
        // if (self.prev_command) |prev| {
        //     if (prev.kind == command.kind and prev.speed == command.speed) {
        //         return true; // Skip duplicate commands
        //     }
        // }

        std.debug.print("Processing Command => {s}\n", .{@tagName(command.kind)});

        const cmd_str = switch (command.kind) {
            .SetSpeed => blk: {
                break :blk std.fmt.allocPrint(
                    self.allocator,
                    "SetSpeed {d} {d:.1}\n",
                    .{
                        @intFromEnum(command.motor),
                        command.speed,
                    },
                ) catch |err| {
                    std.debug.print("Failed to format SetSpeed: {}\n", .{err});
                    return false;
                };
            },
            .ReverseDirection => blk: {
                break :blk std.fmt.allocPrint(
                    self.allocator,
                    "ReverseDirection {d}\n",
                    .{@intFromEnum(command.motor)},
                ) catch |err| {
                    std.debug.print("Failed to format ReverseDirection: {}\n", .{err});
                    return false;
                };
            },
            .Arm => blk: {
                break :blk std.fmt.allocPrint(
                    self.allocator,
                    "Arm {d}\n",
                    .{@intFromEnum(command.motor)},
                ) catch |err| {
                    std.debug.print("Failed to format Arm command: {}\n", .{err});
                    return false;
                };
            },
            .Disarm => blk: {
                break :blk std.fmt.allocPrint(
                    self.allocator,
                    "Disarm {d}\n",
                    .{@intFromEnum(command.motor)},
                ) catch |err| {
                    std.debug.print("Failed to format Disarm command: {}\n", .{err});
                    return false;
                };
            },
        };
        defer self.allocator.free(cmd_str);

        _ = std.posix.sendto(
            self.connection_handler.socket.?,
            cmd_str,
            0,
            &self.connection_handler.server_addr.any,
            self.connection_handler.server_addr.getOsSockLen(),
        ) catch |err| {
            std.debug.print("Failed to send command: {}\n", .{err});
            return false;
        };

        self.prev_command = command;
        return true;
    }

    pub fn getConnectionState(self: *Self) ConnectionHandler.ConnectionState {
        return self.connection_handler.state.load(.acquire);
    }

    // pub fn handleInput(self: *@This(), up_pressed: bool) void {
    //     std.debug.print("Handling input => Throttle on : {any}\n", .{up_pressed});
    //     const speed: f32 = if (up_pressed) 25.0 else 0.0;
    //     const command = CommandQueue.Command{ .kind = .SetSpeed, .motor = Motors.Motor_1, .speed = speed };
    //     _ = self.command_queue.push(command);
    // }
};
