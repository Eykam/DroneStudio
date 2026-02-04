const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const PhysicsThread = @import("PhysicsThread.zig");
const bullet = @import("../../bindings/c.zig").bullet;

const Vec3 = Math.Vec3;
const Quaternion = Math.Quaternion;
const PhysicsState = PhysicsThread.PhysicsState;

/// IMU sample data structure following the coordinate conventions
pub const IMUSample = struct {
    timestamp_us: u64,
    gyro: [3]f32, // rad/s  (sensor frame)
    accel: [3]f32, // m/s²   (sensor frame, gravity removed)
};

/// Lock-free SPSC queue for IMU samples
pub fn IMUSampleQueue(comptime size: usize) type {
    return struct {
        const Self = @This();

        data: [size]IMUSample = undefined,
        write_pos: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        read_pos: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

        pub fn push(self: *Self, sample: IMUSample) bool {
            const current_write = self.write_pos.load(.acquire);
            const next_write = (current_write + 1) % size;

            // Check if buffer is full
            if (next_write == self.read_pos.load(.acquire)) {
                return false; // Buffer full
            }

            self.data[current_write] = sample;
            self.write_pos.store(next_write, .release);
            return true;
        }

        pub fn pop(self: *Self) ?IMUSample {
            const current_read = self.read_pos.load(.acquire);

            // Check if buffer is empty
            if (current_read == self.write_pos.load(.acquire)) {
                return null; // Buffer empty
            }

            const sample = self.data[current_read];
            self.read_pos.store((current_read + 1) % size, .release);
            return sample;
        }

        pub fn isEmpty(self: *Self) bool {
            return self.read_pos.load(.acquire) == self.write_pos.load(.acquire);
        }

        pub fn isFull(self: *Self) bool {
            const current_write = self.write_pos.load(.acquire);
            const next_write = (current_write + 1) % size;
            return next_write == self.read_pos.load(.acquire);
        }
    };
}

/// IMU sensor component with realistic noise modeling
pub const IMUSensorComponent = struct {
    const Self = @This();

    // Static configuration (changeable by prefab)
    sample_rate_hz: u32 = 1000, // 1 kHz sampling rate
    noise_gyro_std: f32 = 0.0015, // rad/s/√Hz  (≈0.08 °/s rms @1 kHz)
    noise_accel_std: f32 = 0.02, // m/s²/√Hz    (≈0.6 mG rms @1 kHz)
    bias_walk_gyro: f32 = 2e-5, // rad/s/√s
    bias_walk_accel: f32 = 5e-4, // m/s²/√s
    pos_body: Vec3 = Vec3.init(0, 0, 0), // mount offset (B frame)
    rot_body: Quaternion = Quaternion.identity(), // mount orientation (B→sensor)

    // Runtime state
    entity_id: Core.EntityID = undefined,
    bias_gyro: Vec3 = Vec3.init(0, 0, 0),
    bias_accel: Vec3 = Vec3.init(0, 0, 0),
    last_update_us: u64 = 0,

    // Output queue for flight controller
    sample_queue: IMUSampleQueue(1024), // 1024 samples buffer (~1 second at 1kHz)

    pub fn init() Self {
        return Self{
            .sample_queue = IMUSampleQueue(1024){},
        };
    }

    pub fn attach(self: *Self, ecs: anytype, eid: Core.EntityID) !void {
        self.entity_id = eid;
        try ecs.imu_sensor_components.add(eid, self.*);
    }

    /// Generate a sensor sample from physics state with realistic noise
    pub fn generateSample(
        self: *Self,
        physics_state: PhysicsState,
        timestamp_us: u64,
        rng: *std.Random.DefaultPrng,
    ) IMUSample {
        //TODO: Keep this here or move to after update?
        const dt = if (self.last_update_us > 0)
            @as(f32, @floatFromInt(timestamp_us - self.last_update_us)) / 1_000_000.0
        else
            1.0 / @as(f32, @floatFromInt(self.sample_rate_hz));

        self.last_update_us = timestamp_us;

        const rotation = physics_state.rotation;
        const rotation_bw = rotation.conjugate(); // World to body

        const angular_vel = physics_state.angular_velocity;
        const angular_vel_world = Vec3.from_array(angular_vel);
        const angular_vel_body = angular_vel_world.rotate_by_quaternion(rotation_bw);

        // Transform linear acceleration to body frame
        // Note: alpha_world from physics is total acceleration excludign gravity
        const accel_world = physics_state.accel_world;
        if (accel_world.length() > 5.0) {
            return IMUSample{
                .timestamp_us = timestamp_us,
                .gyro = .{ 0, 0, 0 },
                .accel = .{ 0, 0, 0 },
            };
        }

        const accel_body = accel_world.rotate_by_quaternion(rotation_bw);

        // Transform to sensor frame
        const accel_sensor = accel_body.rotate_by_quaternion(self.rot_body);
        const angular_vel_sensor = angular_vel_body.rotate_by_quaternion(self.rot_body);

        // Evolve bias random walk
        const sqrt_dt = @sqrt(dt);
        const random = rng.random();

        // Update gyro bias
        self.bias_gyro = Vec3.init(
            self.bias_gyro.x() + random.floatNorm(f32) * self.bias_walk_gyro * sqrt_dt,
            self.bias_gyro.y() + random.floatNorm(f32) * self.bias_walk_gyro * sqrt_dt,
            self.bias_gyro.z() + random.floatNorm(f32) * self.bias_walk_gyro * sqrt_dt,
        );

        // Update accel bias
        self.bias_accel = Vec3.init(
            self.bias_accel.x() + random.floatNorm(f32) * self.bias_walk_accel * sqrt_dt,
            self.bias_accel.y() + random.floatNorm(f32) * self.bias_walk_accel * sqrt_dt,
            self.bias_accel.z() + random.floatNorm(f32) * self.bias_walk_accel * sqrt_dt,
        );

        // Add white noise and bias to measurements
        const noise_scale_gyro = self.noise_gyro_std * sqrt_dt;
        const noise_scale_accel = self.noise_accel_std * sqrt_dt;

        const gyro_noisy = [3]f32{
            angular_vel_sensor.x() + self.bias_gyro.x() + random.floatNorm(f32) * noise_scale_gyro,
            angular_vel_sensor.y() + self.bias_gyro.y() + random.floatNorm(f32) * noise_scale_gyro,
            angular_vel_sensor.z() + self.bias_gyro.z() + random.floatNorm(f32) * noise_scale_gyro,
        };

        const accel_noisy = [3]f32{
            accel_sensor.x() + self.bias_accel.x() + random.floatNorm(f32) * noise_scale_accel,
            accel_sensor.y() + self.bias_accel.y() + random.floatNorm(f32) * noise_scale_accel,
            accel_sensor.z() + self.bias_accel.z() + random.floatNorm(f32) * noise_scale_accel,
        };

        return IMUSample{
            .timestamp_us = timestamp_us,
            .gyro = gyro_noisy,
            .accel = accel_noisy,
        };
    }

    /// Send sample to flight controller (non-blocking)
    pub fn publishSample(self: *Self, sample: IMUSample) bool {
        return self.sample_queue.push(sample);
    }

    /// Get latest sample (used by flight controller)
    pub fn getLatestSample(self: *Self) ?IMUSample {
        return self.sample_queue.pop();
    }

    /// Drain all samples and keep only the most recent (anti-FIFO lag)
    pub fn drainToLatest(self: *Self) ?IMUSample {
        var latest: ?IMUSample = null;
        while (self.sample_queue.pop()) |sample| {
            latest = sample;
        }
        return latest;
    }

    /// Reset IMU sensor state to initial values
    pub fn reset(self: *Self) void {
        self.bias_gyro = Vec3.init(0, 0, 0);
        self.bias_accel = Vec3.init(0, 0, 0);

        self.last_update_us = 0;
        while (self.sample_queue.pop()) |_| {}
    }
};

/// IMU system that manages all IMU sensors and runs the IMU thread
pub const IMUSystem = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    imu_components: *SparseSet(IMUSensorComponent),

    // IMU thread management
    imu_thread: ?std.Thread = null,
    should_shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    physics_thread: ?*PhysicsThread.ThreadedPhysicsSystem = null,

    // Timing
    target_rate_hz: f64 = 1000.0, // 1 kHz

    // Random number generator for noise
    rng: std.Random.DefaultPrng,

    pub fn init(allocator: std.mem.Allocator, imu_components: *SparseSet(IMUSensorComponent)) Self {
        return Self{
            .allocator = allocator,
            .imu_components = imu_components,
            .rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp())),
        };
    }

    pub fn deinit(self: *Self) void {
        self.stopIMUThread();
    }

    /// Set reference to physics thread for getting physics states
    pub fn setPhysicsThread(self: *Self, physics_thread: *PhysicsThread.ThreadedPhysicsSystem) void {
        self.physics_thread = physics_thread;
    }

    /// Start the high-priority IMU thread
    pub fn startIMUThread(self: *Self) !void {
        if (self.imu_thread != null) return; // Already running

        self.should_shutdown.store(false, .release);
        self.imu_thread = try std.Thread.spawn(.{}, imuThreadMain, .{self});

        std.debug.print("IMU thread started at {d} Hz\n", .{self.target_rate_hz});
    }

    /// Stop the IMU thread
    pub fn stopIMUThread(self: *Self) void {
        if (self.imu_thread) |thread| {
            self.should_shutdown.store(true, .release);
            thread.join();
            self.imu_thread = null;
            std.debug.print("IMU thread stopped\n", .{});
        }
    }

    /// IMU thread main loop - runs at 1kHz
    fn imuThreadMain(self: *Self) void {
        std.debug.print("IMU thread started\n", .{});

        var timer = std.time.Timer.start() catch {
            std.debug.print("Failed to start IMU timer\n", .{});
            return;
        };

        const target_frame_time_ns = @as(u64, @intFromFloat(1_000_000_000.0 / self.target_rate_hz));
        var frame_count: u64 = 0;

        while (!self.should_shutdown.load(.acquire)) {
            const frame_start = timer.read();
            const timestamp_us = @as(u64, @intCast(std.time.microTimestamp()));

            const physics = self.physics_thread orelse continue;
            if (physics.isPhysicsPaused()) continue;
            // Process all IMU sensors
            self.updateAllSensors(timestamp_us);

            frame_count += 1;

            // Debug output every second
            if (frame_count % 1000 == 0) {
                std.debug.print("IMU thread: {d:.3} kHz, processed {d} sensors\n", .{
                    @as(f64, @floatFromInt(frame_count)) / (@as(f64, @floatFromInt(timer.read())) / 1_000_000_000.0),
                    self.imu_components.dense.items.len,
                });
            }

            // Sleep to maintain target rate
            const frame_time = timer.read() - frame_start;
            if (frame_time < target_frame_time_ns) {
                const sleep_time = target_frame_time_ns - frame_time;
                std.time.sleep(sleep_time);
            }
        }

        std.debug.print("IMU thread shutting down after {} frames\n", .{frame_count});
    }

    /// Update all IMU sensors with latest physics data
    fn updateAllSensors(self: *Self, timestamp_us: u64) void {
        // Get latest physics states from physics thread
        //
        const physics_system = self.physics_thread orelse return;
        const physics_states = physics_system.getPhysicsStates();

        // Process each IMU sensor
        var imu_iter = self.imu_components.iterator();
        while (imu_iter.next()) |entry| {
            const imu_component = entry.component;
            const entity_id = entry.entity_id;

            // Find corresponding physics states
            for (physics_states) |physics_state| {
                if (physics_state.entity_id.id == entity_id.id) {
                    const sample = imu_component.generateSample(
                        physics_state,
                        timestamp_us,
                        &self.rng,
                    );

                    // Publish sample (non-blocking)
                    if (!imu_component.publishSample(sample)) {
                        // Queue full - this indicates flight controller is not consuming fast enough
                        std.debug.print("IMU sample queue full for entity {}\n", .{entity_id.id});
                    }

                    break;
                }
            }
        }
    }
};
