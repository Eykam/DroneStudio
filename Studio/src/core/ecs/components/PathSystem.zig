const std = @import("std");
const Math = @import("../../Math.zig");
const bt = @import("../../bindings/c.zig").bullet;
const gl = @import("../../bindings/gl.zig");
const Mesh = @import("../../Mesh.zig");
const Core = @import("../Core.zig");
const Transform = @import("./Transform.zig");
const Renderer = @import("./Renderer.zig");
const PathPrefab = @import("../prefabs/Path.zig");
const ECSManager = @import("../ECSManager.zig");

const Vec3 = Math.Vec3;
const Quaternion = Math.Quaternion;
const Quat = Math.Quat;
const glad = gl.glad;

const Self = @This();

allocator: std.mem.Allocator,
rng: std.Random.DefaultPrng,
paths: std.ArrayList(PathResult),
ecs: *ECSManager,
path_counter: usize = 0,
worker_collision_worlds: ?[]bt.CbtWorldHandle = null,
worker_fleet: *WorkerFleet,

// Async generation state
generation_thread: ?std.Thread = null,
is_generating: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
progress_current: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
progress_total: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
generation_failed: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
pending_result: ?GenerateMultipleResult = null,

pub fn init(allocator: std.mem.Allocator, ecs: *ECSManager) !*Self {
    const system = try allocator.create(Self);

    const num_threads = try std.Thread.getCpuCount() - 1;
    const fleet = try WorkerFleet.init(allocator, num_threads, pathGenWorker);

    system.* = .{
        .allocator = allocator,
        .rng = std.Random.DefaultPrng.init(0),
        .paths = std.ArrayList(PathResult).init(allocator),
        .ecs = ecs,
        .worker_fleet = fleet,
    };
    return system;
}

pub fn deinit(self: *Self) void {
    for (self.paths.items) |*path| {
        path.deinit();
    }
    self.paths.deinit();

    // Free worker worlds array (worlds themselves are intentionally leaked - see generateMultiplePaths)
    if (self.worker_collision_worlds) |worlds| {
        self.allocator.free(worlds);
    }

    // Deinit worker fleet
    self.worker_fleet.deinit();

    self.allocator.destroy(self);
}

pub fn getPaths(self: *Self) []PathResult {
    return self.paths.items;
}

pub fn getPath(self: *Self, index: usize) ?*PathResult {
    if (index >= self.paths.items.len) return null;
    return &self.paths.items[index];
}

pub fn clearPaths(self: *Self) void {
    for (self.paths.items) |*path| {
        path.deinit();
    }
    self.paths.clearRetainingCapacity();
}

pub fn getSceneBounds(self: *Self, shrink_factor: f32) !AABB3 {
    const physics_thread = self.ecs.collision_system.physics_thread orelse return error.NoPhysicsWorld;
    const world = physics_thread.bullet_world;

    const num_bodies = bt.cbtWorldGetNumBodies(world);
    if (num_bodies == 0) {
        return AABB3{
            .min = Vec3.init(-10, -10, 0),
            .max = Vec3.init(10, 10, 10),
        };
    }

    var scene_min = Vec3.init(std.math.floatMax(f32), std.math.floatMax(f32), std.math.floatMax(f32));
    var scene_max = Vec3.init(-std.math.floatMax(f32), -std.math.floatMax(f32), -std.math.floatMax(f32));

    var i: c_int = 0;
    while (i < num_bodies) : (i += 1) {
        const body = bt.cbtWorldGetBody(world, i);
        if (body == null) continue;

        var aabb_min: [3]f32 = undefined;
        var aabb_max: [3]f32 = undefined;
        bt.cbtBodyGetAabb(body, &aabb_min, &aabb_max);

        scene_min = Vec3.init(
            @min(scene_min.x(), aabb_min[0]),
            @min(scene_min.y(), aabb_min[1]),
            @min(scene_min.z(), aabb_min[2]),
        );
        scene_max = Vec3.init(
            @max(scene_max.x(), aabb_max[0]),
            @max(scene_max.y(), aabb_max[1]),
            @max(scene_max.z(), aabb_max[2]),
        );
    }

    // Apply shrink factor
    const center = scene_min.add(scene_max).scale(0.5);
    const half_size = scene_max.sub(scene_min).scale(0.5 * shrink_factor);

    return AABB3{
        .min = center.sub(half_size),
        .max = center.add(half_size),
    };
}

// NOTE: @fieldParentPtr approach was causing segfaults due to memory corruption
// TODO: Investigate why @fieldParentPtr didn't work correctly here
// For now, we pass ECS pointer directly in init()
// fn getECS(self: *Self) *ECSManager {
//     const fields = @typeInfo(ECSManager).@"struct".fields;
//     inline for (fields) |field| {
//         if (field.type == ?*Self) {
//             const optional_ptr: *?*Self = @ptrCast(@alignCast(self));
//             return @fieldParentPtr(field.name, optional_ptr);
//         }
//     }
//     @compileError("PathSystem not found in ECSManager");
// }

pub const Waypoint = struct {
    p: Vec3,
    yaw: f32,
};

pub const CreatePathParams = struct {
    bounds: AABB3,
    bounds_shrink_factor: f32 = 0.75, // 1.0 = no shrink, 0.8 = 80% of original size
    // length & spacing
    L_min: f32,
    L_max: f32,
    s_min: f32,
    s_max: f32,
    max_pts: u32,
    // vertical & turning
    z_lo: f32,
    z_hi: f32,
    dz_max: f32,
    R_min: f32,
    max_turn_deg: f32,
    // yaw
    yaw_bias_w: f32,
    yaw_noise_deg: f32,
    // collision
    drone_radius: f32,
    sweep_margin: f32,
    // curve & tessellation
    tension_base: f32,
    flatness_eps: f32,
    // dynamics constraints
    v_max: f32, // maximum velocity
    a_max: f32, // maximum acceleration
    j_max: f32, // maximum jerk
    // rng & retries
    seed: u64,
    max_local_retries: u32,
    backtrack_points: u32,
};

pub const PathResult = struct {
    waypoints: []Waypoint,
    beziers: []CubicBezier3,
    samples: []Vec3,
    tangents: []Vec3,
    curvature: []f32,
    s_cumsum: []f32,
    // Time parameterization
    t_cumsum: []f32, // cumulative time at each sample
    velocities: []f32, // velocity magnitude at each sample
    // Quaternion orientation at each sample
    orientations: []Quaternion,
    // Visualization entities
    path_entities: ?PathPrefab.PathEntities,
    visible: bool = true,
    allocator: std.mem.Allocator,

    pub fn setVisible(self: *PathResult, ecs: *ECSManager, visible: bool) void {
        self.visible = visible;

        if (self.path_entities) |*entities| {
            entities.setVisible(ecs, visible);
        }
    }

    pub fn deinit(self: *PathResult) void {
        self.allocator.free(self.waypoints);
        self.allocator.free(self.beziers);
        self.allocator.free(self.samples);
        self.allocator.free(self.tangents);
        self.allocator.free(self.curvature);
        self.allocator.free(self.s_cumsum);
        self.allocator.free(self.t_cumsum);
        self.allocator.free(self.velocities);
        self.allocator.free(self.orientations);

        // PathEntities no longer has allocated memory, so no deinit needed
        if (self.path_entities) |*entities| {
            entities.deinit();
        }
    }

    pub fn evalPos(self: *const PathResult, t_norm: f32) Vec3 {
        const t = Math.clamp(t_norm, 0.0, 1.0);
        const total_len = self.length();
        const target_s = t * total_len;

        var lo: usize = 0;
        var hi: usize = self.s_cumsum.len - 1;

        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (self.s_cumsum[mid] < target_s) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if (lo >= self.samples.len) lo = self.samples.len - 1;
        if (lo == 0) return self.samples[0];

        const s0 = self.s_cumsum[lo - 1];
        const s1 = self.s_cumsum[lo];
        const local_t = if (s1 - s0 > 1e-6) (target_s - s0) / (s1 - s0) else 0.0;

        return Vec3.lerp(self.samples[lo - 1], self.samples[lo], local_t);
    }

    pub fn evalYaw(self: *const PathResult, t_norm: f32) f32 {
        const quat = self.evalOrientation(t_norm);
        const euler = quat.to_euler();
        return euler[1]; // yaw is the second component
    }

    pub fn evalOrientation(self: *const PathResult, t_norm: f32) Quaternion {
        const t = Math.clamp(t_norm, 0.0, 1.0);
        const total_len = self.length();
        const target_s = t * total_len;

        var lo: usize = 0;
        var hi: usize = self.s_cumsum.len - 1;

        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (self.s_cumsum[mid] < target_s) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if (lo >= self.orientations.len) lo = self.orientations.len - 1;
        if (lo == 0) return self.orientations[0];

        const s0 = self.s_cumsum[lo - 1];
        const s1 = self.s_cumsum[lo];
        const local_t = if (s1 - s0 > 1e-6) (target_s - s0) / (s1 - s0) else 0.0;

        return Quaternion.slerp(self.orientations[lo - 1], self.orientations[lo], local_t);
    }

    pub fn length(self: *const PathResult) f32 {
        if (self.s_cumsum.len == 0) return 0.0;
        return self.s_cumsum[self.s_cumsum.len - 1];
    }

    pub fn duration(self: *const PathResult) f32 {
        if (self.t_cumsum.len == 0) return 0.0;
        return self.t_cumsum[self.t_cumsum.len - 1];
    }
};

pub const WorkerFleet = struct {
    const Fleet = @This();

    pub const WorkerFn = *const fn (*anyopaque, usize) void;

    allocator: std.mem.Allocator,
    threads: []std.Thread,
    num_workers: usize,

    // round control
    mu: std.Thread.Mutex = .{},
    cv: std.Thread.Condition = .{},
    shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    epoch: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    // set at startRound, read by workers
    run_fn: WorkerFn,
    round_ctx: *anyopaque = undefined,
    wg: *std.Thread.WaitGroup = undefined,

    pub fn init(allocator: std.mem.Allocator, num_workers: usize, run_fn: WorkerFn) !*Fleet {
        std.debug.assert(num_workers > 0);

        var self = try allocator.create(Fleet);
        self.* = .{
            .allocator = allocator,
            .threads = try allocator.alloc(std.Thread, num_workers),
            .num_workers = num_workers,
            .run_fn = run_fn,
        };

        // spawn exactly num_workers OS threads once
        for (0..num_workers) |wid| {
            self.threads[wid] = try std.Thread.spawn(.{}, workerLoop, .{ self, wid });
        }
        return self;
    }

    pub fn deinit(self: *Fleet) void {
        _ = self.shutdown.store(true, .release);
        self.cv.broadcast(); // wake all
        for (self.threads) |*t| t.join();
        self.allocator.free(self.threads);
        self.allocator.destroy(self);
    }

    /// Start a round: all workers will run run_fn(ctx, wid) once and then park again.
    pub fn startRound(self: *Fleet, ctx: *anyopaque, wg: *std.Thread.WaitGroup) void {
        // prepare wait group for num_workers finishes
        wg.reset();
        for (0..self.num_workers) |_| wg.start();

        self.mu.lock();
        self.round_ctx = ctx;
        self.wg = wg;
        _ = self.epoch.fetchAdd(1, .release); // signal a new round
        self.cv.broadcast();
        self.mu.unlock();
    }

    fn workerLoop(self: *Fleet, wid: usize) void {
        var last_seen_epoch: u64 = 0;

        while (!self.shutdown.load(.acquire)) {
            // Wait for a new round
            self.mu.lock();
            while (!self.shutdown.load(.acquire)) {
                const e = self.epoch.load(.acquire);
                if (e != last_seen_epoch) {
                    last_seen_epoch = e;
                    break;
                }
                self.cv.wait(&self.mu);
            }
            const run_fn = self.run_fn;
            const ctx = self.round_ctx;
            const wg = self.wg;
            self.mu.unlock();

            if (self.shutdown.load(.acquire)) break;

            // Run the user function once for this round
            run_fn(ctx, wid);

            // Signal this worker finished the round
            wg.finish();
        }
    }
};

pub const GenerateMultipleResult = struct {
    successful: usize,
    total_attempts: usize,
    failed: bool,
    results: []PathResult,
};

const PathGenSharedState = struct {
    results: []PathResult,
    next_slot: std.atomic.Value(usize),
    consecutive_failures: std.atomic.Value(usize),
    total_attempts: std.atomic.Value(usize),
    params: CreatePathParams,
    use_random_start: bool,
    use_random_seed: bool,
    seed_base: u64,
    max_consecutive: usize,
    path_system: *Self,
    worker_allocator: std.mem.Allocator,
    worker_worlds: []bt.CbtWorldHandle,
};

fn pathGenWorker(ctx: *anyopaque, worker_id: usize) void {
    const state: *PathGenSharedState = @ptrCast(@alignCast(ctx));

    const world = state.worker_worlds[worker_id];

    while (true) {
        // Check if we're done or hit max consecutive failures
        const current_slot = state.next_slot.load(.acquire);
        const consec_fails = state.consecutive_failures.load(.acquire);

        if (current_slot >= state.results.len or consec_fails >= state.max_consecutive) {
            return;
        }

        const attempt = state.total_attempts.fetchAdd(1, .monotonic);

        // Compute seed for this attempt
        var params = state.params;
        if (state.use_random_seed) {
            const timestamp = @as(u64, @intCast(std.time.milliTimestamp()));
            params.seed = timestamp +% attempt;
        } else {
            params.seed = state.seed_base +% attempt;
        }

        // Try to generate a path
        const empty_anchors: []const Waypoint = &[_]Waypoint{};
        const anchors = if (state.use_random_start) null else empty_anchors;
        var result = state.path_system.createPath(
            params,
            anchors,
            world,
        ) catch {
            _ = state.consecutive_failures.fetchAdd(1, .monotonic);
            continue;
        };

        // Claim a slot
        const slot = state.next_slot.fetchAdd(1, .monotonic);
        if (slot < state.results.len) {
            state.results[slot] = result;
            _ = state.consecutive_failures.store(0, .release); // Reset consecutive failures

            // Update progress
            _ = state.path_system.progress_current.store(slot + 1, .release);

            std.debug.print("Successfully generated path {d}/{d}\n", .{ slot + 1, state.results.len });
        } else {
            // Buffer full, clean up this result
            result.deinit();
            return;
        }
    }
}

pub fn generateMultiplePaths(
    self: *Self,
    allocator: std.mem.Allocator,
    num_paths: usize,
    base_params: CreatePathParams,
    use_random_start: bool,
    use_random_seed: bool,
    seed_base: u64,
) !GenerateMultipleResult {
    const max_consecutive_failures: usize = 1000;

    // Allocate results buffer
    const results = try allocator.alloc(PathResult, num_paths);

    // Lazy initialize worker collision worlds (once per PathSystem lifetime)
    if (self.worker_collision_worlds == null) {
        const physics_thread = self.ecs.collision_system.physics_thread orelse return error.NoPhysicsWorld;
        const source_world = physics_thread.bullet_world;

        const num_workers = try std.Thread.getCpuCount() - 1;
        const worker_worlds = try allocator.alloc(bt.CbtWorldHandle, num_workers);

        for (worker_worlds) |*world| {
            world.* = try cloneCollisionWorld(source_world, allocator);
        }

        self.worker_collision_worlds = worker_worlds;
    }

    const worker_worlds = self.worker_collision_worlds.?;

    var wait_group: std.Thread.WaitGroup = undefined;

    var shared = PathGenSharedState{
        .results = results,
        .next_slot = std.atomic.Value(usize).init(0),
        .consecutive_failures = std.atomic.Value(usize).init(0),
        .total_attempts = std.atomic.Value(usize).init(0),
        .params = base_params,
        .use_random_start = use_random_start,
        .use_random_seed = use_random_seed,
        .seed_base = seed_base,
        .max_consecutive = max_consecutive_failures,
        .path_system = self,
        .worker_allocator = allocator,
        .worker_worlds = worker_worlds,
    };

    // Start a round with the worker fleet
    self.worker_fleet.startRound(@ptrCast(&shared), &wait_group);
    wait_group.wait();

    const successful = shared.next_slot.load(.acquire);
    const total = shared.total_attempts.load(.acquire);
    const failed = successful < num_paths;

    if (failed) {
        // Clean up partial results
        for (results[0..successful]) |*result| {
            result.deinit();
        }
        allocator.free(results);
        return error.PathGenerationFailed;
    }

    // Return results without visualization (will be done on main thread)
    return GenerateMultipleResult{
        .successful = successful,
        .total_attempts = total,
        .failed = failed,
        .results = results,
    };
}

const AsyncGenParams = struct {
    path_system: *Self,
    allocator: std.mem.Allocator,
    num_paths: usize,
    base_params: CreatePathParams,
    use_random_start: bool,
    use_random_seed: bool,
    seed_base: u64,
};

fn asyncGenerationThread(params: AsyncGenParams) void {
    const self = params.path_system;

    // Reset progress
    _ = self.progress_current.store(0, .release);
    _ = self.progress_total.store(params.num_paths, .release);
    _ = self.generation_failed.store(false, .release);

    // Run generation
    const result = self.generateMultiplePaths(
        params.allocator,
        params.num_paths,
        params.base_params,
        params.use_random_start,
        params.use_random_seed,
        params.seed_base,
    ) catch {
        _ = self.generation_failed.store(true, .release);
        _ = self.is_generating.store(false, .release);
        return;
    };

    // Store result (will be picked up by main thread)
    self.pending_result = result;

    // Mark complete
    _ = self.is_generating.store(false, .release);
}

pub fn startAsyncGeneration(
    self: *Self,
    allocator: std.mem.Allocator,
    num_paths: usize,
    base_params: CreatePathParams,
    use_random_start: bool,
    use_random_seed: bool,
    seed_base: u64,
) !void {
    // Check if already generating
    if (self.is_generating.load(.acquire)) {
        return error.AlreadyGenerating;
    }

    // Join previous thread if exists
    if (self.generation_thread) |thread| {
        thread.join();
        self.generation_thread = null;
    }

    // Mark as generating
    _ = self.is_generating.store(true, .release);

    // Spawn thread
    const params = AsyncGenParams{
        .path_system = self,
        .allocator = allocator,
        .num_paths = num_paths,
        .base_params = base_params,
        .use_random_start = use_random_start,
        .use_random_seed = use_random_seed,
        .seed_base = seed_base,
    };

    self.generation_thread = try std.Thread.spawn(.{}, asyncGenerationThread, .{params});
}

pub fn getGenerationProgress(self: *Self) struct { current: usize, total: usize, is_generating: bool, failed: bool } {
    return .{
        .current = self.progress_current.load(.acquire),
        .total = self.progress_total.load(.acquire),
        .is_generating = self.is_generating.load(.acquire),
        .failed = self.generation_failed.load(.acquire),
    };
}

pub fn finalizePendingPaths(self: *Self) !void {
    if (self.pending_result) |result| {
        defer self.pending_result = null;

        // Visualize all paths on main thread using Path prefab
        for (result.results) |*path_result| {
            const path_entities = try PathPrefab.spawn(
                self.allocator,
                self.ecs,
                path_result.waypoints,
                path_result.samples,
                path_result.velocities,
                self.path_counter,
                .{ 1.0, 1.0, 0.0 },
            );
            std.debug.print("{s}\n", .{"=" ** 80});
            std.debug.print("{s}\n", .{"=" ** 80});
            for (path_result.waypoints, 0..) |waypoint, idx| {
                std.debug.print("Waypoint: {} => {d:.2}\n", .{ idx, Math.degrees(waypoint.yaw) });
            }

            path_result.path_entities = path_entities;

            // Hide all previous paths
            for (self.paths.items) |*existing_path| {
                existing_path.setVisible(self.ecs, false);
            }

            try self.paths.append(path_result.*);
            self.path_counter += 1;
        }

        self.allocator.free(result.results);
    }
}

pub fn createPath(self: *Self, p: CreatePathParams, anchors: ?[]const Waypoint, collision_world: ?bt.CbtWorldHandle) !PathResult {
    self.rng = std.Random.DefaultPrng.init(p.seed);

    // Generate waypoints
    var waypoints = std.ArrayList(Waypoint).init(self.allocator);
    defer waypoints.deinit();

    if (anchors) |anchor_slice| {
        try waypoints.appendSlice(anchor_slice);
    } else {
        // Generate random starting waypoint
        const start = try self.generateRandomWaypoint(p, collision_world);
        try waypoints.append(start);
    }

    var total_length: f32 = 0.0;
    var attempts: u32 = 0;
    const max_attempts = p.max_pts * p.max_local_retries;
    var bounds_fails: u32 = 0;
    var radius_fails: u32 = 0;
    var spacing_fails: u32 = 0;
    var collision_fails: u32 = 0;

    while (waypoints.items.len < p.max_pts and total_length < p.L_max and attempts < max_attempts) : (attempts += 1) {
        const prev = waypoints.items[waypoints.items.len - 1];
        const prev_tangent = if (waypoints.items.len > 1)
            waypoints.items[waypoints.items.len - 1].p.sub(waypoints.items[waypoints.items.len - 2].p).normalize()
        else blk: {
            // Random initial direction in X-Z plane (horizontal) with slight vertical component
            const rand_yaw = self.rng.random().float(f32) * 2.0 * std.math.pi;
            const rand_pitch = (self.rng.random().float(f32) * 2.0 - 1.0) * Math.radians(15.0); // ±15° initial pitch
            break :blk Vec3.init(
                @cos(rand_yaw) * @cos(rand_pitch),
                @sin(rand_pitch),
                @sin(rand_yaw) * @cos(rand_pitch),
            ).normalize();
        };

        // Generate candidate point
        const step_len = p.s_min + self.rng.random().float(f32) * (p.s_max - p.s_min);

        // Horizontal turn (yaw in X-Z plane)
        const yaw_angle = (self.rng.random().float(f32) * 2.0 - 1.0) * p.max_turn_deg;
        const yaw_rad = Math.radians(yaw_angle);

        // Vertical turn (pitch change)
        const pitch_angle = (self.rng.random().float(f32) * 2.0 - 1.0) * Math.radians(20.0); // ±20° pitch change

        const cos_yaw = @cos(yaw_rad);
        const sin_yaw = @sin(yaw_rad);

        // Apply yaw rotation in X-Z plane (horizontal)
        var new_dir = Vec3.init(
            prev_tangent.x() * cos_yaw - prev_tangent.z() * sin_yaw,
            prev_tangent.y(),
            prev_tangent.x() * sin_yaw + prev_tangent.z() * cos_yaw,
        );

        // Apply pitch change (up/down)
        const current_y = new_dir.y();
        const xz_length = @sqrt(new_dir.x() * new_dir.x() + new_dir.z() * new_dir.z());
        const new_y = current_y + @sin(pitch_angle) * xz_length;
        const scale = @sqrt(new_dir.x() * new_dir.x() + new_dir.z() * new_dir.z() + new_y * new_y);

        new_dir = Vec3.init(
            new_dir.x() / scale,
            new_y / scale,
            new_dir.z() / scale,
        );
        new_dir.normalize_inplace();

        var candidate_pos = prev.p.add(new_dir.scale(step_len));

        // Apply vertical constraints
        candidate_pos.set_z(Math.clamp(candidate_pos.z(), p.z_lo, p.z_hi));
        const dz = @abs(candidate_pos.z() - prev.p.z());
        if (dz > p.dz_max) {
            const clamped_dz = @min(dz, p.dz_max);
            const sign: f32 = if (candidate_pos.z() > prev.p.z()) 1.0 else -1.0;
            candidate_pos.set_z(prev.p.z() + clamped_dz * sign);
        }

        // Check bounds
        if (!p.bounds.contains(candidate_pos)) {
            bounds_fails += 1;
            continue;
        }

        // Check turning radius
        if (waypoints.items.len > 1) {
            const prev2 = waypoints.items[waypoints.items.len - 2].p;
            const r = computeTurnRadius(prev2, prev.p, candidate_pos);
            if (r < p.R_min) {
                radius_fails += 1;
                continue;
            }
        }

        // Poisson-disc spacing check
        var too_close = false;
        for (waypoints.items) |w| {
            if (w.p.sub(candidate_pos).length() < p.s_min * 0.5) {
                too_close = true;
                break;
            }
        }
        if (too_close) {
            spacing_fails += 1;
            continue;
        }

        // Collision check (TODO: implement bullet sweep)
        if (!try self.segmentCollisionFree(prev.p, candidate_pos, p.drone_radius, p.sweep_margin, collision_world)) {
            collision_fails += 1;
            continue;
        }

        // Compute yaw (rotation around Y-axis in XZ plane for Y-up world)
        const motion_yaw = std.math.atan2(new_dir.x(), new_dir.z());

        const q_prev = Quaternion.from_axis_angle(Vec3.init(0, 1, 0), Math.degrees(prev.yaw));
        const q_motion = Quaternion.from_axis_angle(Vec3.init(0, 1, 0), Math.degrees(motion_yaw));

        // Bias toward motion using SLERP on the circle
        const t_bias = Math.clamp(p.yaw_bias_w, 0.0, 1.0);
        var q_biased = Quaternion.slerp(q_prev, q_motion, t_bias);

        // Convert back to yaw (radians). to_euler() returns [pitch, yaw, roll]
        var desired_yaw = q_biased.to_euler()[1];

        // Add noise
        desired_yaw += (self.rng.random().float(f32) * 2.0 - 1.0) * Math.radians(p.yaw_noise_deg);

        // Clamp per-step yaw change
        const max_yaw_change = Math.radians(p.max_turn_deg);
        const d = shortestAngleDiff(prev.yaw, desired_yaw);
        if (@abs(d) > max_yaw_change) {
            desired_yaw = prev.yaw + max_yaw_change * (d / @abs(d));
        }

        desired_yaw = wrapPi(desired_yaw);
        try waypoints.append(.{ .p = candidate_pos, .yaw = desired_yaw });
        total_length += step_len;

        if (total_length >= p.L_min and waypoints.items.len >= 2) {
            break;
        }
    }

    // std.debug.print("PathSystem: Generated {d} waypoints, total_length={d:.2}, L_min={d:.2}, attempts={d}/{d}\n", .{
    //     waypoints.items.len,
    //     total_length,
    //     p.L_min,
    //     attempts,
    //     max_attempts,
    // });
    // std.debug.print("  Failure breakdown - bounds:{d}, radius:{d}, spacing:{d}, collision:{d}\n", .{
    //     bounds_fails,
    //     radius_fails,
    //     spacing_fails,
    //     collision_fails,
    // });

    if (total_length < p.L_min) {
        return error.PathTooShort;
    }

    // Curve fitting
    var beziers = std.ArrayList(CubicBezier3).init(self.allocator);
    defer beziers.deinit();

    const tangents = try self.computeCatmullRomTangents(waypoints.items, p.tension_base);
    defer self.allocator.free(tangents);

    for (0..waypoints.items.len - 1) |i| {
        const p0 = waypoints.items[i].p;
        const p1 = waypoints.items[i + 1].p;
        const t0 = tangents[i];
        const t1 = tangents[i + 1];

        const bezier = CubicBezier3{
            .p0 = p0,
            .p1 = p0.add(t0.scale(1.0 / 3.0)),
            .p2 = p1.sub(t1.scale(1.0 / 3.0)),
            .p3 = p1,
        };
        try beziers.append(bezier);
    }

    // Tessellation
    var samples = std.ArrayList(Vec3).init(self.allocator);
    defer samples.deinit();
    var sample_tangents = std.ArrayList(Vec3).init(self.allocator);
    defer sample_tangents.deinit();
    var s_cumsum = std.ArrayList(f32).init(self.allocator);
    defer s_cumsum.deinit();
    var curvature_list = std.ArrayList(f32).init(self.allocator);
    defer curvature_list.deinit();

    var arc_len: f32 = 0.0;
    for (beziers.items) |bez| {
        try self.tessellate(bez, p.flatness_eps, &samples, &sample_tangents, &s_cumsum, &curvature_list, &arc_len);
    }

    // Time parameterization with velocity/acceleration constraints
    var t_cumsum = std.ArrayList(f32).init(self.allocator);
    defer t_cumsum.deinit();
    var velocities = std.ArrayList(f32).init(self.allocator);
    defer velocities.deinit();

    try self.computeTimeParameterization(
        samples.items,
        curvature_list.items,
        s_cumsum.items,
        p.v_max,
        p.a_max,
        p.j_max,
        &t_cumsum,
        &velocities,
    );

    // Compute quaternion orientations
    var orientations = std.ArrayList(Quaternion).init(self.allocator);
    defer orientations.deinit();

    try self.computeOrientations(
        samples.items,
        sample_tangents.items,
        curvature_list.items,
        waypoints.items,
        s_cumsum.items,
        &orientations,
    );

    const path_result = PathResult{
        .waypoints = try waypoints.toOwnedSlice(),
        .beziers = try beziers.toOwnedSlice(),
        .samples = try samples.toOwnedSlice(),
        .tangents = try sample_tangents.toOwnedSlice(),
        .curvature = try curvature_list.toOwnedSlice(),
        .s_cumsum = try s_cumsum.toOwnedSlice(),
        .t_cumsum = try t_cumsum.toOwnedSlice(),
        .velocities = try velocities.toOwnedSlice(),
        .orientations = try orientations.toOwnedSlice(),
        .path_entities = null,
        .allocator = self.allocator,
    };

    return path_result;
}

// pub fn visualizePath(
//     self: *Self,
//     result: *const PathResult,
//     waypoint_color: [3]f32,
// ) !struct { path: Core.EntityID, waypoints: Core.EntityID } {
//     const ecs = self.ecs;

//     // Find min/max velocities for color mapping
//     var v_min: f32 = std.math.floatMax(f32);
//     var v_max: f32 = -std.math.floatMax(f32);
//     for (result.velocities) |v| {
//         v_min = @min(v_min, v);
//         v_max = @max(v_max, v);
//     }

//     const v_range = v_max - v_min;

//     // Create path line segments
//     var path_vertices = std.ArrayList(Mesh.Vertex).init(self.allocator);
//     defer path_vertices.deinit();

//     for (0..result.samples.len - 1) |i| {
//         const sample = result.samples[i];
//         const next_sample = result.samples[i + 1];

//         const v = if (i < result.velocities.len) result.velocities[i] else result.velocities[result.velocities.len - 1];
//         const t = if (v_range > 1e-6) (v - v_min) / v_range else 0.0;

//         const color = [3]f32{
//             t,
//             1.0 - t,
//             0.0,
//         };

//         try path_vertices.append(.{
//             .position = [3]f32{ sample.x(), sample.y(), sample.z() },
//             .color = color,
//         });
//         try path_vertices.append(.{
//             .position = [3]f32{ next_sample.x(), next_sample.y(), next_sample.z() },
//             .color = color,
//         });
//     }

//     const path_vertices_owned = try self.allocator.dupe(Mesh.Vertex, path_vertices.items);
//     const path_mesh = try Mesh.init(self.allocator, path_vertices_owned, null, drawPathLines);
//     path_mesh.drawType = Mesh.DrawType.lines.toGL();

//     const path_mesh_name = try std.fmt.allocPrint(self.allocator, "path_lines_{d}", .{self.path_counter});
//     defer self.allocator.free(path_mesh_name);
//     const path_mesh_name_owned = try self.allocator.dupe(u8, path_mesh_name);

//     try ecs.world.resource_manager.meshes.put(path_mesh_name_owned, .{
//         .mesh = path_mesh,
//         .instance_count = 0,
//     });

//     const path_transform = Transform.TransformComponent.init(self.allocator);
//     const path_renderer = try Renderer.Renderable.init(self.allocator, path_mesh_name_owned);

//     const path_entity = try ecs.spawn(.{
//         path_transform,
//         path_renderer,
//     });

//     // Create waypoint points
//     var waypoint_vertices = std.ArrayList(Mesh.Vertex).init(self.allocator);
//     defer waypoint_vertices.deinit();

//     for (result.waypoints) |wp| {
//         try waypoint_vertices.append(.{
//             .position = [3]f32{ wp.p.x(), wp.p.y(), wp.p.z() },
//             .color = waypoint_color,
//         });
//     }

//     const waypoint_vertices_owned = try self.allocator.dupe(Mesh.Vertex, waypoint_vertices.items);
//     const waypoint_mesh = try Mesh.init(self.allocator, waypoint_vertices_owned, null, drawPathPoints);
//     waypoint_mesh.drawType = Mesh.DrawType.points.toGL();

//     const waypoint_mesh_name = try std.fmt.allocPrint(self.allocator, "path_waypoints_{d}", .{self.path_counter});
//     defer self.allocator.free(waypoint_mesh_name);
//     const waypoint_mesh_name_owned = try self.allocator.dupe(u8, waypoint_mesh_name);

//     try ecs.world.resource_manager.meshes.put(waypoint_mesh_name_owned, .{
//         .mesh = waypoint_mesh,
//         .instance_count = 0,
//     });

//     const waypoint_transform = Transform.TransformComponent.init(self.allocator);
//     const waypoint_renderer = try Renderer.Renderable.init(self.allocator, waypoint_mesh_name_owned);

//     const waypoint_entity = try ecs.spawn(.{
//         waypoint_transform,
//         waypoint_renderer,
//     });

//     std.debug.print("Created path visualization - path entity: {}, waypoints entity: {} (v_min={d:.2}, v_max={d:.2})\n", .{ path_entity, waypoint_entity, v_min, v_max });

//     return .{ .path = path_entity, .waypoints = waypoint_entity };
// }

fn segmentCollisionFree(self: *Self, from: Vec3, to: Vec3, radius: f32, margin: f32, world: ?bt.CbtWorldHandle) !bool {
    _ = self;
    const collision_world = world orelse return true;

    const total_radius = radius + margin;

    // Use multiple ray tests around a cylinder to approximate sweep
    const num_rays = 8;
    const angle_step = 2.0 * std.math.pi / @as(f32, @floatFromInt(num_rays));

    // Test center ray
    var ray_result: bt.CbtRayCastResult = undefined;
    const from_arr = [3]f32{ from.x(), from.y(), from.z() };
    const to_arr = [3]f32{ to.x(), to.y(), to.z() };

    if (bt.cbtWorldRayTestClosest(
        collision_world,
        &from_arr,
        &to_arr,
        -1, // collision_filter_group
        -1, // collision_filter_mask
        0, // flags
        &ray_result,
    )) {
        return false;
    }

    // Test rays around the perimeter
    const dir = to.sub(from).normalize();
    const perpendicular = if (@abs(dir.z()) < 0.9)
        Vec3.cross(dir, Vec3.init(0, 0, 1)).normalize()
    else
        Vec3.cross(dir, Vec3.init(1, 0, 0)).normalize();

    var i: usize = 0;
    while (i < num_rays) : (i += 1) {
        const angle = @as(f32, @floatFromInt(i)) * angle_step;
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);

        // Rotate perpendicular vector around direction
        const offset = perpendicular.scale(total_radius * cos_a)
            .add(Vec3.cross(dir, perpendicular).scale(total_radius * sin_a));

        const offset_from = from.add(offset);
        const offset_to = to.add(offset);

        const offset_from_arr = [3]f32{ offset_from.x(), offset_from.y(), offset_from.z() };
        const offset_to_arr = [3]f32{ offset_to.x(), offset_to.y(), offset_to.z() };

        if (bt.cbtWorldRayTestClosest(
            collision_world,
            &offset_from_arr,
            &offset_to_arr,
            -1,
            -1,
            0,
            &ray_result,
        )) {
            return false;
        }
    }

    return true;
}

fn cloneCollisionWorld(source_world: bt.CbtWorldHandle, allocator: std.mem.Allocator) !bt.CbtWorldHandle {
    _ = allocator;

    // Create new world
    const new_world = bt.cbtWorldCreate() orelse return error.FailedToCreateWorld;

    // Copy gravity
    var gravity: [3]f32 = undefined;
    bt.cbtWorldGetGravity(source_world, &gravity);
    bt.cbtWorldSetGravity(new_world, &gravity);

    // Clone all static bodies (collision geometry)
    const num_bodies = bt.cbtWorldGetNumBodies(source_world);
    var i: i32 = 0;
    while (i < num_bodies) : (i += 1) {
        const body = bt.cbtWorldGetBody(source_world, i);

        // Only clone static bodies (mass == 0)
        const mass = bt.cbtBodyGetMass(body);
        if (mass == 0.0) {
            // Get shape
            const shape = bt.cbtBodyGetShape(body);

            // Get transform (4x3 matrix)
            var transform: [4][3]f32 = undefined;
            bt.cbtBodyGetCenterOfMassTransform(body, &transform);

            // Create new body - allocate first
            const new_body = bt.cbtBodyAllocate();

            // Create with same shape, transform and mass (mass, transform, shape)
            bt.cbtBodyCreate(new_body, 0.0, &transform, shape);
            bt.cbtWorldAddBody(new_world, new_body);
        }
    }

    return new_world;
}

fn computeCatmullRomTangents(self: *Self, points: []const Waypoint, tension: f32) ![]Vec3 {
    const n = points.len;
    const tangents = try self.allocator.alloc(Vec3, n);

    for (0..n) |i| {
        if (i == 0) {
            tangents[i] = points[1].p.sub(points[0].p).scale(tension);
        } else if (i == n - 1) {
            tangents[i] = points[n - 1].p.sub(points[n - 2].p).scale(tension);
        } else {
            const p_prev = points[i - 1].p;
            const p_curr = points[i].p;
            const p_next = points[i + 1].p;

            const d1 = p_curr.sub(p_prev).length();
            const d2 = p_next.sub(p_curr).length();

            const alpha = 0.5; // centripetal
            const t1 = std.math.pow(f32, d1, alpha);
            const t2 = std.math.pow(f32, d2, alpha);

            const m = p_next.sub(p_prev).scale(1.0 / (t1 + t2));
            tangents[i] = m.scale(tension);
        }
    }

    return tangents;
}

fn tessellate(
    self: *Self,
    bezier: CubicBezier3,
    flatness: f32,
    samples: *std.ArrayList(Vec3),
    tangents: *std.ArrayList(Vec3),
    s_cumsum: *std.ArrayList(f32),
    curvature: *std.ArrayList(f32),
    arc_len: *f32,
) !void {
    var stack = std.ArrayList(TessSegment).init(self.allocator);
    defer stack.deinit();

    try stack.append(.{ .bez = bezier, .t0 = 0.0, .t1 = 1.0 });

    while (stack.items.len > 0) {
        const seg = stack.pop();
        const bez = seg.bez;

        if (isFlatEnough(bez, flatness)) {
            if (samples.items.len == 0 or !bez.p0.approx_eq(samples.items[samples.items.len - 1], 1e-4)) {
                try samples.append(bez.p0);
                const tang = bez.tangentAt(0.0).normalize();
                try tangents.append(tang);
                try s_cumsum.append(arc_len.*);
                const curv = bez.curvatureAt(0.0);
                try curvature.append(curv);
            }

            try samples.append(bez.p3);
            const tang = bez.tangentAt(1.0).normalize();
            try tangents.append(tang);
            const seg_len = bez.p3.sub(bez.p0).length();
            arc_len.* += seg_len;
            try s_cumsum.append(arc_len.*);
            const curv = bez.curvatureAt(1.0);
            try curvature.append(curv);
        } else {
            const split = bez.subdivide(0.5);
            try stack.append(.{ .bez = split[1], .t0 = (seg.t0 + seg.t1) / 2.0, .t1 = seg.t1 });
            try stack.append(.{ .bez = split[0], .t0 = seg.t0, .t1 = (seg.t0 + seg.t1) / 2.0 });
        }
    }
}

fn computeTimeParameterization(
    self: *Self,
    samples: []const Vec3,
    curvature: []const f32,
    s_cumsum: []const f32,
    v_max: f32,
    a_max: f32,
    j_max: f32,
    t_cumsum: *std.ArrayList(f32),
    velocities: *std.ArrayList(f32),
) !void {
    _ = j_max; // TODO: implement jerk limiting

    if (samples.len == 0) return;

    // Forward pass: compute maximum velocity considering curvature and acceleration
    var v_forward = try self.allocator.alloc(f32, samples.len);
    defer self.allocator.free(v_forward);

    v_forward[0] = 0.0; // start from rest

    for (1..samples.len) |i| {
        // Curvature-limited velocity: v = sqrt(a_max / curvature)
        const curv = @max(curvature[i], 1e-6);
        const v_curv = @sqrt(a_max / curv);

        // Acceleration-limited velocity from previous point
        const ds = s_cumsum[i] - s_cumsum[i - 1];
        const v_prev = v_forward[i - 1];
        const v_accel = @sqrt(v_prev * v_prev + 2.0 * a_max * ds);

        v_forward[i] = @min(@min(v_curv, v_accel), v_max);
    }

    // Backward pass: ensure deceleration constraints
    var v_backward = try self.allocator.alloc(f32, samples.len);
    defer self.allocator.free(v_backward);

    v_backward[samples.len - 1] = 0.0; // end at rest

    var i: usize = samples.len - 1;
    while (i > 0) : (i -= 1) {
        const ds = s_cumsum[i] - s_cumsum[i - 1];
        const v_next = v_backward[i];
        const v_decel = @sqrt(v_next * v_next + 2.0 * a_max * ds);

        v_backward[i - 1] = @min(v_decel, v_max);
    }

    // Take minimum of forward and backward passes
    var time: f32 = 0.0;
    try t_cumsum.append(0.0);
    try velocities.append(0.0);

    for (1..samples.len) |idx| {
        const v = @min(v_forward[idx], v_backward[idx]);
        try velocities.append(v);

        const ds = s_cumsum[idx] - s_cumsum[idx - 1];
        const v_avg = (velocities.items[idx - 1] + v) / 2.0;
        const dt = if (v_avg > 1e-3) ds / v_avg else 0.0;

        time += dt;
        try t_cumsum.append(time);
    }
}

fn computeOrientations(
    self: *Self,
    samples: []const Vec3,
    tangents: []const Vec3,
    curvature: []const f32,
    waypoints: []const Waypoint,
    s_cumsum: []const f32,
    orientations: *std.ArrayList(Quaternion),
) !void {
    _ = self;

    for (0..samples.len) |i| {
        const forward = tangents[i].normalize();

        // Compute bank angle from curvature (coordinated turn)
        const g = 9.81;
        const curv = curvature[i];
        const v = 5.0; // nominal velocity for bank calculation
        const bank_angle = std.math.atan2(v * v * curv, g);

        // Get yaw from nearest waypoint
        var nearest_idx: usize = 0;
        var min_dist = std.math.inf(f32);
        for (waypoints, 0..) |_, idx| {
            const dist = @abs(s_cumsum[i] - if (idx < s_cumsum.len) s_cumsum[idx] else s_cumsum[s_cumsum.len - 1]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest_idx = idx;
            }
        }
        const yaw = waypoints[nearest_idx].yaw;

        // Compute pitch from forward vector (OpenGL Y-up: pitch is vertical angle)
        const pitch = std.math.asin(forward.y());

        // Build quaternion: yaw → pitch → roll (OpenGL Y-up convention)
        const q_yaw = Quaternion.from_axis_angle(Vec3.init(0, 1, 0), Math.degrees(yaw));
        const q_pitch = Quaternion.from_axis_angle(Vec3.init(1, 0, 0), Math.degrees(pitch));
        const q_roll = Quaternion.from_axis_angle(Vec3.init(0, 0, 1), Math.degrees(bank_angle));

        const orientation = q_yaw.multiply(q_pitch).multiply(q_roll).normalize();
        try orientations.append(orientation);
    }
}

fn generateRandomWaypoint(self: *Self, p: CreatePathParams, world: ?bt.CbtWorldHandle) !Waypoint {
    const max_attempts = 1000;
    var attempts: u32 = 0;

    std.debug.print("Generating random waypoint...\n", .{});

    while (attempts < max_attempts) : (attempts += 1) {
        // Random position within bounds
        const x = p.bounds.min.x() + self.rng.random().float(f32) * (p.bounds.max.x() - p.bounds.min.x());
        const y = p.bounds.min.y() + self.rng.random().float(f32) * (p.bounds.max.y() - p.bounds.min.y());
        const z = Math.clamp(
            p.z_lo + self.rng.random().float(f32) * (p.z_hi - p.z_lo),
            p.bounds.min.z(),
            p.bounds.max.z(),
        );

        const pos = Vec3.init(x, y, z);

        // Check if position is collision-free (test with small sphere)
        const collision_world = world orelse {
            // No physics, just return the position
            const yaw = self.rng.random().float(f32) * 2.0 * std.math.pi - std.math.pi;
            return Waypoint{ .p = pos, .yaw = yaw };
        };

        // Test if this point is in free space using raycasts in multiple directions
        var collision_free = true;
        const test_dirs = [_]Vec3{
            Vec3.init(1, 0, 0),
            Vec3.init(-1, 0, 0),
            Vec3.init(0, 1, 0),
            Vec3.init(0, -1, 0),
            Vec3.init(0, 0, 1),
            Vec3.init(0, 0, -1),
        };

        for (test_dirs) |dir| {
            const test_point = pos.add(dir.scale(p.drone_radius + p.sweep_margin));
            var ray_result: bt.CbtRayCastResult = undefined;
            const from_arr = [3]f32{ pos.x(), pos.y(), pos.z() };
            const to_arr = [3]f32{ test_point.x(), test_point.y(), test_point.z() };

            if (bt.cbtWorldRayTestClosest(
                collision_world,
                &from_arr,
                &to_arr,
                -1,
                -1,
                0,
                &ray_result,
            )) {
                collision_free = false;
                break;
            }
        }

        if (collision_free) {
            const yaw = self.rng.random().float(f32) * 2.0 * std.math.pi - std.math.pi;
            std.debug.print("Found valid waypoint after {d} attempts at ({d:.2}, {d:.2}, {d:.2})\n", .{ attempts, pos.x(), pos.y(), pos.z() });
            return Waypoint{ .p = pos, .yaw = yaw };
        }
    }

    std.debug.print("Failed to find valid waypoint after {d} attempts\n", .{max_attempts});
    return error.NoValidStartingPoint;
}

const TessSegment = struct {
    bez: CubicBezier3,
    t0: f32,
    t1: f32,
};

pub const AABB3 = struct {
    min: Vec3,
    max: Vec3,

    pub fn contains(self: AABB3, p: Vec3) bool {
        return p.x() >= self.min.x() and p.x() <= self.max.x() and
            p.y() >= self.min.y() and p.y() <= self.max.y() and
            p.z() >= self.min.z() and p.z() <= self.max.z();
    }
};

pub const CubicBezier3 = struct {
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,

    pub fn eval(self: CubicBezier3, t: f32) Vec3 {
        const u = 1.0 - t;
        const uu = u * u;
        const uuu = uu * u;
        const tt = t * t;
        const ttt = tt * t;

        var p = self.p0.scale(uuu);
        p = p.add(self.p1.scale(3.0 * uu * t));
        p = p.add(self.p2.scale(3.0 * u * tt));
        p = p.add(self.p3.scale(ttt));

        return p;
    }

    pub fn tangentAt(self: CubicBezier3, t: f32) Vec3 {
        const u = 1.0 - t;
        const uu = u * u;
        const tt = t * t;

        var deriv = self.p1.sub(self.p0).scale(3.0 * uu);
        deriv = deriv.add(self.p2.sub(self.p1).scale(6.0 * u * t));
        deriv = deriv.add(self.p3.sub(self.p2).scale(3.0 * tt));

        return deriv;
    }

    pub fn curvatureAt(self: CubicBezier3, t: f32) f32 {
        const first = self.tangentAt(t);
        const dt = 0.01;
        const t_next = @min(t + dt, 1.0);
        const second = self.tangentAt(t_next).sub(first).scale(1.0 / dt);

        const cross_prod = Vec3.cross(first, second);
        const numerator = cross_prod.length();
        const denominator = std.math.pow(f32, first.length(), 3.0);

        if (denominator < 1e-6) return 0.0;
        return numerator / denominator;
    }

    pub fn subdivide(self: CubicBezier3, t: f32) [2]CubicBezier3 {
        const q0 = Vec3.lerp(self.p0, self.p1, t);
        const q1 = Vec3.lerp(self.p1, self.p2, t);
        const q2 = Vec3.lerp(self.p2, self.p3, t);

        const r0 = Vec3.lerp(q0, q1, t);
        const r1 = Vec3.lerp(q1, q2, t);

        const s = Vec3.lerp(r0, r1, t);

        return .{
            CubicBezier3{ .p0 = self.p0, .p1 = q0, .p2 = r0, .p3 = s },
            CubicBezier3{ .p0 = s, .p1 = r1, .p2 = q2, .p3 = self.p3 },
        };
    }
};

fn isFlatEnough(bez: CubicBezier3, eps: f32) bool {
    const ux = 3.0 * bez.p1.x() - 2.0 * bez.p0.x() - bez.p3.x();
    const uy = 3.0 * bez.p1.y() - 2.0 * bez.p0.y() - bez.p3.y();
    const uz = 3.0 * bez.p1.z() - 2.0 * bez.p0.z() - bez.p3.z();
    const vx = 3.0 * bez.p2.x() - 2.0 * bez.p3.x() - bez.p0.x();
    const vy = 3.0 * bez.p2.y() - 2.0 * bez.p3.y() - bez.p0.y();
    const vz = 3.0 * bez.p2.z() - 2.0 * bez.p3.z() - bez.p0.z();

    const max_u = @max(@abs(ux), @max(@abs(uy), @abs(uz)));
    const max_v = @max(@abs(vx), @max(@abs(vy), @abs(vz)));

    return @max(max_u, max_v) <= eps;
}

fn computeTurnRadius(p0: Vec3, p1: Vec3, p2: Vec3) f32 {
    const v1 = p1.sub(p0);
    const v2 = p2.sub(p1);

    const cross = Vec3.cross(v1, v2);
    const area = cross.length() / 2.0;

    const a = v1.length();
    const b = v2.length();
    const c = p2.sub(p0).length();

    if (area < 1e-6) return std.math.inf(f32);

    return (a * b * c) / (4.0 * area);
}

fn wrapPi(angle: f32) f32 {
    const pi = std.math.pi;
    var a = angle;
    while (a > pi) a -= 2.0 * pi;
    while (a < -pi) a += 2.0 * pi;
    return a;
}

fn shortestAngleDiff(a: f32, b: f32) f32 {
    var diff = b - a;
    diff = wrapPi(diff);
    return diff;
}

fn lerpAngle(a: f32, b: f32, t: f32) f32 {
    const d = shortestAngleDiff(a, b);
    const tt = Math.clamp(t, 0.0, 1.0);
    return wrapPi(a + d * tt);
}
