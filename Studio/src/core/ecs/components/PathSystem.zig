const std = @import("std");
const Math = @import("../../Math.zig");
const bt = @import("../../bindings/c.zig").bullet;
const gl = @import("../../bindings/gl.zig");
const Mesh = @import("../../Mesh.zig");
const Core = @import("../Core.zig");
const Transform = @import("./Transform.zig");
const Renderer = @import("./Renderer.zig");

const Vec3 = Math.Vec3;
const Quaternion = Math.Quaternion;
const glad = gl.glad;

const ECSManager = @import("../ECSManager.zig");

const Self = @This();

fn drawPathLines(mesh: *Mesh) void {
    glad.glLineWidth(3.0);
    glad.glBindVertexArray(mesh.meta.VAO);
    glad.glDrawArrays(mesh.drawType, 0, @intCast(mesh.vertices.len));
    glad.glLineWidth(1.0);
}

fn drawPathPoints(mesh: *Mesh) void {
    glad.glPointSize(8.0);
    glad.glBindVertexArray(mesh.meta.VAO);
    glad.glDrawArrays(mesh.drawType, 0, @intCast(mesh.vertices.len));
    glad.glPointSize(1.0);
}

allocator: std.mem.Allocator,
rng: std.Random.DefaultPrng,
paths: std.ArrayList(PathResult),
ecs: *ECSManager,
path_counter: usize = 0,

pub fn init(allocator: std.mem.Allocator, ecs: *ECSManager) !*Self {
    const system = try allocator.create(Self);
    system.* = .{
        .allocator = allocator,
        .rng = std.Random.DefaultPrng.init(0),
        .paths = std.ArrayList(PathResult).init(allocator),
        .ecs = ecs,
    };
    return system;
}

pub fn deinit(self: *Self) void {
    for (self.paths.items) |*path| {
        path.deinit();
    }
    self.paths.deinit();
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

pub fn getSceneBounds(self: *Self) !AABB3 {
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

    return AABB3{
        .min = scene_min,
        .max = scene_max,
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
    path_entity: ?Core.EntityID,
    waypoint_entity: ?Core.EntityID,
    visible: bool = true,
    allocator: std.mem.Allocator,

    pub fn setVisible(self: *PathResult, ecs: *ECSManager, visible: bool) void {
        self.visible = visible;

        if (self.path_entity) |entity_id| {
            if (ecs.render_system.renderables.get(entity_id)) |renderable_ptr| {
                renderable_ptr.is_visible = visible;
            }
        }

        if (self.waypoint_entity) |entity_id| {
            if (ecs.render_system.renderables.get(entity_id)) |renderable_ptr| {
                renderable_ptr.is_visible = visible;
            }
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

pub fn createPath(self: *Self, p: CreatePathParams, anchors: ?[]const Waypoint) !PathResult {
    self.rng = std.Random.DefaultPrng.init(p.seed);

    // Generate waypoints
    var waypoints = std.ArrayList(Waypoint).init(self.allocator);
    defer waypoints.deinit();

    if (anchors) |anchor_slice| {
        try waypoints.appendSlice(anchor_slice);
    } else {
        // Generate random starting waypoint
        const start = try self.generateRandomWaypoint(p);
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
        if (!try self.segmentCollisionFree(prev.p, candidate_pos, p.drone_radius, p.sweep_margin)) {
            collision_fails += 1;
            continue;
        }

        // Compute yaw
        const motion_yaw = std.math.atan2(new_dir.y(), new_dir.x());
        const noise = (self.rng.random().float(f32) * 2.0 - 1.0) * Math.radians(p.yaw_noise_deg);
        var new_yaw = motion_yaw * p.yaw_bias_w + noise;

        // Clamp yaw change
        const yaw_diff = shortestAngleDiff(prev.yaw, new_yaw);
        const max_yaw_change = Math.radians(p.max_turn_deg);
        if (@abs(yaw_diff) > max_yaw_change) {
            const sign: f32 = if (yaw_diff > 0) 1.0 else -1.0;
            new_yaw = prev.yaw + max_yaw_change * sign;
        }

        new_yaw = wrapPi(new_yaw);

        try waypoints.append(.{ .p = candidate_pos, .yaw = new_yaw });
        total_length += step_len;

        if (total_length >= p.L_min and waypoints.items.len >= 2) {
            break;
        }
    }

    std.debug.print("PathSystem: Generated {d} waypoints, total_length={d:.2}, L_min={d:.2}, attempts={d}/{d}\n", .{
        waypoints.items.len,
        total_length,
        p.L_min,
        attempts,
        max_attempts,
    });
    std.debug.print("  Failure breakdown - bounds:{d}, radius:{d}, spacing:{d}, collision:{d}\n", .{
        bounds_fails,
        radius_fails,
        spacing_fails,
        collision_fails,
    });

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

    var path_result = PathResult{
        .waypoints = try waypoints.toOwnedSlice(),
        .beziers = try beziers.toOwnedSlice(),
        .samples = try samples.toOwnedSlice(),
        .tangents = try sample_tangents.toOwnedSlice(),
        .curvature = try curvature_list.toOwnedSlice(),
        .s_cumsum = try s_cumsum.toOwnedSlice(),
        .t_cumsum = try t_cumsum.toOwnedSlice(),
        .velocities = try velocities.toOwnedSlice(),
        .orientations = try orientations.toOwnedSlice(),
        .path_entity = null,
        .waypoint_entity = null,
        .allocator = self.allocator,
    };

    const entities = try self.visualizePath(&path_result, .{ 1.0, 1.0, 0.0 });
    path_result.path_entity = entities.path;
    path_result.waypoint_entity = entities.waypoints;

    // Hide all previous paths
    for (self.paths.items) |*existing_path| {
        existing_path.setVisible(self.ecs, false);
    }

    try self.paths.append(path_result);
    self.path_counter += 1;

    std.debug.print("Stored path in buffer. Total paths: {d}\n", .{self.paths.items.len});

    return path_result;
}

pub fn visualizePath(
    self: *Self,
    result: *const PathResult,
    waypoint_color: [3]f32,
) !struct { path: Core.EntityID, waypoints: Core.EntityID } {
    const ecs = self.ecs;

    // Find min/max velocities for color mapping
    var v_min: f32 = std.math.floatMax(f32);
    var v_max: f32 = -std.math.floatMax(f32);
    for (result.velocities) |v| {
        v_min = @min(v_min, v);
        v_max = @max(v_max, v);
    }

    const v_range = v_max - v_min;

    // Create path line segments
    var path_vertices = std.ArrayList(Mesh.Vertex).init(self.allocator);
    defer path_vertices.deinit();

    for (0..result.samples.len - 1) |i| {
        const sample = result.samples[i];
        const next_sample = result.samples[i + 1];

        const v = if (i < result.velocities.len) result.velocities[i] else result.velocities[result.velocities.len - 1];
        const t = if (v_range > 1e-6) (v - v_min) / v_range else 0.0;

        const color = [3]f32{
            t,
            1.0 - t,
            0.0,
        };

        try path_vertices.append(.{
            .position = [3]f32{ sample.x(), sample.y(), sample.z() },
            .color = color,
        });
        try path_vertices.append(.{
            .position = [3]f32{ next_sample.x(), next_sample.y(), next_sample.z() },
            .color = color,
        });
    }

    const path_vertices_owned = try self.allocator.dupe(Mesh.Vertex, path_vertices.items);
    const path_mesh = try Mesh.init(self.allocator, path_vertices_owned, null, drawPathLines);
    path_mesh.drawType = Mesh.DrawType.lines.toGL();

    const path_mesh_name = try std.fmt.allocPrint(self.allocator, "path_lines_{d}", .{self.path_counter});
    defer self.allocator.free(path_mesh_name);
    const path_mesh_name_owned = try self.allocator.dupe(u8, path_mesh_name);

    try ecs.world.resource_manager.meshes.put(path_mesh_name_owned, .{
        .mesh = path_mesh,
        .instance_count = 0,
    });

    const path_transform = Transform.TransformComponent.init(self.allocator);
    const path_renderer = try Renderer.Renderable.init(self.allocator, path_mesh_name_owned);

    const path_entity = try ecs.spawn(.{
        path_transform,
        path_renderer,
    });

    // Create waypoint points
    var waypoint_vertices = std.ArrayList(Mesh.Vertex).init(self.allocator);
    defer waypoint_vertices.deinit();

    for (result.waypoints) |wp| {
        try waypoint_vertices.append(.{
            .position = [3]f32{ wp.p.x(), wp.p.y(), wp.p.z() },
            .color = waypoint_color,
        });
    }

    const waypoint_vertices_owned = try self.allocator.dupe(Mesh.Vertex, waypoint_vertices.items);
    const waypoint_mesh = try Mesh.init(self.allocator, waypoint_vertices_owned, null, drawPathPoints);
    waypoint_mesh.drawType = Mesh.DrawType.points.toGL();

    const waypoint_mesh_name = try std.fmt.allocPrint(self.allocator, "path_waypoints_{d}", .{self.path_counter});
    defer self.allocator.free(waypoint_mesh_name);
    const waypoint_mesh_name_owned = try self.allocator.dupe(u8, waypoint_mesh_name);

    try ecs.world.resource_manager.meshes.put(waypoint_mesh_name_owned, .{
        .mesh = waypoint_mesh,
        .instance_count = 0,
    });

    const waypoint_transform = Transform.TransformComponent.init(self.allocator);
    const waypoint_renderer = try Renderer.Renderable.init(self.allocator, waypoint_mesh_name_owned);

    const waypoint_entity = try ecs.spawn(.{
        waypoint_transform,
        waypoint_renderer,
    });

    std.debug.print("Created path visualization - path entity: {}, waypoints entity: {} (v_min={d:.2}, v_max={d:.2})\n", .{ path_entity, waypoint_entity, v_min, v_max });

    return .{ .path = path_entity, .waypoints = waypoint_entity };
}

fn segmentCollisionFree(self: *Self, from: Vec3, to: Vec3, radius: f32, margin: f32) !bool {
    const physics_thread = self.ecs.collision_system.physics_thread orelse return true;

    const total_radius = radius + margin;

    // Use multiple ray tests around a cylinder to approximate sweep
    const num_rays = 8;
    const angle_step = 2.0 * std.math.pi / @as(f32, @floatFromInt(num_rays));

    // Test center ray
    var ray_result: bt.CbtRayCastResult = undefined;
    const from_arr = [3]f32{ from.x(), from.y(), from.z() };
    const to_arr = [3]f32{ to.x(), to.y(), to.z() };

    if (bt.cbtWorldRayTestClosest(
        physics_thread.bullet_world,
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
            physics_thread.bullet_world,
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

        // Compute pitch from forward vector
        const pitch = std.math.asin(-forward.z());

        // Build quaternion: yaw → pitch → roll
        const q_yaw = Quaternion.from_axis_angle(Vec3.init(0, 0, 1), Math.degrees(yaw));
        const q_pitch = Quaternion.from_axis_angle(Vec3.init(1, 0, 0), Math.degrees(pitch));
        const q_roll = Quaternion.from_axis_angle(Vec3.init(0, 1, 0), Math.degrees(bank_angle));

        const orientation = q_yaw.multiply(q_pitch).multiply(q_roll).normalize();
        try orientations.append(orientation);
    }
}

fn generateRandomWaypoint(self: *Self, p: CreatePathParams) !Waypoint {
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
        const physics_thread = self.ecs.collision_system.physics_thread orelse {
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
                physics_thread.bullet_world,
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
