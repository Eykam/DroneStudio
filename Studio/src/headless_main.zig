//! Headless physics-only episode runner for the auto-researcher inner loop.
//!
//! Design (HEADLESS_API.md v1, mandated implementation 2026-09-03):
//! - No window, no GL, no renderer, no ECS. One Bullet world, one rigid
//!   body (the drone), analytic obstacle/floor/goal checks identical to
//!   QuadNavEnv so results stay comparable across backends.
//! - Fast loop REUSES FlightController.RateController + PIDController
//!   verbatim (his firmware code path), dt = 1/500 s, 25 fast steps per
//!   policy step (20 Hz), matching QuadNavEnv's constants.
//! - Concrete scenes are sampled Python-side and passed in on reset
//!   (spawn/goal/obstacle list), so the Zig engine and the numpy engine
//!   run the exact same scenario - parity is in dynamics, not RNG.
//! - Actuation model v1: PID body torques + collective thrust applied
//!   directly (QuadNavEnv parity). The app's motor mixer + motor lag path
//!   (FlightControllerComponent.updateMotorMixer/updateMotorLag) is the
//!   v1.1 fidelity upgrade, gated on verifying its free parameters.
//!
//! Protocol: stdio JSON-lines, one message per line.
//!   <- {"cmd":"reset","seed":7,"scene":{"spawn":[x,y,z],"goal":[x,y,z],
//!        "obstacles":[[x,y,z,r],...],"extent":40,"max_steps":250,
//!        "dynamics_noise":0.05}}
//!   -> {"obs":[15 floats],"reward":0,"done":false}
//!   <- {"cmd":"step","action":[roll,pitch,yaw,thrust]}   // each in [-1,1]
//!   -> {"obs":[...],"reward":R,"done":B,"info":{"collided":B,"succeeded":B,"steps":N}}
//!   <- {"cmd":"ping"} -> {"ok":true}
//!   <- {"cmd":"close"} -> exits

const std = @import("std");
const Math = @import("core/Math.zig");
const FC = @import("core/ecs/components/FlightController.zig");

const bullet = @cImport({
    @cInclude("cbullet.h");
});

const Vec3 = Math.Vec3;
const Quaternion = Math.Quaternion;

// --- constants: identical to autoresearch/env_quad.py ---------------------
const FAST_DT: f32 = 1.0 / 500.0;
const FAST_PER_POLICY: u32 = 25;
const MAX_RATES = [3]f32{ 10.47, 10.47, 5.24 };
const MAX_THRUST: f32 = 40.0;
const FILTER_TAU: f32 = 0.05;
const MASS: f32 = 1.5;
const IXX: f32 = 0.040;
const IYY: f32 = 0.040;
const IZZ: f32 = 0.047;
const DRONE_RADIUS: f32 = 0.3;
const GOAL_RADIUS: f32 = 2.0;
const GROUND_Y: f32 = 0.05;

const Obstacle = struct { center: Vec3, radius: f32 };

const Scene = struct {
    spawn: Vec3,
    goal: Vec3,
    obstacles: []Obstacle,
    extent: f32 = 40.0,
    max_steps: u32 = 250,
    dynamics_noise: f32 = 0.05,
};

fn vecFromJson(v: std.json.Value) !Vec3 {
    const arr = switch (v) {
        .array => |a| a,
        else => return error.BadVec,
    };
    if (arr.items.len != 3) return error.BadVec;
    var out: [3]f32 = undefined;
    for (arr.items, 0..) |item, i| {
        out[i] = switch (item) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => return error.BadVec,
        };
    }
    return Vec3.init(out[0], out[1], out[2]);
}

fn f32FromJson(v: std.json.Value, default: f32) f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |n| @floatFromInt(n),
        else => default,
    };
}

const World = struct {
    world: bullet.CbtWorldHandle,
    body: bullet.CbtBodyHandle,
    shape: bullet.CbtShapeHandle,
    rate_ctrl: FC.RateController,
    filtered_rates: [3]f32 = .{ 0, 0, 0 },
    filtered_thrust: f32 = 0.0,
    rng: std.Random.Xoshiro256,
    scene: Scene = undefined,
    steps: u32 = 0,
    prev_dist: f32 = 0,
    collided: bool = false,
    succeeded: bool = false,
    done: bool = false,

    fn init(alloc: std.mem.Allocator) !*World {
        _ = alloc;
        const w = try std.heap.page_allocator.create(World);
        w.world = bullet.cbtWorldCreate();
        var gravity = [3]f32{ 0, -9.81, 0 };
        bullet.cbtWorldSetGravity(w.world, &gravity);

        // Drone body: sphere proxy (radius 0.3 - exact parity with
        // QuadNavEnv's point + 0.3 m collision margin), mass/inertia from
        // prefabs/Drone.zig (computed values: 1.5 kg, 0.040/0.040/0.047).
        w.shape = bullet.cbtShapeAllocate(bullet.CBT_SHAPE_TYPE_SPHERE);
        bullet.cbtShapeSphereCreate(w.shape, DRONE_RADIUS);
        w.body = bullet.cbtBodyAllocate();
        const identity = [4][3]f32{
            .{ 1, 0, 0 },
            .{ 0, 1, 0 },
            .{ 0, 0, 1 },
            .{ 0, 2, 0 },
        };
        bullet.cbtBodyCreate(w.body, MASS, @ptrCast(&identity), w.shape);
        const inertia = [3]f32{ IXX, IYY, IZZ };
        bullet.cbtBodySetMassProps(w.body, MASS, &inertia);
        bullet.cbtBodySetDamping(w.body, 0.0, 0.0); // QuadNavEnv has no aero damping
        bullet.cbtBodySetActivationState(w.body, bullet.CBT_DISABLE_DEACTIVATION);
        bullet.cbtWorldAddBody(w.world, w.body);

        w.rate_ctrl = FC.RateController.init(1.0, 1.0); // integrator/output limits
        w.rng = std.Random.Xoshiro256.init(0);
        return w;
    }

    fn gaussian(self: *World) f32 {
        // Box-Muller on Xoshiro256 - statistically equivalent to numpy's
        // normal(0,1); per-draw values differ by engine, that's fine for a
        // stochastic disturbance term.
        const r = self.rng.random();
        var ua = r.float(f32);
        if (ua < 1e-7) ua = 1e-7;
        const ub = r.float(f32);
        return @sqrt(-2.0 * @log(ua)) * @cos(2.0 * std.math.pi * ub);
    }

    fn bodyQuat(self: *World) Quaternion {
        var t: [4][3]f32 = undefined;
        bullet.cbtBodyGetCenterOfMassTransform(self.body, @ptrCast(&t));
        // Bullet basis is row-major 3x3 (+ position row). Mat3.from_array
        // takes column-major - transpose.
        const cm = [9]f32{
            t[0][0], t[1][0], t[2][0],
            t[0][1], t[1][1], t[2][1],
            t[0][2], t[1][2], t[2][2],
        };
        return Quaternion.from_mat3(Math.Mat3.from_array(cm));
    }

    fn bodyPos(self: *World) Vec3 {
        var p: [3]f32 = undefined;
        bullet.cbtBodyGetCenterOfMassPosition(self.body, &p);
        return Vec3.from_array(p);
    }

    fn bodyVel(self: *World) Vec3 {
        var v: [3]f32 = undefined;
        bullet.cbtBodyGetLinearVelocity(self.body, &v);
        return Vec3.from_array(v);
    }

    fn bodyOmega(self: *World) Vec3 {
        var o: [3]f32 = undefined;
        bullet.cbtBodyGetAngularVelocity(self.body, &o);
        return Vec3.from_array(o);
    }

    fn reset(self: *World, scene: Scene, seed: u64) void {
        self.scene = scene;
        self.rng = std.Random.Xoshiro256.init(seed);
        self.rate_ctrl.reset();
        self.filtered_rates = .{ 0, 0, 0 };
        self.filtered_thrust = 0;
        self.steps = 0;
        self.collided = false;
        self.succeeded = false;
        self.done = false;

        const t = [4][3]f32{
            .{ 1, 0, 0 },
            .{ 0, 1, 0 },
            .{ 0, 0, 1 },
            .{ scene.spawn.x(), scene.spawn.y(), scene.spawn.z() },
        };
        bullet.cbtBodySetCenterOfMassTransform(self.body, @ptrCast(&t));
        const zero = [3]f32{ 0, 0, 0 };
        bullet.cbtBodySetLinearVelocity(self.body, &zero);
        bullet.cbtBodySetAngularVelocity(self.body, &zero);
        bullet.cbtBodySetActivationState(self.body, bullet.CBT_ACTIVE_TAG);
        self.prev_dist = scene.goal.sub(scene.spawn).length();
    }

    /// One fast (500 Hz) step: input filter -> his RateController PID ->
    /// torque + thrust -> Bullet integration. Mirrors env_quad._fast_step.
    fn fastStep(self: *World, desired_rates: [3]f32, thrust_cmd: f32) void {
        const alpha = FAST_DT / (FILTER_TAU + FAST_DT);
        // disturbance on the commanded rates, like env_quad (noise * 5.0)
        var noisy = desired_rates;
        if (self.scene.dynamics_noise > 0) {
            inline for (0..3) |i| {
                noisy[i] += self.gaussian() * self.scene.dynamics_noise * 5.0;
            }
        }
        inline for (0..3) |i| {
            self.filtered_rates[i] += alpha * (noisy[i] - self.filtered_rates[i]);
        }
        self.filtered_thrust += alpha * (thrust_cmd - self.filtered_thrust);

        const omega = self.bodyOmega();
        const torque_body = self.rate_ctrl.update(self.filtered_rates, omega, FAST_DT);

        const q = self.bodyQuat();
        const thrust_world = Vec3.init(0, self.filtered_thrust, 0).rotate_by_quaternion(q);
        const torque_world = Vec3.from_array(torque_body).rotate_by_quaternion(q);

        var f = [3]f32{ thrust_world.x(), thrust_world.y(), thrust_world.z() };
        var tq = [3]f32{ torque_world.x(), torque_world.y(), torque_world.z() };
        bullet.cbtBodyApplyCentralForce(self.body, &f);
        bullet.cbtBodyApplyTorque(self.body, &tq);
        _ = bullet.cbtWorldStepSimulation(self.world, FAST_DT, 1, FAST_DT);
    }

    /// One policy (20 Hz) step: 25 fast steps + reward/termination,
    /// identical formulas to env_quad.step.
    fn policyStep(self: *World, action: [4]f32) struct { reward: f32, done: bool } {
        const desired = [3]f32{
            std.math.clamp(action[0], -1.0, 1.0) * MAX_RATES[0],
            std.math.clamp(action[1], -1.0, 1.0) * MAX_RATES[1],
            std.math.clamp(action[2], -1.0, 1.0) * MAX_RATES[2],
        };
        const thrust_cmd = (std.math.clamp(action[3], -1.0, 1.0) + 1.0) / 2.0 * MAX_THRUST;
        for (0..FAST_PER_POLICY) |_| {
            if (self.done) break;
            self.fastStep(desired, thrust_cmd);
        }
        self.steps += 1;

        const pos = self.bodyPos();
        const dist = self.scene.goal.sub(pos).length();
        var reward = (self.prev_dist - dist) - 0.01;
        self.prev_dist = dist;

        if (pos.y() < GROUND_Y) {
            reward -= 5.0;
            self.collided = true;
            self.done = true;
        }
        if (!self.done) {
            for (self.scene.obstacles) |ob| {
                if (ob.center.sub(pos).length() < ob.radius + DRONE_RADIUS) {
                    reward -= 5.0;
                    self.collided = true;
                    self.done = true;
                    break;
                }
            }
        }
        if (!self.done and dist < GOAL_RADIUS) {
            reward += 10.0;
            self.succeeded = true;
            self.done = true;
        }
        if (!self.done and self.steps >= self.scene.max_steps) {
            self.done = true;
        }
        return .{ .reward = reward, .done = self.done };
    }

    /// Observation: 15 floats, exact QuadNavEnv layout:
    /// rel_goal/extent(3), vel/10(3), gravity-in-body/9.81(3),
    /// omega/10(3), nearest-obstacle-rel/extent(3)
    fn obs(self: *World) [15]f32 {
        const pos = self.bodyPos();
        const vel = self.bodyVel();
        const omega = self.bodyOmega();
        const q = self.bodyQuat();
        const q_conj = q.conjugate();

        const extent = @max(self.scene.extent, 1.0);
        const rel_goal = self.scene.goal.sub(pos).scale(1.0 / extent);
        const v = vel.scale(1.0 / 10.0);
        const g_body = Vec3.init(0, -9.81, 0).rotate_by_quaternion(q_conj).scale(1.0 / 9.81);
        const rates = omega.scale(1.0 / 10.0);
        var rel_obs = Vec3.zero();
        if (self.scene.obstacles.len > 0) {
            var best_d: f32 = std.math.inf(f32);
            var best = Vec3.zero();
            for (self.scene.obstacles) |ob| {
                const dvec = ob.center.sub(pos);
                const d = dvec.length();
                if (d < best_d) {
                    best_d = d;
                    best = dvec;
                }
            }
            rel_obs = best.scale(1.0 / extent);
        }
        return .{
            rel_goal.x(), rel_goal.y(), rel_goal.z(),
            v.x(),        v.y(),        v.z(),
            g_body.x(),   g_body.y(),   g_body.z(),
            rates.x(),    rates.y(),    rates.z(),
            rel_obs.x(),  rel_obs.y(),  rel_obs.z(),
        };
    }
};

fn writeObsReply(writer: anytype, w: *World, reward: f32, done: bool, with_info: bool) !void {
    const o = w.obs();
    try writer.print("{{\"obs\":[{d:.6}", .{o[0]});
    for (o[1..]) |x| try writer.print(",{d:.6}", .{x});
    try writer.print("],\"reward\":{d:.6},\"done\":{}", .{ reward, done });
    if (with_info) {
        try writer.print(",\"info\":{{\"collided\":{},\"succeeded\":{},\"steps\":{d}}}", .{
            w.collided, w.succeeded, w.steps,
        });
    }
    try writer.writeAll("}\n");
    try writer.context.flush();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const stdin = std.io.getStdIn().reader();
    var stdout_buf = std.io.bufferedWriter(std.io.getStdOut().writer());
    const stdout = stdout_buf.writer();

    var world = try World.init(alloc);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    // Scene data (obstacle list) must outlive the line that parsed it -
    // separate lifetime from the per-line parse arena.
    var scene_arena = std.heap.ArenaAllocator.init(alloc);
    defer scene_arena.deinit();

    while (true) {
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();
        const line = stdin.readUntilDelimiterOrEofAlloc(a, '\n', 1 << 20) catch |e| {
            if (e == error.EndOfStream) break;
            return e;
        } orelse break;
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;

        const parsed = std.json.parseFromSlice(std.json.Value, a, trimmed, .{}) catch {
            try stdout.writeAll("{\"error\":\"bad json\"}\n");
            try stdout_buf.flush();
            continue;
        };
        const root = parsed.value;
        if (root != .object) continue;
        const cmd_v = root.object.get("cmd") orelse continue;
        const cmd = switch (cmd_v) {
            .string => |s| s,
            else => continue,
        };

        if (std.mem.eql(u8, cmd, "ping")) {
            try stdout.writeAll("{\"ok\":true}\n");
            try stdout_buf.flush();
        } else if (std.mem.eql(u8, cmd, "close")) {
            break;
        } else if (std.mem.eql(u8, cmd, "reset")) {
            const scene_v = root.object.get("scene") orelse {
                try stdout.writeAll("{\"error\":\"reset needs scene\"}\n");
                try stdout_buf.flush();
                continue;
            };
            var scene = Scene{
                .spawn = try vecFromJson(scene_v.object.get("spawn").?),
                .goal = try vecFromJson(scene_v.object.get("goal").?),
                .obstacles = &.{},
            };
            if (scene_v.object.get("extent")) |e| scene.extent = f32FromJson(e, 40.0);
            if (scene_v.object.get("max_steps")) |m| scene.max_steps = @intFromFloat(f32FromJson(m, 250));
            if (scene_v.object.get("dynamics_noise")) |n| scene.dynamics_noise = f32FromJson(n, 0.05);
            if (scene_v.object.get("obstacles")) |obs_v| {
                const arr = obs_v.array;
                _ = scene_arena.reset(.retain_capacity);
                var list = try scene_arena.allocator().alloc(Obstacle, arr.items.len);
                for (arr.items, 0..) |item, i| {
                    const oarr = item.array.items;
                    if (oarr.len != 4) return error.BadObstacle;
                    var vals: [4]f32 = undefined;
                    for (oarr, 0..) |ov, j| vals[j] = f32FromJson(ov, 0);
                    list[i] = .{
                        .center = Vec3.init(vals[0], vals[1], vals[2]),
                        .radius = vals[3],
                    };
                }
                scene.obstacles = list;
            }
            const seed: u64 = if (root.object.get("seed")) |s| switch (s) {
                .integer => |n| @intCast(n),
                .float => |f| @intFromFloat(f),
                else => 0,
            } else 0;
            world.reset(scene, seed);
            try writeObsReply(stdout, world, 0.0, false, false);
        } else if (std.mem.eql(u8, cmd, "step")) {
            const act_v = root.object.get("action") orelse continue;
            const arr = act_v.array.items;
            if (arr.len != 4) continue;
            var action: [4]f32 = undefined;
            for (arr, 0..) |av, i| action[i] = f32FromJson(av, 0);
            const r = world.policyStep(action);
            try writeObsReply(stdout, world, r.reward, r.done, true);
        }
    }
}
