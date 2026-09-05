//! vision_raster.zig - purpose-built CPU ray-caster for the headless
//! training API (NAV_STACK.md build item D, decided 2026-09-05:
//! depth+seg first, Pi Cam 3 stereo targets, in-repo).
//!
//! The headless scene is analytic: a ground plane at GROUND_Y and sphere
//! obstacles. Ray-casting that is exact (no tessellation error),
//! deterministic, and fast: 640x480 x N spheres is a few ms per frame.
//!
//! Outputs per camera:
//!   depth: [h*w]f32 metric ray length (inf on sky)
//!   seg:   [h*w]u8  class id (0 sky, 1 floor, 2 obstacle, 3 goal/pad)
//!
//! Sensor models layered on top (sensor_models.zig, next):
//!   rolling shutter = per-row time offset applied to the camera pose
//!   (pose interpolated along the trajectory between policy steps),
//!   depth noise = range-dependent sigma + grazing-angle dropout (ToF).

const std = @import("std");
const Math = @import("core/Math.zig");
const Vec3 = Math.Vec3;
const Quaternion = Math.Quaternion;

pub const SegClass = enum(u8) { sky = 0, floor = 1, obstacle = 2, goal = 3 };

pub const Camera = struct {
    width: u32 = 640,
    height: u32 = 480,
    // intrinsics from hfov; cx/cy centered
    hfov_deg: f32 = 75.0, // Pi Cam 3 standard-lens target (NAV_STACK decisions)
    mount_pos: Vec3, // body frame, meters
    mount_yaw: f32 = 0.0, // body frame, rad (0 = nose)

    pub fn focal(self: Camera) f32 {
        const w: f32 = @floatFromInt(self.width);
        return 0.5 * w / @tan(0.5 * self.hfov_deg * std.math.pi / 180.0);
    }
};

pub const Frame = struct {
    depth: []f32,
    seg: []u8,
    width: u32,
    height: u32,
};

pub const RasterObstacle = struct { center: Vec3, radius: f32 };

pub const RasterScene = struct {
    ground_y: f32,
    obstacles: []const RasterObstacle,
    goal: Vec3,
    goal_radius: f32, // rendered as a flat pad disc on the ground
};

const Hit = struct { t: f32, class: SegClass };

fn raySphere(ro: Vec3, rd: Vec3, c: Vec3, r: f32) ?f32 {
    const oc = ro.sub(c);
    const b = oc.dot(rd);
    const cc = oc.dot(oc) - r * r;
    const disc = b * b - cc;
    if (disc < 0.0) return null;
    const s = @sqrt(disc);
    const t0 = -b - s;
    if (t0 > 1e-4) return t0;
    const t1 = -b + s;
    if (t1 > 1e-4) return t1;
    return null;
}

fn rayPlaneDown(ro: Vec3, rd: Vec3, y: f32) ?f32 {
    // plane y = const, normal +Y; only hit when looking downward-ish
    if (@abs(rd.y()) < 1e-6) return null;
    const t = (y - ro.y()) / rd.y();
    if (t > 1e-4) return t;
    return null;
}

fn rayDisc(ro: Vec3, rd: Vec3, y: f32, center: Vec3, r: f32) ?f32 {
    const t = rayPlaneDown(ro, rd, y) orelse return null;
    const p = ro.add(rd.scale(t));
    const dx = p.x() - center.x();
    const dz = p.z() - center.z();
    if (dx * dx + dz * dz <= r * r) return t;
    return null;
}

/// Cast one ray; returns the nearest hit (sky if none).
pub fn castRay(scene: RasterScene, ro: Vec3, rd: Vec3) Hit {
    var best: Hit = .{ .t = std.math.inf(f32), .class = .sky };
    if (rayPlaneDown(ro, rd, scene.ground_y)) |t| {
        if (t < best.t) best = .{ .t = t, .class = .floor };
    }
    for (scene.obstacles) |ob| {
        if (raySphere(ro, rd, ob.center, ob.radius)) |t| {
            if (t < best.t) best = .{ .t = t, .class = .obstacle };
        }
    }
    // pad marker: disc slightly proud of the floor so it wins ties
    if (rayDisc(ro, rd, scene.ground_y + 0.01, scene.goal, scene.goal_radius)) |t| {
        if (t < best.t) best = .{ .t = t, .class = .goal };
    }
    return best;
}

/// Render a full frame. body_pos/body_quat: world pose of the drone;
/// row_time_offset (rolling shutter) is applied by the CALLER passing a
/// per-row adjusted pose via renderRow - this function renders the
/// zero-offset (global shutter) frame; renderRow is the per-row hook.
pub fn render(scene: RasterScene, cam: Camera, body_pos: Vec3, body_quat: Quaternion, frame: Frame) void {
    for (0..cam.height) |row| {
        renderRow(scene, cam, body_pos, body_quat, frame, row);
    }
}

pub fn renderRow(scene: RasterScene, cam: Camera, body_pos: Vec3, body_quat: Quaternion, frame: Frame, row: usize) void {
    const f = cam.focal();
    const cx = 0.5 * @as(f32, @floatFromInt(cam.width));
    const cy = 0.5 * @as(f32, @floatFromInt(cam.height));
    // camera origin: body pose * mount offset, mount yaw folded in
    const yaw_q = Quaternion.from_axis_angle(Vec3.init(0, 1, 0), cam.mount_yaw);
    const q = body_quat.multiply(yaw_q);
    const ro = body_pos.add(Vec3.rotate_by_quaternion(cam.mount_pos, body_quat));
    const v: f32 = (cy - @as(f32, @floatFromInt(row))) / f;
    for (0..cam.width) |col| {
        const u: f32 = (@as(f32, @floatFromInt(col)) - cx) / f;
        // camera looks along body +X (nose), up = +Y after yaw fold
        const dir_cam = Vec3.init(1.0, v, u); // +X forward, +Y up, +Z right (right-handed: fwd x up = +Z)
        const rd = Vec3.rotate_by_quaternion(dir_cam, q).normalize();
        const hit = castRay(scene, ro, rd);
        const idx = row * cam.width + col;
        frame.depth[idx] = hit.t;
        frame.seg[idx] = @intFromEnum(hit.class);
    }
}

test "ray-sphere hit and miss" {
    const ro = Vec3.init(0, 1, 0);
    const rd = Vec3.init(1, 0, 0);
    try std.testing.expect(raySphere(ro, rd, Vec3.init(5, 1, 0), 1.0) != null);
    try std.testing.expect(raySphere(ro, rd, Vec3.init(5, 5, 0), 1.0) == null);
}

test "ground plane below" {
    const ro = Vec3.init(0, 2, 0);
    const down = Vec3.init(0, -1, 0);
    const t = rayPlaneDown(ro, down, 0.05).?;
    try std.testing.expectApproxEqAbs(@as(f32, 1.95), t, 1e-5);
}
