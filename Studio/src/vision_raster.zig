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
    mount_pitch: f32 = 0.0, // body frame, rad about +Z (right); -pi/2 = straight down (ToF)

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
    rgb: ?[]u8 = null, // optional 3 bytes/px display shading (shade())
};

pub const RasterObstacle = struct { center: Vec3, radius: f32 };

pub const RasterScene = struct {
    ground_y: f32,
    obstacles: []const RasterObstacle,
    goal: Vec3,
    goal_radius: f32, // rendered as a flat pad disc on the ground
    visual: Visual = .{},
};

/// Visual domain-randomization parameters (rung-2, 2026-09-09: extend
/// option D for RGB - NAV_STACK.md rung-2 kickoff decisions). Defaults
/// reproduce the pre-DR look bit-for-bit; the dataset generator sends a
/// seeded per-scene visual block with the render command.
pub const Visual = struct {
    sun_dir: ?Vec3 = null, // null -> legacy (-0.4, 0.85, 0.35) normalized
    ambient: f32 = 0.35, // lam = ambient + (1-ambient) * max(0, n.sun)
    fog_scale: f32 = 45.0, // fog = 1 - exp(-t / fog_scale)
    sky_lo: [3]f32 = .{ 26.0, 34.0, 46.0 },
    sky_hi: [3]f32 = .{ 92.0, 108.0, 126.0 },
    fog_col: [3]f32 = .{ 38.0, 50.0, 66.0 },
    floor_col: [3]f32 = .{ 150.0, 148.0, 142.0 },
    obstacle_col: [3]f32 = .{ 92.0, 106.0, 124.0 },
    goal_col: [3]f32 = .{ 30.0, 150.0, 84.0 },
    checker_m: f32 = 0.0, // 0 = off; floor checker square size in meters
    checker_gain: f32 = 0.85, // albedo multiplier on alternating squares
    noise_gain: f32 = 0.0, // 0 = off; multiplicative value-noise amplitude on floor/obstacle albedo
    noise_scale_m: f32 = 0.35, // fine noise feature size in meters (2 octaves: 1x + 4x)
    plank_m: f32 = 0.0, // 0 = off; floor plank strip width in meters (strips along x)
    plank_gain: f32 = 0.55, // seam albedo multiplier
    exposure: f32 = 1.0, // global gain, clamped to [0,255]
};

const Hit = struct { t: f32, class: SegClass, n: Vec3 };

/// Sun-lit lambert shade + distance fog -> RGB model input / scene panel.
/// All look parameters come from Visual so the dataset generator can
/// randomize them per scene (domain randomization, rung-2).
/// World-anchored integer-lattice hash -> [0,1). Stable under camera motion
/// (function of world position only), so texture is trackable frame to frame.
fn hash2(ix: i32, iz: i32) f32 {
    var h: u32 = @bitCast(ix *% 374761393 +% iz *% 668265263);
    h = (h ^ (h >> 13)) *% 1274126177;
    h = h ^ (h >> 16);
    return @as(f32, @floatFromInt(h & 0xFFFFFF)) / @as(f32, 0x1000000);
}

fn smoot(t: f32) f32 {
    return t * t * (3.0 - 2.0 * t);
}

/// Bilinear value noise on the XZ plane, one octave at `scale` meters.
fn vnoise(x: f32, z: f32, scale: f32) f32 {
    const xs = x / scale;
    const zs = z / scale;
    const x0: i32 = @intFromFloat(@floor(xs));
    const z0: i32 = @intFromFloat(@floor(zs));
    const fx = smoot(xs - @floor(xs));
    const fz = smoot(zs - @floor(zs));
    const a = hash2(x0, z0);
    const b = hash2(x0 + 1, z0);
    const c = hash2(x0, z0 + 1);
    const d = hash2(x0 + 1, z0 + 1);
    const lo = a + (b - a) * fx;
    const hi = c + (d - c) * fx;
    return lo + (hi - lo) * fz;
}

fn mix3(a: [3]f32, b: [3]f32, t: f32) [3]f32 {
    return .{ a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t };
}

fn toU8(c: [3]f32, gain: f32) [3]u8 {
    return .{
        @intFromFloat(std.math.clamp(c[0] * gain, 0.0, 255.0)),
        @intFromFloat(std.math.clamp(c[1] * gain, 0.0, 255.0)),
        @intFromFloat(std.math.clamp(c[2] * gain, 0.0, 255.0)),
    };
}

fn shade(hit: Hit, rd: Vec3, ro: Vec3, v: Visual) [3]u8 {
    if (hit.class == .sky) {
        const g = std.math.clamp(0.5 + 0.5 * rd.y(), 0.0, 1.0);
        return toU8(mix3(v.sky_lo, v.sky_hi, g), v.exposure);
    }
    var base: [3]f32 = switch (hit.class) {
        .floor => v.floor_col,
        .obstacle => v.obstacle_col,
        .goal => v.goal_col,
        .sky => unreachable,
    };
    if ((hit.class == .floor or hit.class == .obstacle) and
        (v.checker_m > 0.0 or v.noise_gain > 0.0 or v.plank_m > 0.0))
    {
        const p = ro.add(rd.scale(hit.t));
        if (hit.class == .floor and v.checker_m > 0.0) {
            const cx: i32 = @intFromFloat(@floor(p.x() / v.checker_m));
            const cz: i32 = @intFromFloat(@floor(p.z() / v.checker_m));
            if (@mod(cx + cz, 2) != 0) base = .{ base[0] * v.checker_gain, base[1] * v.checker_gain, base[2] * v.checker_gain };
        }
        if (hit.class == .floor and v.plank_m > 0.0) {
            const pi: i32 = @intFromFloat(@floor(p.z() / v.plank_m));
            const pv = 0.9 + 0.2 * hash2(pi, 7919); // per-plank brightness
            base = .{ base[0] * pv, base[1] * pv, base[2] * pv };
            const seam = @mod(p.z(), v.plank_m);
            if (seam < 0.012 or seam > v.plank_m - 0.012)
                base = .{ base[0] * v.plank_gain, base[1] * v.plank_gain, base[2] * v.plank_gain };
        }
        if (v.noise_gain > 0.0) {
            const n = 0.65 * vnoise(p.x(), p.z(), v.noise_scale_m) +
                0.35 * vnoise(p.x(), p.z(), v.noise_scale_m * 4.0);
            const g = 1.0 + v.noise_gain * (n - 0.5) * 2.0;
            base = .{ base[0] * g, base[1] * g, base[2] * g };
        }
    }
    const sun = if (v.sun_dir) |sd| sd.normalize() else Vec3.init(-0.4, 0.85, 0.35).normalize();
    const lam = v.ambient + (1.0 - v.ambient) * @max(0.0, hit.n.dot(sun));
    const fog = 1.0 - @exp(-hit.t / v.fog_scale);
    const c = mix3(.{ base[0] * lam, base[1] * lam, base[2] * lam }, v.fog_col, fog * 0.55);
    return toU8(c, v.exposure);
}

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
    const up = Vec3.init(0, 1, 0);
    var best: Hit = .{ .t = std.math.inf(f32), .class = .sky, .n = up };
    if (rayPlaneDown(ro, rd, scene.ground_y)) |t| {
        if (t < best.t) best = .{ .t = t, .class = .floor, .n = up };
    }
    for (scene.obstacles) |ob| {
        if (raySphere(ro, rd, ob.center, ob.radius)) |t| {
            if (t < best.t) {
                const p = ro.add(rd.scale(t));
                best = .{ .t = t, .class = .obstacle, .n = p.sub(ob.center).normalize() };
            }
        }
    }
    // pad marker: disc slightly proud of the floor so it wins ties
    if (rayDisc(ro, rd, scene.ground_y + 0.01, scene.goal, scene.goal_radius)) |t| {
        if (t < best.t) best = .{ .t = t, .class = .goal, .n = up };
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
    // NB: Math.from_axis_angle takes DEGREES (codebase convention); the
    // Camera API is radians.
    const r2d = 180.0 / std.math.pi;
    const yaw_q = Quaternion.from_axis_angle(Vec3.init(0, 1, 0), cam.mount_yaw * r2d);
    const pitch_q = Quaternion.from_axis_angle(Vec3.init(0, 0, 1), cam.mount_pitch * r2d);
    const q = body_quat.multiply(yaw_q).multiply(pitch_q);
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
        if (frame.rgb) |rgb| {
            const c = shade(hit, rd, ro, scene.visual);
            rgb[idx * 3] = c[0];
            rgb[idx * 3 + 1] = c[1];
            rgb[idx * 3 + 2] = c[2];
        }
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

// ---------------------------------------------------------------------
// ToF rangefinder ring (rung-2, 2026-09-09) - decided 8-sensor suite:
// 4 cardinal nav ToF (0/90/180/270 deg) + 4 diagonal arm/prop-line
// proximity monitors (45/135/225/315 deg). Sensor-suite decision doc:
// diagonals are keep-and-accept arm monitors - on the real frame ~56%
// of each diagonal cone reads arm/motor at a fixed 40-140mm baseline
// (weighted 2:1 toward 40-100mm), ~44% is clear to the scene. The sim
// reproduces that split so the estimator learns the diagonals as
// proximity channels, not nav sensors.
// Noise model (NAV_STACK.md): range-dependent sigma, dropout on grazing
// incidence, max-range clamp. VL53L9CX class: ~4m practical ceiling.

pub const TofRole = enum { nav, arm_monitor };

pub const TofSensor = struct {
    name: []const u8,
    azimuth_deg: f32, // body yaw about +Y, 0 = nose (+X)
    elevation_deg: f32 = 0.0, // pitch about +Z, -90 = straight down
    offset: Vec3, // body frame, meters
    role: TofRole = .nav,
};

pub const TofConfig = struct {
    // VL53L9CX data brief (Farnell 4759648 / ST): ranging <5cm - 9m.
    max_range_m: f32 = 9.0,
    min_range_m: f32 = 0.05,
    // Noise v0 (TOF_SIM_SCOPE.md): sigma = base + k*range^2 (ST family
    // behavior; k is an ESTIMATE pending a bench capture), scaled by an
    // ambient-light factor (1.0 = benign indoor; >1 = strong ambient).
    sigma_base_mm: f32 = 5.0,
    sigma_k_mm_per_m2: f32 = 3.0, // ESTIMATE
    ambient_factor: f32 = 1.0,
    base_dropout: f32 = 0.01,
    // Dropout ramps past ~80% of max range (scope v0).
    far_dropout_start: f32 = 0.8,
    far_dropout_max: f32 = 0.5, // ESTIMATE: 50% no-return at the 9m edge
    graze_cos: f32 = 0.3, // below this |cos(incidence)| dropout ramps up
    arm_block_p: f32 = 0.56, // measured ~56% of diagonal cone blocked
    // arm reading distribution: 2:1 weighted U(40,100) : U(100,140) mm
    arm_lo_mm: f32 = 40.0,
    arm_mid_mm: f32 = 100.0,
    arm_hi_mm: f32 = 140.0,
};

pub const TofReading = struct {
    name: []const u8,
    clean_mm: u32, // noiseless ray length; 65535 = no surface within clamp
    range_mm: u32, // noise model applied; 0 when valid=false
    valid: bool,
    cls: SegClass,
};

/// The decided suite (sensor-suite-coverage decision, 2026-09-09).
pub fn defaultTofRing() [8]TofSensor {
    const r_nav: f32 = 0.035; // frame edge
    const r_arm: f32 = 0.040; // arm root
    return .{
        .{ .name = "N", .azimuth_deg = 0, .offset = Vec3.init(r_nav, 0, 0) },
        .{ .name = "E", .azimuth_deg = 90, .offset = Vec3.init(0, 0, r_nav) },
        .{ .name = "S", .azimuth_deg = 180, .offset = Vec3.init(-r_nav, 0, 0) },
        .{ .name = "W", .azimuth_deg = 270, .offset = Vec3.init(0, 0, -r_nav) },
        .{ .name = "NE", .azimuth_deg = 45, .offset = Vec3.init(r_arm * 0.7071, 0, r_arm * 0.7071), .role = .arm_monitor },
        .{ .name = "SE", .azimuth_deg = 135, .offset = Vec3.init(-r_arm * 0.7071, 0, r_arm * 0.7071), .role = .arm_monitor },
        .{ .name = "SW", .azimuth_deg = 225, .offset = Vec3.init(-r_arm * 0.7071, 0, -r_arm * 0.7071), .role = .arm_monitor },
        .{ .name = "NW", .azimuth_deg = 315, .offset = Vec3.init(r_arm * 0.7071, 0, -r_arm * 0.7071), .role = .arm_monitor },
    };
}

/// Read one ToF sensor against the analytic scene.
/// rng: caller-owned seeded PRNG (deterministic per episode+frame).
pub fn readTof(
    scene: RasterScene,
    s: TofSensor,
    body_pos: Vec3,
    body_quat: Quaternion,
    rng: std.Random,
    cfg: TofConfig,
) TofReading {
    const yaw_q = Quaternion.from_axis_angle(Vec3.init(0, 1, 0), s.azimuth_deg);
    const pitch_q = Quaternion.from_axis_angle(Vec3.init(0, 0, 1), s.elevation_deg);
    const q = body_quat.multiply(yaw_q).multiply(pitch_q);
    const ro = body_pos.add(Vec3.rotate_by_quaternion(s.offset, body_quat));
    const rd = Vec3.rotate_by_quaternion(Vec3.init(1.0, 0, 0), q).normalize();
    const hit = castRay(scene, ro, rd);

    var clean_mm: f32 = 65535.0;
    var cls: SegClass = .sky;
    var grazing = false;
    if (!std.math.isInf(hit.t)) {
        if (hit.t <= cfg.max_range_m) {
            clean_mm = hit.t * 1000.0;
            cls = hit.class;
            grazing = @abs(rd.dot(hit.n)) < cfg.graze_cos;
        } else {
            cls = hit.class; // surface exists but beyond clamp
        }
    }

    // arm/prop-line self-occupancy on diagonal monitors: the nearer of
    // the frame baseline draw and any scene surface wins (photon race).
    if (s.role == .arm_monitor and rng.float(f32) < cfg.arm_block_p) {
        const u = rng.float(f32);
        const arm_mm = if (rng.float(f32) < 0.6667)
            cfg.arm_lo_mm + u * (cfg.arm_mid_mm - cfg.arm_lo_mm)
        else
            cfg.arm_mid_mm + u * (cfg.arm_hi_mm - cfg.arm_mid_mm);
        if (arm_mm < clean_mm) {
            clean_mm = arm_mm;
            cls = .obstacle;
            grazing = false; // arm face is near-normal to the boresight
        }
    }

    var out: TofReading = .{
        .name = s.name,
        .clean_mm = @intFromFloat(@min(clean_mm, 65535.0)),
        .range_mm = 0,
        .valid = false,
        .cls = cls,
    };
    if (clean_mm > 65000.0) return out; // no surface within clamp

    const mm = applyTofNoise(clean_mm, grazing, rd, hit.n, rng, cfg) orelse return out;
    out.range_mm = mm;
    out.valid = true;
    return out;
}

/// Scoped noise model (TOF_SIM_SCOPE.md v0): returns null on no-return.
fn applyTofNoise(clean_mm: f32, grazing: bool, rd: Vec3, n: Vec3, rng: std.Random, cfg: TofConfig) ?u32 {
    var p_drop = cfg.base_dropout;
    const range_m = clean_mm / 1000.0;
    if (grazing) {
        const c = @abs(rd.dot(n));
        p_drop += (1.0 - c / cfg.graze_cos) * (1.0 - cfg.base_dropout);
    }
    const far = range_m / cfg.max_range_m;
    if (far > cfg.far_dropout_start) {
        p_drop += (far - cfg.far_dropout_start) / (1.0 - cfg.far_dropout_start) * cfg.far_dropout_max;
    }
    if (rng.float(f32) < @min(p_drop, 1.0)) return null;
    const sigma = (cfg.sigma_base_mm + cfg.sigma_k_mm_per_m2 * range_m * range_m) * cfg.ambient_factor;
    var noisy = clean_mm + rng.floatNorm(f32) * sigma;
    noisy = @max(cfg.min_range_m * 1000.0, @min(noisy, cfg.max_range_m * 1000.0));
    return @intFromFloat(noisy);
}

/// Per-zone scan grid (tof_scan): the real module is multizone dToF -
/// 54x42 zones max, binned modes in-spec. Pinned (ST product page +
/// Farnell data brief 4759648): FoV 55 x 42 deg (71 diagonal),
/// ranging <5cm - 9m, up to 100 Hz frame rate.
pub const TofScanConfig = struct {
    rows: u32 = 8,
    cols: u32 = 8,
    hfov_deg: f32 = 55.0,
    vfov_deg: f32 = 42.0,
    noise: TofConfig = .{},
};

pub const TofZone = struct {
    range_mm: u32, // 0 when status != 0
    status: u8, // 0 ok, 1 over-range, 2 no-return
    cls: SegClass,
};

/// Scan one sensor zone grid. out must hold rows*cols zones, row-major.
/// Zone (r,c): r=0 is the TOP of the sensor FoV, c=0 is LEFT (boresight).
pub fn scanTof(
    scene: RasterScene,
    s: TofSensor,
    cfg: TofScanConfig,
    body_pos: Vec3,
    body_quat: Quaternion,
    rng: std.Random,
    out: []TofZone,
) void {
    const yaw_q = Quaternion.from_axis_angle(Vec3.init(0, 1, 0), s.azimuth_deg);
    const pitch_q = Quaternion.from_axis_angle(Vec3.init(0, 0, 1), s.elevation_deg);
    const q = body_quat.multiply(yaw_q).multiply(pitch_q);
    const ro = body_pos.add(Vec3.rotate_by_quaternion(s.offset, body_quat));
    const tu = @tan(0.5 * cfg.hfov_deg * std.math.pi / 180.0);
    const tv = @tan(0.5 * cfg.vfov_deg * std.math.pi / 180.0);
    var idx: usize = 0;
    for (0..cfg.rows) |r| {
        const v: f32 = (0.5 - (@as(f32, @floatFromInt(r)) + 0.5) / @as(f32, @floatFromInt(cfg.rows))) * 2.0 * tv;
        for (0..cfg.cols) |c| {
            const u: f32 = ((@as(f32, @floatFromInt(c)) + 0.5) / @as(f32, @floatFromInt(cfg.cols)) - 0.5) * 2.0 * tu;
            const rd = Vec3.rotate_by_quaternion(Vec3.init(1.0, v, u), q).normalize();
            const hit = castRay(scene, ro, rd);
            var clean_mm: f32 = 65535.0;
            var cls: SegClass = .sky;
            var grazing = false;
            if (!std.math.isInf(hit.t)) {
                cls = hit.class;
                if (hit.t <= cfg.noise.max_range_m) {
                    clean_mm = hit.t * 1000.0;
                    grazing = @abs(rd.dot(hit.n)) < cfg.noise.graze_cos;
                }
            }
            if (s.role == .arm_monitor and rng.float(f32) < cfg.noise.arm_block_p) {
                const x = rng.float(f32);
                const arm_mm = if (rng.float(f32) < 0.6667)
                    cfg.noise.arm_lo_mm + x * (cfg.noise.arm_mid_mm - cfg.noise.arm_lo_mm)
                else
                    cfg.noise.arm_mid_mm + x * (cfg.noise.arm_hi_mm - cfg.noise.arm_mid_mm);
                if (arm_mm < clean_mm) {
                    clean_mm = arm_mm;
                    cls = .obstacle;
                    grazing = false;
                }
            }
            if (clean_mm > 65000.0) {
                out[idx] = .{ .range_mm = 0, .status = 1, .cls = cls };
            } else if (applyTofNoise(clean_mm, grazing, rd, hit.n, rng, cfg.noise)) |mm| {
                out[idx] = .{ .range_mm = mm, .status = 0, .cls = cls };
            } else {
                out[idx] = .{ .range_mm = 0, .status = 2, .cls = cls };
            }
            idx += 1;
        }
    }
}
