const std = @import("std");
const vr = @import("vision_raster.zig");
const Math = @import("core/Math.zig");
const Vec3 = Math.Vec3;
const Quaternion = Math.Quaternion;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    const W = 64;
    const H = 48;
    const obs = [_]vr.RasterObstacle{
        .{ .center = Vec3.init(8, 1.0, 2.0), .radius = 1.5 },
        .{ .center = Vec3.init(12, 0.8, -3.0), .radius = 1.0 },
    };
    const scene = vr.RasterScene{
        .ground_y = 0.05,
        .obstacles = &obs,
        .goal = Vec3.init(6, 0, 0),
        .goal_radius = 1.0,
    };
    const cam = vr.Camera{ .width = W, .height = H, .mount_pos = Vec3.zero() };
    const depth = try alloc.alloc(f32, W * H);
    const seg = try alloc.alloc(u8, W * H);
    // drone at origin, 2m up, identity orientation (nose +X)
    vr.render(scene, cam, Vec3.init(0, 2, 0), Quaternion.identity(), .{ .depth = depth, .seg = seg, .width = W, .height = H });
    const chars = " .#@"; // sky, floor, obstacle, goal
    var tbuf: [8]f32 = undefined;
    _ = &tbuf;
    for (0..H) |r| {
        for (0..W) |c| {
            const s = seg[r * W + c];
            std.debug.print("{c}", .{chars[@min(s, 3)]});
        }
        std.debug.print("\n", .{});
    }
    // depth stats
    var n_sky: u32 = 0;
    var dmin: f32 = 1e9;
    var dmax: f32 = 0;
    for (depth, 0..) |d, i| {
        if (seg[i] == 0) {
            n_sky += 1;
        } else {
            dmin = @min(dmin, d);
            dmax = @max(dmax, d);
        }
    }
    std.debug.print("sky={d}/{d} depth_range=[{d:.2},{d:.2}]\n", .{ n_sky, W * H, dmin, dmax });
}
