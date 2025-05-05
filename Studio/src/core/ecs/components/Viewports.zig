// src/ecs/components/Viewport.zig
const std = @import("std");
const Core = @import("../Core.zig");
const OpenGL = @import("../graphics/OpenGL.zig");
const ECSManager = @import("../ECSManager.zig");
const Globals = @import("../components/Globals.zig");
const Transform = @import("../components/Transform.zig");

const SparseSet = @import("../SparseSet.zig").SparseSet;
const Viewport = OpenGL.Viewport;

pub const ViewportComponent = struct {
    const Self = @This();

    vp: Viewport,
    active: bool = true,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, width: u32, height: u32) !Self {
        return .{
            .vp = try Viewport.init(
                allocator,
                name,
                @intCast(width),
                @intCast(height),
            ),
        };
    }

    pub fn attach(self: *ViewportComponent, ecs: *ECSManager, eid: Core.EntityID) !void {
        try ecs.viewport_components.add(eid, self.*);
    }

    pub fn deinit(self: *ViewportComponent) void {
        self.vp.deinit();
    }
};

pub const ViewportSystem = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    globals: *Globals.GlobalsComponent,
    viewports: *SparseSet(ViewportComponent),

    pub fn init(
        allocator: std.mem.Allocator,
        globals: *Globals.GlobalsComponent,
        vps: *SparseSet(ViewportComponent),
    ) Self {
        return .{ .allocator = allocator, .globals = globals, .viewports = vps };
    }

    /// if window size changed: resize all FBOs
    pub fn update(self: *Self) !void {
        const w: i32 = @intCast(self.globals.scene_width);
        const h: i32 = @intCast(self.globals.scene_height);

        if (w <= 0 or h <= 0) {
            std.debug.print("Invalid viewport dimensions: {d}x{d}\n", .{ w, h });
            return;
        }

        var it = self.viewports.iterator();
        while (it.next()) |tuple| {
            var vp = &tuple.component.vp;

            if (vp.fbo.width != w or vp.fbo.height != h) {
                const old_name = try self.allocator.dupeZ(u8, vp.name);
                defer self.allocator.free(old_name);

                vp.deinit(); // old GL objects
                vp.* = Viewport.init( // recreate
                    self.allocator, old_name, w, h) catch {
                    std.debug.print("failed to resize viewport\n", .{});
                    continue;
                };
            }
        }
    }
};
