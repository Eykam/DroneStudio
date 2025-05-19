// src/ecs/components/Viewport.zig
const std = @import("std");
const gl = @import("../../bindings/gl.zig");
const Core = @import("../Core.zig");
const OpenGL = @import("../graphics/OpenGL.zig");
const CudaGL = @import("../graphics/CudaGL.zig");
const ECSManager = @import("../ECSManager.zig");
const Globals = @import("../components/Globals.zig");
const Transform = @import("../components/Transform.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;

const glad = gl.glad;
const Viewport = OpenGL.Viewport;

pub const ViewportComponent = struct {
    const Self = @This();

    vp: Viewport,
    resizable: bool = false,
    shared: bool = false,
    shared_info: ?CudaGL.CUDAGLTexture = null,
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

    pub fn enableSharing(self: *Self) void {
        // Create shared CUDA-GL texture
        if (!self.shared or self.shared_info == null) {
            self.shared_info = CudaGL.createCUDAGLTexture(
                self.vp.fbo.width,
                self.vp.fbo.height,
                self.vp.fbo.texture,
            );
            self.shared = true;
        }
    }

    pub fn attach(self: *Self, ecs: *ECSManager, eid: Core.EntityID) !void {
        try ecs.viewport_components.add(eid, self.*);
    }

    pub fn deinit(self: *Self) void {
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
            var vc = tuple.component;
            var vp = &vc.vp;

            if (vc.resizable and (vp.fbo.width != w or vp.fbo.height != h)) {
                std.debug.print("Resizing: {s}\n", .{vp.name});

                const old_name = try self.allocator.dupeZ(u8, vp.name);
                defer self.allocator.free(old_name);

                // Cleanup CUDA resources first if shared
                if (vc.shared and vc.shared_info != null) {
                    vc.shared_info.?.deinit();
                    vc.shared_info = null;
                }

                vp.deinit(); // old GL objects
                vp.* = Viewport.init( // recreate
                    self.allocator, old_name, w, h) catch {
                    std.debug.print("failed to resize viewport\n", .{});
                    continue;
                };

                glad.glFlush();
                glad.glFinish();

                if (vc.shared) {
                    vc.enableSharing();
                }
            }
        }
    }
};
