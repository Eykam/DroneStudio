const std = @import("std");
const c = @import("../../bindings/c.zig");
const gl = @import("../../bindings/gl.zig");
const CudaGL = @import("../graphics/CudaGL.zig");
const Globals = @import("Globals.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const Viewports = @import("Viewports.zig");
const ViewportComponent = Viewports.ViewportComponent;
const GlobalsComponent = Globals.GlobalsComponent;
const cuda = c.cuda;
const glad = gl.glad;

const SHM_ENV = "drone_shm";
const MAX_VIEWPORTS = 8;

pub const ViewportEntry = extern struct {
    name: [32]u8,
    ipc: cuda.cudaIpcMemHandle_t, // 64 B
    width: u32,
    height: u32,
    pitch: u32, // bytes per row
    _pad: [19]u8, // 19 bytes to reach 128 total

    comptime {
        // Compile-time check to ensure size is exactly 64 bytes
        std.debug.assert(@sizeOf(@This()) == 128);
    }
};

pub const SharedHeader = extern struct {
    seq: u64, // heartbeat
    ack: u64, // from Python
    count: u32, // number of valid entries
    _pad0: u32,
    entries: [MAX_VIEWPORTS]ViewportEntry,
};

pub const SharedMemSystem = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    globals: *GlobalsComponent,
    viewports: *SparseSet(ViewportComponent),
    mem_region: []align(std.mem.page_size) u8, // 16‑byte struct we mapped
    seq: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, globals: *GlobalsComponent, viewports: *SparseSet(ViewportComponent)) !Self {
        return .{
            .allocator = allocator,
            .globals = globals,
            .viewports = viewports,
            .mem_region = try initSharedMem(allocator), // create + write $DRONE_SHM
        };
    }

    pub fn deinit(self: *Self) void {
        // Safe even if Python is still mapped – segment stays until its fd closes.
        std.debug.print("Cleaning up Shared mem...\n", .{});
        const shm_name = getShmName(self.allocator) catch @panic("Failed to get SHM_NAME, shared mem file will need to be manually deleted");
        defer self.allocator.free(shm_name);

        _ = std.c.shm_unlink(shm_name.ptr);
        std.posix.munmap(self.mem_region);
    }

    pub fn update(self: *Self) void {
        const hdr: *SharedHeader = @ptrCast(self.mem_region.ptr);

        var idx: usize = 0;
        var it = self.viewports.iterator();
        while (it.next()) |tuple| {
            const vc = tuple.component;
            if (!vc.shared or idx >= MAX_VIEWPORTS) continue;

            // CudaGL.inspectFBO(vc.vp.fbo.fbo, vc.vp.fbo.width, vc.vp.fbo.height,);
            vc.shared_info.?.copyFromGL();
            // CudaGL.inspectCUDABuffer(vc.shared_info.?.gpu_ptr, vc.shared_info.?.width, vc.shared_info.?.height, vc.shared_info.?.pitch,);
            const e = &hdr.entries[idx];
            @memcpy(&e.name, vc.vp.name.ptr);

            // copy FD and dimensions
            if (vc.shared_info) |info| {
                e.ipc = info.ipc;
                e.width = @intCast(info.width);
                e.height = @intCast(info.height);
                e.pitch = @intCast(info.pitch);
            }

            idx += 1;
        }
        hdr.count = @intCast(idx);

        // heartbeat
        @atomicStore(u64, &hdr.seq, self.seq, .release);
        self.seq += 1;
    }
};

fn getShmName(allocator: std.mem.Allocator) ![:0]const u8 {
    return allocator.dupeZ(u8, SHM_ENV); // Allocates & appends '\x00'.
}

fn initSharedMem(allocator: std.mem.Allocator) ![]align(std.mem.page_size) u8 {
    const shm_name = try getShmName(allocator);
    defer allocator.free(shm_name);

    const expected_size = @sizeOf(SharedHeader);
    std.debug.print("Shared memory size: {} bytes\n", .{expected_size});

    const fd = std.c.shm_open(
        shm_name.ptr,
        @bitCast(std.os.linux.O{ .ACCMODE = .RDWR, .CREAT = true }),
        0o660,
    );
    try std.posix.ftruncate(fd, @sizeOf(SharedHeader));
    const ptr = try std.posix.mmap(
        null,
        @sizeOf(SharedHeader),
        std.posix.PROT.READ | std.posix.PROT.WRITE,
        .{ .TYPE = .SHARED },
        fd,
        0,
    );
    _ = std.posix.close(fd);

    return ptr;
}
