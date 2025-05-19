// src/ecs/systems/RecorderSystem.zig
const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");

const os = std.os;
const Quaternion = Math.Quaternion;
const ECSManager = @import("../ECSManager.zig");

pub const PackedTransform = extern struct {
    t: f32, // absolute seconds from Global.time
    eid: u32, // Core.EntityID.id
    pos: [3]f32,
    rot: [4]f32, // quaternion
    scale: [3]f32,
    _pad: u32 = 0, // 16‑byte align → 64 B total
};

pub const RecorderSystem = struct {
    const Self = @This();

    pub var default_max_secs: u32 = 600;
    pub var default_max_entities_per_frame: u32 = 256; // mmap bookkeeping

    allocator: std.mem.Allocator,

    is_recording: bool = false,
    is_playback: bool = false,
    duration: f32 = 0,
    bytes_written: usize = 0,

    // mmap bookkeeping
    file: ?std.fs.File = null,
    base_ptr: ?[]align(std.mem.page_size) u8 = null,
    map_len: usize = 0,

    play_file: ?std.fs.File = null,
    play_map: ?[]align(std.mem.page_size) u8 = null,
    play_len: usize = 0,

    // sparse seek index
    index_stride_pkts: usize = 512,
    pkts_since_index: usize = 0,
    index: std.ArrayList(IndexEntry),

    pub const IndexEntry = struct { t: f32, offset: u64 };

    pub fn init(alloc: std.mem.Allocator) !*Self {
        const self = try alloc.create(Self);
        self.* = .{
            .allocator = alloc,
            .index = std.ArrayList(IndexEntry).init(alloc),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        _ = self.stop() catch {};
        self.index.deinit();
        self.allocator.destroy(self);
    }

    /// Toggle recording on/off using the defaults above
    pub fn toggle(self: *Self) !void {
        if (self.is_recording)
            try self.stop()
        else
            try self.start(default_max_secs, default_max_entities_per_frame);
    }

    /// Explicit start with caps
    pub fn start(self: *Self, max_secs: u32, max_entities_per_frame: u32) !void {
        if (self.is_recording) return; // already on

        std.debug.print("Starting Recording...\n", .{});

        // ── size estimate ──
        const est_pkts = @as(usize, max_secs) * 60 * @as(usize, max_entities_per_frame);
        const est_bytes = est_pkts * @sizeOf(PackedTransform);

        const pg = if (@hasDecl(std.os, "getPageSize"))
            std.os.getPageSize()
        else
            std.mem.page_size;

        self.map_len = ((est_bytes + pg - 1) / pg) * pg;

        // ── file + map ──
        self.file = try std.fs.cwd().createFile("capture.tmp", .{
            .read = true,
            .truncate = true,
            .exclusive = false,
        });
        try self.file.?.setEndPos(self.map_len);

        self.base_ptr = try std.posix.mmap(
            null,
            self.map_len,
            std.posix.PROT.READ | std.posix.PROT.WRITE,
            .{ .TYPE = .SHARED },
            self.file.?.handle,
            0,
        );

        // reset counters
        self.duration = 0;
        self.bytes_written = 0;
        self.pkts_since_index = 0;
        self.index.clearRetainingCapacity();
        self.is_recording = true;
    }

    /// Stop recording, flush, close file.  Leaves `duration` intact.
    pub fn stop(self: *Self) !void {
        if (!self.is_recording) return;

        std.debug.print("Stopping Recording...\n", .{});

        try self.file.?.setEndPos(self.bytes_written); // shrink to real size
        _ = std.posix.munmap(self.base_ptr.?);
        self.file.?.close();

        self.file = null;
        self.base_ptr = null;
        self.map_len = 0;
        self.is_recording = false;
    }

    /// Discard data & reset counters (also stops if active).
    pub fn reset(self: *Self) !void {
        try self.stop();
        self.duration = 0;
        self.bytes_written = 0;
    }

    /// Call once per ECS frame AFTER all transforms are updated
    pub fn update(self: *Self, ecs: *ECSManager) !void {
        if (!self.is_recording) return;

        self.duration += @as(f32, @floatCast(ecs.globals.dt));

        var it = ecs.transform_components.iterator();
        while (it.next()) |entry| {
            var tf = entry.component;
            if (!tf.changed_this_frame) continue;

            var pkt = PackedTransform{
                .t = self.duration,
                .eid = @intCast(entry.entity_id.id),
                .pos = tf.position,
                .rot = .{ tf.rotation.x(), tf.rotation.y(), tf.rotation.z(), tf.rotation.w() },
                .scale = tf.scale,
            };

            std.mem.copyForwards(
                u8,
                self.base_ptr.?[self.bytes_written .. self.bytes_written + @sizeOf(PackedTransform)],
                std.mem.asBytes(&pkt),
            );
            self.bytes_written += @sizeOf(PackedTransform);
            tf.changed_this_frame = false;

            // sparse index
            self.pkts_since_index += 1;
            if (self.pkts_since_index >= self.index_stride_pkts) {
                try self.index.append(.{ .t = self.duration, .offset = @intCast(self.bytes_written) });
                self.pkts_since_index = 0;
            }
        }
    }

    pub fn seek(self: *Self, ecs: *ECSManager, target_time: f32) !void {
        if (self.is_recording)
            return error.CannotSeekWhileRecording;

        std.debug.print("Seeking {d}...\n", .{target_time});
        try self.ensurePlaybackMapped();

        // find starting offset via sparse index
        var offset: usize = 0;
        if (self.index.items.len > 0) {
            var lo: usize = 0;
            var hi: usize = self.index.items.len - 1;
            while (lo <= hi) {
                const mid = (lo + hi) / 2;
                const entry = self.index.items[mid];
                if (entry.t <= target_time) {
                    offset = @intCast(entry.offset);
                    lo = mid + 1;
                } else {
                    if (mid == 0) break;
                    hi = mid - 1;
                }
            }
        }

        const pkt_size = @sizeOf(PackedTransform);
        const slice = self.play_map.?; // []align(4096)u8

        var off: usize = offset; // byte offset into the slice
        while (off + pkt_size <= slice.len) : (off += pkt_size) {
            const pkt = std.mem.bytesAsValue(PackedTransform, slice[off .. off + pkt_size]);
            if (pkt.*.t > target_time) break;

            const eid = Core.EntityID{ .id = pkt.*.eid };
            if (ecs.transform_components.get(eid)) |tf| {
                tf.position = pkt.*.pos;
                tf.rotation = Quaternion.init(pkt.*.rot[0], pkt.*.rot[1], pkt.*.rot[2], pkt.*.rot[3]);
                tf.scale = pkt.*.scale;
                tf.updateLocalTransform();
            }
        }
    }

    /// Ensure the file is mmap’d read‑only for playback.
    fn ensurePlaybackMapped(self: *Self) !void {
        if (self.play_map != null) return; // already mapped

        self.play_file = try std.fs.cwd().openFile("capture.tmp", .{ .mode = .read_only });
        self.play_len = try self.play_file.?.getEndPos();

        self.play_map = try std.posix.mmap(
            null,
            self.play_len,
            std.posix.PROT.READ,
            .{ .TYPE = .SHARED },
            self.play_file.?.handle,
            0,
        );
    }

    pub fn saveToDisk(self: *Self, path: []const u8) !void {
        if (self.is_recording) try self.toggle();
        _ = self.file.?;
        try std.fs.cwd().rename("capture.tmp", path);
    }

    pub fn loadFromDisk(self: *Self, path: []const u8) !void {
        if (self.is_recording) try self.stop();
        if (self.play_map != null) {
            _ = std.posix.munmap(self.play_map.?);
            self.play_file.?.close();
            self.play_map = null;
            self.play_file = null;
        }

        // Open & mmap read-only
        self.play_file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        self.play_len = try self.play_file.?.getEndPos();
        self.play_map = try std.posix.mmap(
            null,
            self.play_len,
            std.posix.PROT.READ,
            .{ .TYPE = .SHARED },
            self.play_file.?.handle,
            0,
        );
        // Clear out any old sparse index
        self.index.clearRetainingCapacity();

        // Walk through every PackedTransform in the file to rebuild index + find duration
        const pkt_size = @sizeOf(PackedTransform);
        var off: usize = 0;
        var pkt_counter: usize = 0;
        var last_t: f32 = 0;

        while (off + pkt_size <= self.play_len) : (off += pkt_size) {
            // Interpret the bytes as a PackedTransform
            const pkt = std.mem.bytesAsValue(PackedTransform, self.play_map.?[off .. off + pkt_size]);
            last_t = pkt.t;

            // Every index_stride_pkts packets, record a sparse index entry
            if (pkt_counter % self.index_stride_pkts == 0) {
                try self.index.append(.{
                    .t = pkt.t,
                    .offset = @intCast(off),
                });
            }

            pkt_counter += 1;
        }

        self.duration = last_t;
        std.debug.print("Detected Duration: {d}\n", .{self.duration});
    }

    /// Return rough MB written so UI can display “file size …”
    pub fn getMegabytes(self: *Self) f32 {
        return @as(f32, @floatFromInt(self.bytes_written)) / 1_048_576.0;
    }
};
