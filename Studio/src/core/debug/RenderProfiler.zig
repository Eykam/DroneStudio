const std = @import("std");
const build_options = @import("build_options");

pub const level: u8 = if (@hasDecl(build_options, "render_profiler")) build_options.render_profiler else 0;
const Enabled = level != 0;
const TimersEnabled = level >= 2;

pub const Section = enum {
    frame,
    gather_commands,
    material_setup,
    draw_submission,

    pub fn name(self: Section) []const u8 {
        return @tagName(self);
    }
};

pub const FrameStats = struct {
    commands: u32 = 0,
    draws: u32 = 0,
    shader_switches: u32 = 0,
    material_switches: u32 = 0,
    vao_switches: u32 = 0,
    section_cpu_ns: std.enums.EnumArray(Section, u64) = std.enums.EnumArray(Section, u64).initFill(0),
};

/// Accumulated statistics over multiple frames
pub const AveragedStats = struct {
    commands: f32 = 0,
    draws: f32 = 0,
    shader_switches: f32 = 0,
    material_switches: f32 = 0,
    vao_switches: f32 = 0,
    section_cpu_ns: std.enums.EnumArray(Section, f64) = std.enums.EnumArray(Section, f64).initFill(0),
};

/// Represents a single timed span in the call hierarchy
pub const Span = struct {
    section: Section,
    start_ns: u64,
    end_ns: u64,
    parent_index: ?usize,
    depth: u32,

    pub fn duration_ns(self: Span) u64 {
        return self.end_ns - self.start_ns;
    }

    pub fn duration_us(self: Span) f64 {
        return @as(f64, @floatFromInt(self.duration_ns())) / 1000.0;
    }

    pub fn duration_ms(self: Span) f64 {
        return @as(f64, @floatFromInt(self.duration_ns())) / 1_000_000.0;
    }
};

/// Frame data suitable for flamegraph generation
pub const FlameGraphFrame = struct {
    spans: std.ArrayList(Span),
    frame_start_ns: u64,
    frame_end_ns: u64,

    pub fn deinit(self: *FlameGraphFrame) void {
        self.spans.deinit();
    }

    pub fn duration_ms(self: FlameGraphFrame) f64 {
        return @as(f64, @floatFromInt(self.frame_end_ns - self.frame_start_ns)) / 1_000_000.0;
    }
};

pub const Profiler = struct {
    allocator: std.mem.Allocator,
    stats: FrameStats = .{},
    frame_timer: std.time.Timer = undefined,
    last_shader_id: u32 = 0,
    last_material_key: usize = 0,
    last_vao: u32 = 0,

    // Hierarchical timing support
    spans: std.ArrayList(Span),
    active_spans: std.ArrayList(ActiveSpan),
    current_depth: u32 = 0,
    frame_start_ns: u64 = 0,

    // Sampling and averaging
    sample_rate: u32 = 10, // Sample every Nth frame
    snapshot_interval_s: f64 = 1.0, // Update snapshot every second
    frame_count: u64 = 0,
    accumulated_stats: AveragedStats = .{},
    sample_count: u32 = 0,
    last_snapshot_time: f64 = 0,
    snapshot_stats: AveragedStats = .{},
    wall_clock_timer: std.time.Timer = undefined,

    const ActiveSpan = struct {
        section: Section,
        start_ns: u64,
        parent_index: ?usize,
        depth: u32,
    };

    pub fn init(allocator: std.mem.Allocator) Profiler {
        var profiler = Profiler{
            .allocator = allocator,
            .stats = .{},
            .frame_timer = undefined,
            .last_shader_id = 0,
            .last_material_key = 0,
            .last_vao = 0,
            .spans = std.ArrayList(Span).init(allocator),
            .active_spans = std.ArrayList(ActiveSpan).init(allocator),
            .current_depth = 0,
            .frame_start_ns = 0,
            .frame_count = 0,
            .accumulated_stats = .{},
            .sample_count = 0,
            .last_snapshot_time = 0,
            .snapshot_stats = .{},
            .wall_clock_timer = undefined,
        };

        if (TimersEnabled) {
            profiler.frame_timer = std.time.Timer.start() catch unreachable;
            profiler.wall_clock_timer = std.time.Timer.start() catch unreachable;
        }

        return profiler;
    }

    pub fn deinit(self: *Profiler) void {
        self.spans.deinit();
        self.active_spans.deinit();
    }

    pub inline fn beginFrame(self: *Profiler) void {
        if (!Enabled) return;

        self.frame_count += 1;

        // Only sample every Nth frame
        const should_sample = (self.frame_count % self.sample_rate) == 0;
        if (!should_sample) return;

        self.stats = .{};
        self.stats.section_cpu_ns = std.enums.EnumArray(Section, u64).initFill(0);
        self.last_shader_id = 0;
        self.last_material_key = 0;
        self.last_vao = 0;

        if (TimersEnabled) {
            self.frame_timer.reset();
            self.frame_start_ns = self.frame_timer.read();
            self.spans.clearRetainingCapacity();
            self.active_spans.clearRetainingCapacity();
            self.current_depth = 0;
        }
    }

    pub inline fn endFrame(self: *Profiler) void {
        if (!Enabled or !TimersEnabled) return;

        // Only process sampled frames
        const should_sample = (self.frame_count % self.sample_rate) == 0;
        if (!should_sample) return;

        const frame_end_ns = self.frame_timer.read();
        self.stats.section_cpu_ns.set(.frame, frame_end_ns);

        // Accumulate stats for averaging
        self.accumulated_stats.commands += @floatFromInt(self.stats.commands);
        self.accumulated_stats.draws += @floatFromInt(self.stats.draws);
        self.accumulated_stats.shader_switches += @floatFromInt(self.stats.shader_switches);
        self.accumulated_stats.material_switches += @floatFromInt(self.stats.material_switches);
        self.accumulated_stats.vao_switches += @floatFromInt(self.stats.vao_switches);

        for (std.meta.tags(Section)) |section| {
            const current = self.accumulated_stats.section_cpu_ns.get(section);
            const new_val = @as(f64, @floatFromInt(self.stats.section_cpu_ns.get(section)));
            self.accumulated_stats.section_cpu_ns.set(section, current + new_val);
        }

        self.sample_count += 1;

        // Check if it's time to update the snapshot
        const elapsed_s = @as(f64, @floatFromInt(self.wall_clock_timer.read())) / 1_000_000_000.0;
        if (elapsed_s - self.last_snapshot_time >= self.snapshot_interval_s) {
            self.updateSnapshot();
            self.last_snapshot_time = elapsed_s;
        }
    }

    fn updateSnapshot(self: *Profiler) void {
        if (self.sample_count == 0) return;

        const count_f: f32 = @floatFromInt(self.sample_count);
        self.snapshot_stats.commands = self.accumulated_stats.commands / count_f;
        self.snapshot_stats.draws = self.accumulated_stats.draws / count_f;
        self.snapshot_stats.shader_switches = self.accumulated_stats.shader_switches / count_f;
        self.snapshot_stats.material_switches = self.accumulated_stats.material_switches / count_f;
        self.snapshot_stats.vao_switches = self.accumulated_stats.vao_switches / count_f;

        for (std.meta.tags(Section)) |section| {
            const avg = self.accumulated_stats.section_cpu_ns.get(section) / @as(f64, count_f);
            self.snapshot_stats.section_cpu_ns.set(section, avg);
        }

        // Reset accumulators
        self.accumulated_stats = .{};
        self.sample_count = 0;
    }

    pub inline fn trackCommand(self: *Profiler) void {
        if (!Enabled) return;
        self.stats.commands += 1;
    }

    pub inline fn trackDraw(self: *Profiler) void {
        if (!Enabled) return;
        self.stats.draws += 1;
    }

    pub inline fn trackShaderBind(self: *Profiler, program_id: u32) void {
        if (!Enabled) return;
        if (self.last_shader_id != program_id) {
            self.last_shader_id = program_id;
            self.stats.shader_switches += 1;
        }
    }

    pub inline fn trackMaterialBind(self: *Profiler, material_key: usize) void {
        if (!Enabled) return;
        if (self.last_material_key != material_key) {
            self.last_material_key = material_key;
            self.stats.material_switches += 1;
        }
    }

    pub inline fn trackVaoBind(self: *Profiler, vao: u32) void {
        if (!Enabled) return;
        if (self.last_vao != vao) {
            self.last_vao = vao;
            self.stats.vao_switches += 1;
        }
    }

    pub inline fn snapshot(self: *Profiler) ?AveragedStats {
        if (!Enabled) return null;
        return self.snapshot_stats;
    }

    pub inline fn sectionScope(self: *Profiler, section: Section) SectionScope {
        return SectionScope.start(self, section);
    }

    /// Begin a hierarchical span (internal use by SectionScope)
    inline fn beginSpan(self: *Profiler, section: Section) void {
        if (!TimersEnabled) return;

        const now = self.frame_timer.read();
        const parent_index = if (self.active_spans.items.len > 0)
            self.spans.items.len - 1
        else
            null;

        self.active_spans.append(ActiveSpan{
            .section = section,
            .start_ns = now,
            .parent_index = parent_index,
            .depth = self.current_depth,
        }) catch return;

        self.current_depth += 1;
    }

    /// End a hierarchical span (internal use by SectionScope)
    inline fn endSpan(self: *Profiler, section: Section) void {
        if (!TimersEnabled) return;
        if (self.active_spans.items.len == 0) return;

        const now = self.frame_timer.read();
        const active = self.active_spans.pop();

        // Verify we're ending the correct span
        if (active.section != section) {
            std.debug.print("WARNING: Profiler span mismatch! Expected {s}, got {s}\n", .{
                @tagName(section),
                @tagName(active.section),
            });
            return;
        }

        self.spans.append(Span{
            .section = active.section,
            .start_ns = active.start_ns,
            .end_ns = now,
            .parent_index = active.parent_index,
            .depth = active.depth,
        }) catch return;

        if (self.current_depth > 0) {
            self.current_depth -= 1;
        }
    }

    /// Get flamegraph data for the current frame
    pub fn getFlameGraphFrame(self: *Profiler) ?FlameGraphFrame {
        if (!TimersEnabled) return null;

        return FlameGraphFrame{
            .spans = self.spans.clone() catch return null,
            .frame_start_ns = self.frame_start_ns,
            .frame_end_ns = self.frame_timer.read(),
        };
    }

    /// Print hierarchical timing report to stderr
    pub fn printHierarchical(self: *Profiler) void {
        if (!TimersEnabled) return;

        std.debug.print("\n=== Render Frame Profile ===\n", .{});
        std.debug.print("Total spans: {d}\n", .{self.spans.items.len});

        for (self.spans.items, 0..) |span, i| {
            const indent = "  " ** span.depth;
            std.debug.print("{d:4} {s}{s}: {d:.3}ms (depth={d}, parent={?d})\n", .{
                i,
                indent,
                span.section.name(),
                span.duration_ms(),
                span.depth,
                span.parent_index,
            });
        }
        std.debug.print("===========================\n\n", .{});
    }
};

pub const SectionScope = struct {
    profiler: *Profiler = undefined,
    section: Section = undefined,
    timer: std.time.Timer = undefined,
    active: bool = false,

    inline fn start(profiler: *Profiler, section: Section) SectionScope {
        if (!Enabled or !TimersEnabled) {
            return SectionScope{};
        }

        var scope = SectionScope{};
        scope.profiler = profiler;
        scope.section = section;
        scope.timer = std.time.Timer.start() catch unreachable;
        scope.active = true;

        // Start hierarchical span tracking
        profiler.beginSpan(section);

        return scope;
    }

    pub inline fn end(self: *SectionScope) void {
        if (!self.active) return;

        const elapsed = self.timer.read();
        const prev = self.profiler.stats.section_cpu_ns.get(self.section);
        self.profiler.stats.section_cpu_ns.set(self.section, prev + elapsed);

        // End hierarchical span tracking
        self.profiler.endSpan(self.section);

        self.active = false;
    }
};

pub inline fn enabled() bool {
    return Enabled;
}

pub inline fn timersEnabled() bool {
    return TimersEnabled;
}
