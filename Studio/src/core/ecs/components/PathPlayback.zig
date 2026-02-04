// src/ecs/components/PathPlayback.zig
const std = @import("std");
const Math = @import("../../Math.zig");
const Core = @import("../Core.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const Transform = @import("./Transform.zig");
const PathSystem = @import("./PathSystem.zig");

const Vec3 = Math.Vec3;
const Quaternion = Math.Quaternion;

const ECSManager = @import("../ECSManager.zig");

pub const PathPlaybackComponent = struct {
    const Self = @This();

    // Entity this component is attached to
    entity_id: Core.EntityID = Core.EntityID.init(0),

    // Reference to the path being played (index into PathSystem.paths)
    path_index: ?usize = null,

    // Playback state
    is_playing: bool = false,
    playback_time: f64 = 0.0, // Current time in seconds along the path
    playback_speed: f32 = 1.0, // Speed multiplier (0.1x - 10x)
    loop: bool = false,

    // Frame stepping mode
    step_mode: bool = false,
    step_requested: bool = false,
    step_direction: i8 = 1, // 1 = forward, -1 = backward

    // Time step for frame-by-frame mode (30 fps equivalent)
    frame_time: f32 = 1.0 / 30.0,

    // Track if we've reached the end
    at_end: bool = false,

    // Track if path playback is actively controlling this entity's transform
    // (used by physics system to skip writeback)
    active: bool = false,

    pub fn play(self: *Self) void {
        if (self.path_index == null) return;
        self.is_playing = true;
        self.at_end = false;
    }

    pub fn pause(self: *Self) void {
        self.is_playing = false;
    }

    pub fn stop(self: *Self) void {
        self.is_playing = false;
        self.playback_time = 0.0;
        self.at_end = false;
    }

    pub fn setSpeed(self: *Self, speed: f32) void {
        self.playback_speed = std.math.clamp(speed, 0.1, 10.0);
    }

    pub fn stepForward(self: *Self) void {
        if (self.step_mode) {
            self.step_requested = true;
            self.step_direction = 1;
        }
    }

    pub fn stepBackward(self: *Self) void {
        if (self.step_mode) {
            self.step_requested = true;
            self.step_direction = -1;
        }
    }

    pub fn seek(self: *Self, time: f64) void {
        self.playback_time = @max(0.0, time);
        self.at_end = false;
    }

    pub fn selectPath(self: *Self, index: usize) void {
        self.path_index = index;
        self.playback_time = 0.0;
        self.is_playing = false;
        self.at_end = false;
        self.active = true; // Path playback now controls this entity
    }

    pub fn clearPath(self: *Self) void {
        self.path_index = null;
        self.playback_time = 0.0;
        self.is_playing = false;
        self.at_end = false;
        self.active = false; // Release control back to physics
    }

    pub fn attach(self: *Self, ecs: *ECSManager, eid: Core.EntityID) !void {
        self.entity_id = eid;
        try ecs.path_playback_components.add(eid, self.*);
    }
};

pub const PathPlaybackSystem = struct {
    const Self = @This();

    path_playback_components: *SparseSet(PathPlaybackComponent),
    transform_components: *SparseSet(Transform.TransformComponent),
    path_system: ?*PathSystem,

    pub fn init(
        path_playback_components: *SparseSet(PathPlaybackComponent),
        transform_components: *SparseSet(Transform.TransformComponent),
    ) Self {
        return .{
            .path_playback_components = path_playback_components,
            .transform_components = transform_components,
            .path_system = null,
        };
    }

    pub fn setPathSystem(self: *Self, path_system: *PathSystem) void {
        self.path_system = path_system;
    }

    pub fn update(self: *Self, dt: f64) void {
        const path_sys = self.path_system orelse return;

        var it = self.path_playback_components.iterator();
        while (it.next()) |entry| {
            const playback = entry.component;
            const eid = entry.entity_id;

            // Skip if no path selected (not active)
            const path_index = playback.path_index orelse {
                playback.active = false;
                continue;
            };
            const path = path_sys.getPath(path_index) orelse {
                playback.active = false;
                continue;
            };

            // Mark as active - we're controlling this entity's transform
            playback.active = true;

            // Get path duration
            const duration: f64 = @floatCast(path.duration());
            if (duration <= 0.0) continue;

            // Update playback time ONLY when playing or stepping
            if (playback.step_mode and playback.step_requested) {
                // Frame-by-frame stepping
                const step_delta: f64 = @as(f64, @floatCast(playback.frame_time)) *
                    @as(f64, @floatFromInt(playback.step_direction));
                playback.playback_time += step_delta;
                playback.step_requested = false;
            } else if (playback.is_playing) {
                // Normal playback - advance time
                playback.playback_time += dt * @as(f64, @floatCast(playback.playback_speed));
            }
            // When paused: don't update playback_time, but still set transform below

            // Handle loop/end
            if (playback.playback_time >= duration) {
                if (playback.loop) {
                    playback.playback_time = @mod(playback.playback_time, duration);
                } else {
                    playback.playback_time = duration;
                    playback.is_playing = false;
                    playback.at_end = true;
                }
            } else if (playback.playback_time < 0.0) {
                if (playback.loop) {
                    playback.playback_time = duration + @mod(playback.playback_time, duration);
                } else {
                    playback.playback_time = 0.0;
                }
            }

            playback.at_end = (playback.playback_time >= duration);

            // Interpolate position and orientation from path
            const t_norm: f32 = if (duration > 0.0)
                @as(f32, @floatCast(playback.playback_time / duration))
            else
                0.0;

            const position = path.evalPos(t_norm);
            const orientation = path.evalOrientation(t_norm);

            // ALWAYS set entity transform when a path is active (even when paused)
            // This prevents physics writeback from overriding our position
            if (self.transform_components.get(eid)) |transform| {
                transform.position = .{ position.x(), position.y(), position.z() };
                // Normalize quaternion for smooth interpolation
                transform.rotation = orientation.normalize();
                transform.markDirty();
            }
        }
    }
};
