// src/ecs/components/Camera.zig
const std = @import("std");
const Core = @import("../Core.zig");
const Math = @import("../../Math.zig");
const Globals = @import("Globals.zig");
const Transform = @import("Transform.zig");
const ECSManager = @import("../ECSManager.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;

const Mat4 = Math.Mat4;
const Vec3 = Math.Vec3;

pub const CameraComponent = struct {
    const Self = @This();

    entity_id: Core.EntityID,

    // projection params
    fov: f32 = 45.0,
    near_plane: f32 = 0.1,
    far_plane: f32 = 100.0,
    aspect: f32 = 16.0 / 9.0,

    active: bool = false,

    pub fn view(self: *const Self, transform: *const Transform.TransformComponent) Mat4 {
        _ = self;
        const position = transform.world_transform.get_position();
        const front = transform.world_transform.get_forward();
        const up = transform.world_transform.get_up();
        return Mat4.look_at(position, position.add(front), up);
    }

    pub fn projection(self: *const Self) Mat4 {
        return Mat4.perspective(self.fov, self.aspect, self.near_plane, self.far_plane);
    }

    pub fn attach(self: *CameraComponent, ecs: *ECSManager, eid: Core.EntityID) !void {
        self.entity_id = eid;

        if (ecs.camera_system.active_camera_eid == null) {
            ecs.camera_system.active_camera_eid = eid;
        }

        try ecs.camera_components.add(eid, self.*);
    }
};

pub const CameraSystem = struct {
    const Self = @This();

    world: *Core.World,
    cameras: *SparseSet(CameraComponent),
    transforms: *SparseSet(Transform.TransformComponent),
    active_camera_eid: ?Core.EntityID = null,

    pub fn init(
        world: *Core.World,
        cams: *SparseSet(CameraComponent),
        transforms: *SparseSet(Transform.TransformComponent),
    ) Self {
        return .{ .world = world, .cameras = cams, .transforms = transforms };
    }

    pub fn set_active(self: *Self, eid: Core.EntityID) void {
        if (self.cameras.has(eid)) {
            // mark old one inactive
            if (self.active_camera_eid) |active_eid| {
                if (self.cameras.get(active_eid)) |active_camera| active_camera.active = false;
            }
            self.active_camera_eid = eid;
            self.cameras.get(eid).?.active = true;
        }
    }

    /// Called once per frame **before** RenderSystem.update()
    pub fn update(self: *Self, globals: *Globals.GlobalsComponent) void {
        if (self.active_camera_eid) |eid| {
            const camera = self.cameras.get(eid).?;
            const transform = self.transforms.get(eid).?;
            globals.view = camera.view(transform);
            globals.proj = camera.projection();
        }
    }
};
