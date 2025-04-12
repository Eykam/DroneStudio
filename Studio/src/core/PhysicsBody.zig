// PhysicsBody.zig
const std = @import("std");
const Node = @import("Node.zig");
const Math = @import("Math.zig");

const Vec3 = Math.Vec3;
const Mat4 = Math.Mat4;
const Quaternion = Math.Quaternion;

pub const PhysicsBodyType = enum {
    Static,
    Dynamic,
    Kinematic,
};

const Self = @This();

// Base physics body properties
node: *Node, // Reference to associated node
allocator: std.mem.Allocator,
body_type: PhysicsBodyType = .Static,
mass: f32 = 1.0,
inertia_tensor: [3][3]f32 = undefined,
velocity: Vec3 = Vec3.new(0, 0, 0),
angular_velocity: Vec3 = Vec3.new(0, 0, 0),
force_accumulator: Vec3 = Vec3.new(0, 0, 0),
torque_accumulator: Vec3 = Vec3.new(0, 0, 0),
linear_damping: f32 = 0.01,
angular_damping: f32 = 0.05,
restitution: f32 = 0.3, // Bounciness
friction: f32 = 0.5,
enabled: bool = true,

// Component interface
update_fn: ?*const fn (body: *Self, dt: f32) void = null,
apply_forces_fn: ?*const fn (body: *Self) void = null,

pub fn init(allocator: std.mem.Allocator, node: *Node, body_type: PhysicsBodyType, mass: f32) !*Self {
    const body = try allocator.create(Self);

    body.* = Self{
        .allocator = allocator,
        .node = node,
        .body_type = body_type,
        .mass = mass,
        .inertia_tensor = calculateInertia(mass),
    };

    return body;
}

pub fn deinit(self: *Self) void {
    self.allocator.destroy(self);
}

// Standard physics update - can be overridden by custom components
pub fn update(self: *Self, dt: f32) void {
    if (!self.enabled or self.body_type == .Static) return;

    // First call any custom force application function
    if (self.apply_forces_fn) |apply_fn| {
        apply_fn(self);
    }

    // Then call any custom update function
    if (self.update_fn) |update_fn| {
        update_fn(self);
        return; // Custom update handles everything
    }

    // Otherwise, perform standard physics update
    self.standardUpdate(dt);
}

fn standardUpdate(self: *Self, dt: f32) void {
    // Apply accumulated forces
    const acceleration = self.force_accumulator.scale(1.0 / self.mass);
    self.velocity = self.velocity.add(acceleration.scale(dt));

    // Apply damping
    self.velocity = self.velocity.scale(1.0 - self.linear_damping);

    // Update position
    const position_delta = self.velocity.scale(dt);
    self.node.position[0] += position_delta.x();
    self.node.position[1] += position_delta.y();
    self.node.position[2] += position_delta.z();

    // Apply torque
    const angular_acceleration = self.applyInverseInertiaTensor(self.torque_accumulator);
    self.angular_velocity = self.angular_velocity.add(angular_acceleration.scale(dt));

    // Apply angular damping
    self.angular_velocity = self.angular_velocity.scale(1.0 - self.angular_damping);

    // Update orientation
    if (self.angular_velocity.length() > 0.0001) {
        const rotation_delta = Quaternion.from_axis_angle(self.angular_velocity.normalize(), self.angular_velocity.length() * dt);
        self.node.rotation = rotation_delta.multiply(self.node.rotation).normalize();
    }

    // Reset force and torque accumulators
    self.force_accumulator = Vec3.new(0, 0, 0);
    self.torque_accumulator = Vec3.new(0, 0, 0);

    // Update node transform
    self.node.updateLocalTransform();
}

pub fn applyForce(self: *Self, force: Vec3) void {
    if (self.body_type == .Static) return;
    self.force_accumulator = self.force_accumulator.add(force);
}

pub fn applyForceAtPoint(self: *Self, force: Vec3, point: Vec3) void {
    if (self.body_type == .Static) return;

    self.force_accumulator = self.force_accumulator.add(force);

    // Calculate torque: τ = r × F
    const center = Vec3.new(self.node.position[0], self.node.position[1], self.node.position[2]);
    const offset = point.subtract(center);
    const torque = offset.cross(force);

    self.torque_accumulator = self.torque_accumulator.add(torque);
}

pub fn applyTorque(self: *Self, torque: Vec3) void {
    if (self.body_type == .Static) return;
    self.torque_accumulator = self.torque_accumulator.add(torque);
}

fn applyInverseInertiaTensor(self: *Self, torque: Vec3) Vec3 {
    // Simplified - assumes diagonal inertia tensor
    return Vec3.new(torque.x() / self.inertia_tensor[0][0], torque.y() / self.inertia_tensor[1][1], torque.z() / self.inertia_tensor[2][2]);
}

fn calculateInertia(mass: f32) [3][3]f32 {
    // Default inertia for a box
    const size = Vec3.new(0.3, 0.1, 0.3);

    const ixx = (1.0 / 12.0) * mass * (size.y() * size.y() + size.z() * size.z());
    const iyy = (1.0 / 12.0) * mass * (size.x() * size.x() + size.z() * size.z());
    const izz = (1.0 / 12.0) * mass * (size.x() * size.x() + size.y() * size.y());

    return [3][3]f32{
        [3]f32{ ixx, 0.0, 0.0 },
        [3]f32{ 0.0, iyy, 0.0 },
        [3]f32{ 0.0, 0.0, izz },
    };
}
