// zig build -Dtarget="aarch64-linux-gnu" -Dcpu="cortex_a53" -Doptimize=ReleaseFast

const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    // Create IMU executable
    const imu_exe = b.addExecutable(.{
        .name = "IMU",
        .root_source_file = b.path("src/IMU.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(imu_exe);

    // Create MotorController executable
    const motor_exe = b.addExecutable(.{
        .name = "MotorController",
        .root_source_file = b.path("src/MotorController.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(motor_exe);

    // Run commands for IMU
    const run_imu_cmd = b.addRunArtifact(imu_exe);
    run_imu_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_imu_cmd.addArgs(args);
    }

    // Run commands for MotorController
    const run_motor_cmd = b.addRunArtifact(motor_exe);
    run_motor_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_motor_cmd.addArgs(args);
    }

    // Create run steps for each executable
    const run_imu_step = b.step("run-imu", "Run the IMU application");
    run_imu_step.dependOn(&run_imu_cmd.step);

    const run_motor_step = b.step("run-motor", "Run the MotorController application");
    run_motor_step.dependOn(&run_motor_cmd.step);

    // Unit tests for IMU
    const imu_tests = b.addTest(.{
        .root_source_file = b.path("src/IMU.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_imu_tests = b.addRunArtifact(imu_tests);

    // Unit tests for MotorController
    const motor_tests = b.addTest(.{
        .root_source_file = b.path("src/MotorController.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_motor_tests = b.addRunArtifact(motor_tests);

    // Create test steps for each executable
    const test_imu_step = b.step("test-imu", "Run IMU unit tests");
    test_imu_step.dependOn(&run_imu_tests.step);

    const test_motor_step = b.step("test-motor", "Run MotorController unit tests");
    test_motor_step.dependOn(&run_motor_tests.step);
}
