const std = @import("std");

// Helper function to determine the OpenGL library based on the target OS
fn getOpenGLLib(target: std.Build.ResolvedTarget) []const u8 {
    return switch (target.result.os.tag) {
        .linux => "GL", // For Linux (including WSL2)
        .windows => "opengl32", // For Windows
        else => {
            if (std.Target.isDarwin(target.result)) {
                return "OpenGL";
            } else {
                return "";
            }
        },
    };
}

pub fn build(b: *std.Build) void {
    // Standard target options allow users to choose the build target
    const target = b.standardTargetOptions(.{});

    // Standard optimization options
    const optimize = b.standardOptimizeOption(.{});

    // Add the executable "DroneStudio" from src/main.zig
    const exe = b.addExecutable(.{
        .name = "DroneStudio",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const glad_files = &.{"src/glad/src/glad.c"};
    exe.addIncludePath(b.path("src/glad/include/"));
    exe.addCSourceFiles(.{ .root = b.path("."), .files = glad_files });

    // Link against GLFW
    exe.linkSystemLibrary("glfw");
    exe.linkLibC();

    // Link against the appropriate OpenGL library based on the target OS
    exe.linkSystemLibrary(getOpenGLLib(target));

    // For macOS, link additional frameworks required by GLFW and OpenGL
    if (std.Target.isDarwin(target.result)) {
        exe.linkSystemLibrary("Cocoa");
        exe.linkSystemLibrary("IOKit");
        exe.linkSystemLibrary("CoreFoundation");
    }

    exe.linkSystemLibrary("c");

    // Optionally, specify additional include and library paths if GLFW is in a non-standard location
    // Uncomment and modify the paths below if necessary

    // exe.addSystemIncludeDir("/path/to/glfw/include");
    // exe.addLibraryPath("/path/to/glfw/lib");

    // Install the executable artifact
    b.installArtifact(exe);

    // Create a Run step to execute the program
    const run_cmd = b.addRunArtifact(exe);

    // Make the run step depend on the install step
    run_cmd.step.dependOn(b.getInstallStep());

    // Allow passing arguments to the application via `zig build run -- arg1 arg2`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Create a build step named "run" to execute the program
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Create unit tests for the executable
    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Create a build step named "test" to run all unit tests
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
