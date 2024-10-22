const std = @import("std");

// Helper function to determine the OpenGL library based on the target OS
fn getOpenGLLib(target: std.Build.ResolvedTarget) []const u8 {
    return switch (target.result.os.tag) {
        .linux => "GL",
        .windows => "opengl32",
        else => if (std.Target.isDarwin(target.result)) "OpenGL" else "",
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

    const glad_path = "src/glad";
    const glad_include_path = b.path(b.pathJoin(&.{ glad_path, "include" }));
    const glad_src_path = b.path(b.pathJoin(&.{ glad_path, "src", "glad.c" }));

    exe.addIncludePath(glad_include_path);
    exe.addCSourceFile(.{
        .file = glad_src_path,
        .flags = &.{"-std=c99"},
    });

    exe.linkLibC();
    exe.linkSystemLibrary("glfw");

    // Link against the appropriate OpenGL library based on the target OS
    exe.linkSystemLibrary(getOpenGLLib(target));

    // For macOS, link additional frameworks required by GLFW and OpenGL
    if (std.Target.isDarwin(target.result)) {
        const frameworks = [_][]const u8{
            "Cocoa", "IOKit", "CoreFoundation",
        };

        for (frameworks) |framework| {
            exe.linkFramework(framework);
        }
    }

    // Optionally, specify additional include and library paths if GLFW is in a non-standard location
    // Uncomment and modify the paths below if necessary

    // Install the executable artifact
    b.installArtifact(exe);

    const exe_win = b.addExecutable(.{
        .name = "main_windows",
        .root_source_file = b.path("src/main.zig"),
        .target = b.resolveTargetQuery(.{
            .os_tag = .windows,
            .abi = .gnu,
        }),
        .optimize = optimize,
    });

    const glfw_path = "lib/glfw";
    const glfw_include_path = b.pathJoin(&.{ glfw_path, "include" });
    const glfw_lib_path = b.pathJoin(&.{ glfw_path, "lib-mingw-w64" });

    // Ensure Windows SDK headers are available
    exe_win.addIncludePath(glad_include_path);
    exe_win.addCSourceFile(.{
        .file = glad_src_path,
        .flags = &.{"-std=c99"},
    });

    // Add GLFW paths
    exe_win.addIncludePath(b.path(glfw_include_path));
    exe_win.addLibraryPath(b.path(glfw_lib_path));
    exe_win.addObjectFile(b.path(b.pathJoin(&.{ glfw_lib_path, "libglfw3.a" })));

    // Add Windows-specific system libraries
    const win_libs = [_][]const u8{
        "gdi32",
        "user32",
        "kernel32",
        "shell32",
        "opengl32",
        "comdlg32",
        "winmm",
        "ole32",
        "uuid",
    };

    // Link Windows system libraries
    for (win_libs) |lib| {
        exe_win.linkSystemLibrary(lib);
    }

    exe_win.linkLibC();
    exe_win.linkage = std.builtin.LinkMode.dynamic;

    b.installArtifact(exe_win);

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
