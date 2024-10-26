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
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "DroneStudio",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // GLAD configuration (same for all platforms)
    const glad_path = "src/glad";
    const glad_include_path = b.path(b.pathJoin(&.{ glad_path, "include" }));
    const glad_src_path = b.path(b.pathJoin(&.{ glad_path, "src", "glad.c" }));

    exe.addIncludePath(glad_include_path);
    exe.addCSourceFile(.{
        .file = glad_src_path,
        .flags = &.{"-std=c99"},
    });

    // Platform-specific configuration
    switch (target.result.os.tag) {
        .windows => {
            // Windows-specific configuration
            const glfw_path = "lib/glfw";
            const glfw_include_path = b.pathJoin(&.{ glfw_path, "include" });
            const glfw_lib_path = b.pathJoin(&.{ glfw_path, "lib-mingw-w64" });

            exe.addIncludePath(b.path(glfw_include_path));
            exe.addLibraryPath(b.path(glfw_lib_path));
            exe.addObjectFile(b.path(b.pathJoin(&.{ glfw_lib_path, "libglfw3.a" })));

            // Windows system libraries
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

            for (win_libs) |lib| {
                exe.linkSystemLibrary(lib);
            }
        },
        .linux => {
            // Linux configuration
            exe.linkSystemLibrary("glfw");
            exe.linkSystemLibrary("GL");
            // Optional: Add X11 dependencies if needed
            exe.linkSystemLibrary("X11");
            exe.linkSystemLibrary("dl");
            exe.linkSystemLibrary("pthread");
        },
        .macos => {
            // macOS configuration
            exe.linkSystemLibrary("glfw");
            exe.linkSystemLibrary("OpenGL");

            const frameworks = [_][]const u8{
                "Cocoa",
                "IOKit",
                "CoreFoundation",
            };

            for (frameworks) |framework| {
                exe.linkFramework(framework);
            }
        },
        else => {
            @panic("Unsupported operating system");
        },
    }

    exe.linkLibC();

    // Install the executable
    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Add test step
    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
