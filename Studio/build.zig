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

    const name = switch (target.result.os.tag) {
        .windows => "DroneStudio-x86_64-windows-gnu.exe",
        .linux => "DroneStudio-x86_64-linux-gnu.exe",
        .macos => "DroneStudio-x86_64-macos",
        else => "DroneStudio-x86_64-unknown",
    };

    const exe = b.addExecutable(.{
        .name = name,
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const zigimg_dependency = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("zigimg", zigimg_dependency.module("zigimg"));

    // FFmpeg library configuration
    const ffmpeg_path = switch (target.result.os.tag) {
        .windows => "lib/ffmpeg-windows",
        .linux => "/usr/lib",
        .macos => "/usr/local/opt/ffmpeg/lib",
        else => @panic("Unsupported operating system"),
    };

    // FFmpeg include paths
    const ffmpeg_include_paths = [_][]const u8{
        b.pathJoin(&.{ ffmpeg_path, "include" }),
    };

    // Add FFmpeg include paths
    for (ffmpeg_include_paths) |include_path| {
        exe.addIncludePath(b.path(include_path));
    }

    // FFmpeg library names
    const ffmpeg_libs = [_][]const u8{ "avcodec", "avformat", "avutil", "swscale" };

    // Link FFmpeg libraries
    switch (target.result.os.tag) {
        .windows => {
            // For Windows, use static libraries
            exe.addLibraryPath(b.path(b.pathJoin(&.{ ffmpeg_path, "lib" })));

            // Use the full library names
            inline for (ffmpeg_libs) |lib| {
                exe.linkSystemLibrary(lib);
            }

            // Windows-specific dependencies
            const win_libs = [_][]const u8{
                "bcrypt",
                "secur32",
                "ws2_32",
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
            // For Linux, use dynamic libraries
            for (ffmpeg_libs) |lib| {
                exe.linkSystemLibrary(lib);
            }
        },
        .macos => {
            // For macOS, use dynamic libraries
            for (ffmpeg_libs) |lib| {
                exe.linkSystemLibrary(lib);
            }
        },
        else => @panic("Unsupported operating system"),
    }

    // GLAD configuration (same for all platforms)
    const glad_path = "lib/glad";
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
