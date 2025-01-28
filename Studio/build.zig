const std = @import("std");
const builtin = @import("builtin");
const Build = std.Build;
const process = std.process;

// TODO: Look for a way to set these env variables in build script on linux
// try std.process.setEnvVar("__NV_PRIME_RENDER_OFFLOAD", "1");
// try std.process.setEnvVar("__GLX_VENDOR_LIBRARY_NAME", "nvidia");

// Helper function to determine the OpenGL library based on the target OS
fn getOpenGLLib(target: std.Build.ResolvedTarget) []const u8 {
    return switch (target.result.os.tag) {
        .linux => "GL",
        .windows => "opengl32",
        else => if (std.Target.isDarwin(target.result)) "OpenGL" else "",
    };
}

// Helper function to configure library paths and link libraries
fn configureLibs(
    exe: *Build.Step.Compile,
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    use_cuda: bool,
    ffmpeg_path: []const u8,
) void {
    // Add common library paths
    exe.addLibraryPath(Build.LazyPath{ .cwd_relative = "/usr/local/lib" });

    // CUDA Configuration
    if (use_cuda) {
        // Determine CUDA paths based on target OS
        const cuda_path = switch (target.result.os.tag) {
            .windows => "lib/cuda-windows",
            .linux => "cuda",
            .macos => @panic("CUDA is currently not supported for MacOS"),
            else => @panic("Unsupported OS for CUDA"),
        };

        // Add CUDA include path
        exe.addIncludePath(b.path(b.pathJoin(&.{ cuda_path, "include" })));

        // Determine CUDA library path
        const cuda_lib_path: ?std.Build.LazyPath = switch (target.result.os.tag) {
            .windows => blk: {
                // If cross-compiling from Linux to Windows, adjust the path accordingly
                // Example placeholder logic; adjust as needed
                if (builtin.target.os.tag == .linux) {
                    break :blk b.path(b.pathJoin(&.{ cuda_path, "lib" }));
                }
                break :blk b.path(b.pathJoin(&.{ cuda_path, "lib", "x64" }));
            },
            .linux => null, // Adjust if needed
            .macos => @panic("CUDA is currently not supported for MacOS"),
            else => @panic("Unsupported OS for CUDA"),
        };
        if (cuda_lib_path) |cu_path| {
            exe.addLibraryPath(cu_path);
        }

        // Link CUDA libraries
        const cuda_libs_to_link = [_][]const u8{
            "cuda",
            "cudart",
            "nppig",
            "npps",
        };

        inline for (cuda_libs_to_link) |lib| {
            exe.linkSystemLibrary(lib);
        }

        // Add CUDA configuration module
        exe.root_module.addAnonymousImport("cuda_config", .{
            .root_source_file = b.addWriteFiles().add("cuda_config.zig", "pub const CUDA_ENABLED = true;\n"),
        });
    } else {
        // Provide a default CUDA configuration module if not enabled
        exe.root_module.addAnonymousImport("cuda_config", .{
            .root_source_file = b.addWriteFiles().add("cuda_config.zig", "pub const CUDA_ENABLED = false;\n"),
        });
    }

    // FFmpeg Configuration

    // Add FFmpeg include paths
    const ffmpeg_include_paths = [_][]const u8{
        b.pathJoin(&.{ ffmpeg_path, "include" }),
    };
    for (ffmpeg_include_paths) |include_path| {
        exe.addIncludePath(Build.LazyPath{ .cwd_relative = include_path });
    }

    // FFmpeg libraries to link
    const ffmpeg_libs = [_][]const u8{
        "avfilter", "avcodec",  "avformat", "avutil",
        "swscale",  "avdevice", "postproc", "swresample",
    };

    // Link FFmpeg libraries based on OS
    switch (target.result.os.tag) {
        .windows => {
            exe.addLibraryPath(b.path(b.pathJoin(&.{ ffmpeg_path, "lib" })));
            inline for (ffmpeg_libs) |lib| {
                exe.linkSystemLibrary(lib);
            }

            // Windows-specific dependencies
            const win_libs = [_][]const u8{
                "bcrypt",   "secur32",  "ws2_32",  "gdi32",
                "user32",   "kernel32", "shell32", "opengl32",
                "comdlg32", "winmm",    "ole32",   "uuid",
            };
            for (win_libs) |lib| {
                exe.linkSystemLibrary(lib);
            }
        },
        .linux => {
            inline for (ffmpeg_libs) |lib| {
                exe.linkSystemLibrary(lib);
            }
        },
        .macos => {
            @panic("MacOS is currently not supported!");
        },
        else => @panic("Unsupported operating system"),
    }

    // GLAD Configuration
    const glad_path = "lib/glad";
    const glad_include_path = b.path(b.pathJoin(&.{ glad_path, "include" }));
    const glad_src_path = b.path(b.pathJoin(&.{ glad_path, "src", "glad.c" }));

    exe.addIncludePath(glad_include_path);
    exe.addCSourceFile(.{
        .file = glad_src_path,
        .flags = &.{"-std=c99"},
    });

    // Platform-specific GLFW Configuration
    switch (target.result.os.tag) {
        .windows => {
            const glfw_path = "lib/glfw";
            const glfw_include_path = b.pathJoin(&.{ glfw_path, "include" });
            const glfw_lib_path = b.pathJoin(&.{ glfw_path, "lib-mingw-w64" });

            exe.addIncludePath(b.path(glfw_include_path));
            exe.addLibraryPath(b.path(glfw_lib_path));
            exe.addObjectFile(b.path(b.pathJoin(&.{ glfw_lib_path, "libglfw3.a" })));

            // Windows system libraries
            const win_libs = [_][]const u8{
                "gdi32",    "user32",   "kernel32", "shell32",
                "opengl32", "comdlg32", "winmm",    "ole32",
                "uuid",
            };
            for (win_libs) |lib| {
                exe.linkSystemLibrary(lib);
            }
        },
        .linux => {
            exe.linkSystemLibrary("glfw");
            exe.linkSystemLibrary(getOpenGLLib(target));
            // Optional: Add X11 dependencies if needed
            exe.linkSystemLibrary("X11");
            exe.linkSystemLibrary("dl");
            exe.linkSystemLibrary("pthread");
        },
        .macos => {
            @panic("MacOS is currently not supported!");
        },
        else => {
            @panic("Unsupported operating system");
        },
    }

    // ImGUI dependencies
    const imgui_path = "lib/cimgui";
    const imgui_sources = [_][]const u8{
        "cimgui.cpp",
        "imgui/imgui.cpp",
        "imgui/imgui_draw.cpp",
        "imgui/imgui_tables.cpp",
        "imgui/imgui_widgets.cpp",
        "imgui/imgui_demo.cpp",
        "imgui/imgui_impl_glfw.cpp",
        "imgui/imgui_impl_opengl3.cpp",
    };

    const cpp_flags = [_][]const u8{
        "-std=c++11",
        "-DIMGUI_IMPL_API=extern \"C\"",
        "-DCIMGUI_USE_GLFW=1", // Enable GLFW backend
        "-DCIMGUI_USE_OPENGL3=1", // Enable OpenGL3 backend
    };

    exe.addIncludePath(b.path(imgui_path));
    for (imgui_sources) |source| {
        exe.addCSourceFile(.{
            .file = b.path(b.pathJoin(&.{ imgui_path, source })),
            .flags = &cpp_flags,
        });
    }

    // Link the C standard library
    exe.linkLibC();
    exe.linkLibCpp();
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const use_cuda = b.option(bool, "cuda", "Enable CUDA hardware acceleration") orelse false;
    const ffmpeg_path = switch (target.result.os.tag) {
        .windows => "lib/ffmpeg-windows",
        .linux => b.option([]const u8, "ffmpeg_path", "Path to ffmpeg installation") orelse "ffmpeg",
        .macos => @panic("MacOS is currently not supported!"),
        else => @panic("Unsupported operating system"),
    };

    const name = switch (target.result.os.tag) {
        .windows => "DroneStudio-x86_64-windows-gnu.exe",
        .linux => "DroneStudio-x86_64-linux-gnu.exe",
        .macos => @panic("MacOS is currently not supported!"),
        else => @panic("Unsupported OS!"),
    };

    // Add the main executable
    const exe = b.addExecutable(.{
        .name = name,
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Configure libraries for the main executable
    configureLibs(exe, b, target, use_cuda, ffmpeg_path);

    if (use_cuda) {
        const cuda_detector_obj = b.addSystemCommand(&.{
            "nvcc",
            "-O3",
            "--compiler-options",
            "'-fPIC'",
            "-c",
            "lib/kernels/kernels.cu",
            "-o",
            "lib/kernels/kernels.o",
        });

        const cuda_detector_artifact = b.addObject(.{
            .name = "cuda_kernels",
            .root_source_file = null,
            .target = target,
            .optimize = optimize,
        });
        cuda_detector_artifact.addObjectFile(b.path("lib/kernels/kernels.o"));
        cuda_detector_artifact.step.dependOn(&cuda_detector_obj.step);

        exe.addObjectFile(cuda_detector_artifact.getEmittedBin());
    }

    exe.addIncludePath(b.path("lib/kernels"));

    // Install the executable
    b.installArtifact(exe);

    // Run command for the main executable
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Add the test executable
    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Configure libraries for the test executable
    configureLibs(exe_unit_tests, b, target, use_cuda, ffmpeg_path);

    // Run command for the test executable
    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
