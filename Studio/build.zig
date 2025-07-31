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
fn configureDesktopLibs(
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

    exe.addIncludePath(b.path("lib/cbullet"));
    exe.addIncludePath(b.path("lib/bullet"));
    exe.addIncludePath(b.path("lib/vhacd"));

    // TODO: Use the old damping method for now otherwise there is a hang in powf().
    const flags = &.{
        "-DBT_USE_OLD_DAMPING_METHOD",
        "-DBT_THREADSAFE=1",
        "-std=c++11",
        "-fno-sanitize=undefined",
        "-O3", // Maximum optimization
        "-ffast-math", // Enable fast math optimizations
        "-march=native", // Use native CPU instructions
        "-mtune=native", // Tune for native CPU
        "-fomit-frame-pointer", // Omit frame pointer for better performance
        "-funroll-loops", // Unroll loops for better performance
    };
    exe.addCSourceFiles(.{
        .files = &.{
            "lib/cbullet/cbullet.cpp",
            "lib/bullet/btLinearMathAll.cpp",
            "lib/bullet/btBulletCollisionAll.cpp",
            "lib/bullet/btBulletDynamicsAll.cpp",
            "lib/vhacd/vhacd_wrapper.cpp",
        },
        .flags = flags,
    });

    // Link the C standard library
    exe.linkLibC();
    exe.linkLibCpp();
}

fn configureKernels(
    b: *std.Build,
    exe: *Build.Step.Compile,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    use_cuda: bool,
) void {
    _ = target;
    _ = optimize;

    if (use_cuda) {
        const kernel_cu_path = "lib/kernels/kernels.cu";
        const kernel_o_path = "lib/kernels/kernels.o";

        // Check if we need to recompile the CUDA kernels
        var need_cuda_compile = false;

        // Check if .o file exists
        const obj_stat = std.fs.cwd().statFile(kernel_o_path) catch |err| blk: {
            if (err == error.FileNotFound) {
                std.debug.print("CUDA kernels.o not found, will compile...\n", .{});
                need_cuda_compile = true;
            }
            break :blk null;
        };

        // If .o exists, check if .cu is newer
        if (!need_cuda_compile and obj_stat != null) {
            const cu_stat = std.fs.cwd().statFile(kernel_cu_path) catch unreachable;
            if (cu_stat.mtime > obj_stat.?.mtime) {
                std.debug.print("CUDA kernels.cu is newer than kernels.o, will recompile...\n", .{});
                need_cuda_compile = true;
            }
        }

        // Only run nvcc if needed
        if (need_cuda_compile) {
            const cuda_compile_cmd = b.addSystemCommand(&.{
                "nvcc",
                "-O3",
                "--compiler-options",
                "'-fPIC'",
                "-c",
                kernel_cu_path,
                "-o",
                kernel_o_path,
            });
            cuda_compile_cmd.step.name = "Compile CUDA kernels";
            // Make sure nvcc runs before we try to link the object file
            b.getInstallStep().dependOn(&cuda_compile_cmd.step);
        }

        // Always link the object file (either newly compiled or existing)
        exe.addObjectFile(b.path(kernel_o_path));
    }

    exe.addIncludePath(b.path("lib/kernels"));
}

// Get command-line options
pub fn build(b: *std.Build) void {
    const build_desktop = b.option(bool, "desktop", "Build the desktop application") orelse true;
    const build_pi = b.option(bool, "pi", "Build the Raspberry Pi applications") orelse true;
    const use_cuda = b.option(bool, "cuda", "Enable CUDA hardware acceleration for desktop") orelse false;
    const test_gui = b.option(bool, "test-gui", "Enable GUI mode for physics tests") orelse false;

    // Get target modification options
    const global_target = b.option(bool, "global-target", "Apply target settings to both desktop and Pi builds") orelse false;
    const desktop_only = b.option(bool, "desktop-target", "Apply target settings only to desktop build") orelse false;
    const pi_only = b.option(bool, "pi-target", "Apply target settings only to Pi build") orelse false;

    // User can provide a global target that applies to all builds
    const user_target = b.standardTargetOptions(.{}); // User can override with -Dtarget

    // Default optimization option
    const optimize = b.standardOptimizeOption(.{});

    // Handle desktop target selection
    const desktop_target = if (global_target or desktop_only)
        user_target // Use user-provided target for desktop if requested
    else
        b.graph.host; // Default to host architecture for desktop

    // Handle Raspberry Pi target selection
    const pi_target = if (global_target or pi_only)
        user_target // Use user-provided target for Pi if requested
    else blk: {
        // Create a specific target for Raspberry Pi with cortex_a53
        const target_options = std.Target.Query{
            .cpu_arch = .aarch64,
            .os_tag = .linux,
            .abi = .gnu,
            .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.cortex_a53 },
        };
        break :blk b.resolveTargetQuery(target_options);
    };

    // Log the targets being used
    std.debug.print("Desktop target: {s} {s}\n", .{ @tagName(desktop_target.result.cpu.arch), desktop_target.result.cpu.model.name });
    std.debug.print("Pi target: {s} {s}\n", .{ @tagName(pi_target.result.cpu.arch), pi_target.result.cpu.model.name });

    // Desktop Application
    var desktop_step: ?*std.Build.Step = null;
    var test_desktop_step: ?*std.Build.Step = null;
    if (build_desktop) {
        desktop_step = b.step("desktop", "Build the desktop application");

        // Determine FFmpeg path based on OS
        const ffmpeg_path = switch (desktop_target.result.os.tag) {
            .windows => "lib/ffmpeg-windows",
            .linux => b.option([]const u8, "ffmpeg_path", "Path to ffmpeg installation") orelse "ffmpeg",
            .macos => @panic("MacOS is currently not supported!"),
            else => @panic("Unsupported operating system"),
        };

        // Determine executable name based on OS and architecture
        const arch_name = @tagName(desktop_target.result.cpu.arch);
        const exe_name = b.fmt("DroneStudio-{s}-{s}-gnu{s}", .{
            arch_name,
            @tagName(desktop_target.result.os.tag),
            if (desktop_target.result.os.tag == .windows) ".exe" else "",
        });

        // Add the main desktop executable
        const desktop_exe = b.addExecutable(.{
            .name = exe_name,
            .root_source_file = b.path("src/Studio.zig"),
            .target = desktop_target,
            .optimize = optimize,
        });

        // Configure libraries for the desktop executable
        configureDesktopLibs(
            desktop_exe,
            b,
            desktop_target,
            use_cuda,
            ffmpeg_path,
        );

        // Configure CUDA if enabled
        configureKernels(
            b,
            desktop_exe,
            desktop_target,
            optimize,
            use_cuda,
        );

        // Install the desktop executable
        b.installArtifact(desktop_exe);
        desktop_step.?.dependOn(&desktop_exe.step);

        const exe_check = b.addExecutable(.{
            .name = exe_name,
            .root_source_file = b.path("src/Studio.zig"),
            .target = desktop_target,
            .optimize = optimize,
        });

        configureDesktopLibs(
            exe_check,
            b,
            desktop_target,
            use_cuda,
            ffmpeg_path,
        );

        // Configure CUDA if enabled
        configureKernels(
            b,
            exe_check,
            desktop_target,
            optimize,
            use_cuda,
        );

        const check = b.step("check", "Check if it compiles");
        check.dependOn(&exe_check.step);

        // Run command for the desktop executable
        const run_desktop_cmd = b.addRunArtifact(desktop_exe);
        run_desktop_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| {
            run_desktop_cmd.addArgs(args);
        }

        const run_desktop_step = b.step("run-desktop", "Run the desktop application");
        run_desktop_step.dependOn(&run_desktop_cmd.step);

        // Add the desktop test executable
        const desktop_tests = b.addTest(.{
            .root_source_file = b.path("src/Studio.zig"),
            .target = desktop_target,
            .optimize = optimize,
        });

        // Configure libraries for the desktop test executable
        configureDesktopLibs(
            desktop_tests,
            b,
            desktop_target,
            use_cuda,
            ffmpeg_path,
        );

        // Configure kernels for tests
        configureKernels(
            b,
            desktop_tests,
            desktop_target,
            optimize,
            use_cuda,
        );

        // Add test GUI configuration
        desktop_tests.root_module.addAnonymousImport("test_config", .{
            .root_source_file = b.addWriteFiles().add("test_config.zig", b.fmt("pub const GUI_ENABLED = {any};\n", .{test_gui})),
        });

        // Run command for the desktop test executable
        const run_desktop_tests = b.addRunArtifact(desktop_tests);
        test_desktop_step = b.step("test-desktop", "Run desktop application unit tests");
        test_desktop_step.?.dependOn(&run_desktop_tests.step);

        // // Add collision tests as a test, not an executable
        // const collision_tests = b.addTest(.{
        //     .name = "collision_tests",
        //     .root_source_file = b.path("src/core/ecs/components/PhysicsThread.zig"),
        //     .target = desktop_target,
        //     .optimize = optimize,
        // });

        // // Configure libraries for collision tests
        // configureDesktopLibs(
        //     collision_tests,
        //     b,
        //     desktop_target,
        //     use_cuda,
        //     ffmpeg_path,
        // );

        // // Run command for collision tests
        // const run_collision_tests = b.addRunArtifact(collision_tests);
        // const test_collision_step = b.step("test-collision", "Run collision system tests");
        // test_collision_step.dependOn(&run_collision_tests.step);
    }

    // Raspberry Pi Applications
    var pi_step: ?*std.Build.Step = null;
    var test_pi_step: ?*std.Build.Step = null;
    if (build_pi) {
        pi_step = b.step("pi", "Build the Raspberry Pi applications");

        // Log Raspberry Pi target information
        std.debug.print("Building for Raspberry Pi: aarch64-linux-gnu with {s} CPU\n", .{pi_target.result.cpu.model.name});

        // Create IMU executable
        const imu_exe = b.addExecutable(.{
            .name = "IMU",
            .root_source_file = b.path("src/IMU.zig"),
            .target = pi_target,
            .optimize = optimize,
        });
        b.installArtifact(imu_exe);
        pi_step.?.dependOn(&imu_exe.step);

        // Create MotorController executable
        const motor_exe = b.addExecutable(.{
            .name = "MotorController",
            .root_source_file = b.path("src/MotorController.zig"),
            .target = pi_target,
            .optimize = optimize,
        });
        b.installArtifact(motor_exe);
        pi_step.?.dependOn(&motor_exe.step);

        // Run commands for IMU (will only run if we're on the correct architecture)
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
            .target = pi_target,
            .optimize = optimize,
        });
        const run_imu_tests = b.addRunArtifact(imu_tests);

        // Unit tests for MotorController
        const motor_tests = b.addTest(.{
            .root_source_file = b.path("src/MotorController.zig"),
            .target = pi_target,
            .optimize = optimize,
        });
        const run_motor_tests = b.addRunArtifact(motor_tests);

        // Create test steps for each executable
        const test_imu_step = b.step("test-imu", "Run IMU unit tests");
        test_imu_step.dependOn(&run_imu_tests.step);

        const test_motor_step = b.step("test-motor", "Run MotorController unit tests");
        test_motor_step.dependOn(&run_motor_tests.step);

        // Combined test step for all Pi applications
        test_pi_step = b.step("test-pi", "Run all Raspberry Pi unit tests");
        test_pi_step.?.dependOn(&run_imu_tests.step);
        test_pi_step.?.dependOn(&run_motor_tests.step);
    }
}
