// src/Scene.zig
const std = @import("std");
const Shape = @import("Shape.zig");
const Math = @import("Math.zig");
const Node = @import("Node.zig");
const Mesh = @import("Mesh.zig");
const gl = @import("bindings/gl.zig");
const c = @import("bindings/c.zig");
const Cameras = @import("Cameras.zig");
const Vision = @import("Vision.zig");
const Sensors = @import("Sensors.zig");
const Drone = @import("Drone.zig");

const Vec3 = Math.Vec3;
const Mat4 = Math.Mat4;
const TextureUnit = Node.TextureUnit;
const Quaternion = Math.Quaternion;
const File = std.fs.File;
const glfw = gl.glfw;
const glad = gl.glad;
const imgui = c.imgui;
const Pose = Vision.CameraPose;
const CameraManager = Cameras.CameraManager;
const MotorController = Drone.MotorControllerClient;

const GSLWError = error{ FailedToCreateWindow, FailedToInitialize };
const ShaderError = error{ UnableToCreateShader, ShaderCompilationFailed, UnableToCreateProgram, ShaderLinkingFailed, UnableToCreateWindow };

pub const AppState = struct {
    last_frame_time: f64 = 0.0, // Time of the last frame
    delta_time: f32 = 0.0, // Time between current frame and last frame
    last_mouse_x: f64 = 0.0,
    last_mouse_y: f64 = 0.0,
    first_mouse: bool = true,
    zoom: f32 = 90.0,
    keys: [1024]bool = .{false} ** 1024,
    paused: bool = false,
    fly: bool = false,
    menu: bool = false,
    simulation_mode: bool = false,
    first_person_view: bool = true,
};

// Timing for operations to debug in UI
pub const timing = struct {
    match_timing: f32,
    detection_timing: f32,
    draw_timing: f32,
};

pub const RenderingMode = enum {
    Standard,
    PBR,
};

pub const Scene = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    nodes: std.StringHashMap(*Node),
    width: f32,
    height: f32,
    appState: AppState,
    camera_manager: *CameraManager,
    motor_controller: ?*MotorController = null,

    update_viewports: bool = false,

    ambientColorLoc: glad.GLint,
    ambientStrengthLoc: glad.GLint,

    ambient_color: [3]f32 = .{ 1.0, 1.0, 1.0 }, // Default white ambient
    ambient_strength: f32 = 0.1,

    uModelLoc: glad.GLint,
    uViewLoc: glad.GLint,
    uProjectionLoc: glad.GLint,
    useTextureLoc: glad.GLint,
    yTextureLoc: glad.GLint,
    uvTextureLoc: glad.GLint,
    depthTextureLoc: glad.GLint,
    useInstancedKeypointLoc: glad.GLint,
    useInstancedLinesLoc: glad.GLint,

    rendering_mode: RenderingMode = .Standard,
    shaderProgram: u32,
    pbrShaderProgram: u32 = 0,

    // Standard MVP uniform locations
    pbr_uModelLoc: glad.GLint = -1,
    pbr_uViewLoc: glad.GLint = -1,
    pbr_uProjectionLoc: glad.GLint = -1,

    // PBR material uniforms
    pbr_baseColorFactorLoc: glad.GLint = -1,
    pbr_metallicFactorLoc: glad.GLint = -1,
    pbr_roughnessFactorLoc: glad.GLint = -1,

    // PBR texture uniforms
    pbr_useTextureLoc: glad.GLint = -1,
    pbr_hasBaseColorTextureLoc: glad.GLint = -1,
    pbr_hasMetallicRoughnessTextureLoc: glad.GLint = -1,
    pbr_hasNormalTextureLoc: glad.GLint = -1,
    pbr_hasOcclusionTextureLoc: glad.GLint = -1,
    pbr_hasEmissiveTextureLoc: glad.GLint = -1,

    // PBR texture samplers
    pbr_baseColorTextureLoc: glad.GLint = -1,
    pbr_metallicRoughnessTextureLoc: glad.GLint = -1,
    pbr_normalTextureLoc: glad.GLint = -1,
    pbr_occlusionTextureLoc: glad.GLint = -1,
    pbr_emissiveTextureLoc: glad.GLint = -1,

    // Specular-Glossiness extension uniforms
    pbr_useSpecularGlossinessLoc: glad.GLint = -1,
    pbr_diffuseFactorLoc: glad.GLint = -1,
    pbr_specularFactorLoc: glad.GLint = -1,
    pbr_glossinessFactorLoc: glad.GLint = -1,

    // Specular extension uniforms
    pbr_useSpecularExtensionLoc: glad.GLint = -1,
    pbr_specularStrengthLoc: glad.GLint = -1,
    pbr_specularColorFactorLoc: glad.GLint = -1,

    // Other PBR uniforms
    pbr_emissiveFactorLoc: glad.GLint = -1,
    pbr_emissiveStrengthLoc: glad.GLint = -1,
    pbr_alphaCutoffLoc: glad.GLint = -1,
    pbr_alphaModeLoc: glad.GLint = -1,

    // Lighting uniforms
    pbr_viewPosLoc: glad.GLint = -1,
    pbr_lightPositionLoc: glad.GLint = -1,
    pbr_lightColorLoc: glad.GLint = -1,
    pbr_lightIntensityLoc: glad.GLint = -1,

    pbr_ambientColorLoc: glad.GLint = -1,
    pbr_ambientStrengthLoc: glad.GLint = -1,

    // Default lighting values
    lightPosition: [3]f32 = .{ 0.0, 10.0, 10.0 },
    lightColor: [3]f32 = .{ 1.0, 1.0, 1.0 },
    lightIntensity: f32 = 1.0,

    texGen: TextureGenerator = TextureGenerator{},

    last_projection: Mat4 = undefined,
    projection_dirty: bool = true,

    frame_count: u64 = 0,
    avg_frame_time: f64 = 0,
    last_fps_time: f64 = 0,
    frame_times: [120]f64 = .{0} ** 120,
    frame_time_index: usize = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        window: ?*glfw.struct_GLFWwindow,
    ) !*Self {
        if (window == null) {
            std.debug.print("Failed to create GLFW window\n", .{});
            return GSLWError.FailedToCreateWindow;
        }

        var width: i32 = undefined;
        var height: i32 = undefined;
        glfw.glfwGetWindowSize(window.?, &width, &height);

        const shader_program = try createShaderProgram("shaders/vertex_shader.glsl", "shaders/fragment_shader.glsl");
        glad.glUseProgram(shader_program);

        var current_program: u32 = 0;
        glad.glGetIntegerv(glad.GL_CURRENT_PROGRAM, @ptrCast(&current_program));
        if (current_program != shader_program) {
            std.debug.print("Shader program not active!\n", .{});
        }

        checkOpenGLError("Uniform Setup");

        var num_active_attributes: c_int = 0;
        glad.glGetProgramiv(shader_program, glad.GL_ACTIVE_ATTRIBUTES, &num_active_attributes);

        std.debug.print("\nGetting Attributes\n", .{});
        std.debug.print("Number of active vertex attributes: {}\n", .{num_active_attributes});

        for (0..@intCast(num_active_attributes)) |i| {
            var name_buffer: [256]u8 = undefined;
            var length: c_int = 0;
            var size: c_int = 0;
            var var_type: c_uint = 0;

            glad.glGetActiveAttrib(
                shader_program,
                @intCast(i),
                name_buffer.len,
                &length,
                &size,
                &var_type,
                &name_buffer,
            );

            std.debug.print(" - Attribute {d}: {s} == Size: {d} == Type: {x}\n", .{ i, name_buffer[0..@as(u32, @intCast(length))], size, var_type });
        }

        glad.glDepthFunc(glad.GL_LESS);

        // Cache uniform locations
        const uModelLoc = glad.glGetUniformLocation(shader_program, "uModel");
        const uViewLoc = glad.glGetUniformLocation(shader_program, "uView");
        const uProjectionLoc = glad.glGetUniformLocation(shader_program, "uProjection");
        const useTextureLoc = glad.glGetUniformLocation(shader_program, "useTexture");
        const yTextureLoc = glad.glGetUniformLocation(shader_program, "yTexture");
        const uvTextureLoc = glad.glGetUniformLocation(shader_program, "uvTexture");
        const depthTextureLoc = glad.glGetUniformLocation(shader_program, "depthTexture");
        const useInstancedKeypointLoc = glad.glGetUniformLocation(shader_program, "uInstancedKeypoints");
        const uInstancedLinesLoc = glad.glGetUniformLocation(shader_program, "uInstancedLines");
        const ambientColorLoc = glad.glGetUniformLocation(shader_program, "ambientColor");
        const ambientStrengthLoc = glad.glGetUniformLocation(shader_program, "ambientStrength");

        if (uModelLoc == -1 or uViewLoc == -1 or uProjectionLoc == -1) {
            std.debug.print("Failed to get one or more uniform locations\n", .{});
            // Handle error appropriately
        }

        const scene = try allocator.create(Scene);

        scene.* = Self{
            .allocator = allocator,
            .nodes = std.StringHashMap(*Node).init(allocator),
            .shaderProgram = shader_program,
            .width = @floatFromInt(width),
            .height = @floatFromInt(height),
            .appState = AppState{},
            .camera_manager = try CameraManager.init(allocator, @floatFromInt(width), @floatFromInt(height)),
            .uModelLoc = uModelLoc,
            .uViewLoc = uViewLoc,
            .uProjectionLoc = uProjectionLoc,
            .useTextureLoc = useTextureLoc,
            .yTextureLoc = yTextureLoc,
            .uvTextureLoc = uvTextureLoc,
            .depthTextureLoc = depthTextureLoc,
            .useInstancedKeypointLoc = useInstancedKeypointLoc,
            .useInstancedLinesLoc = uInstancedLinesLoc,
            .ambientColorLoc = ambientColorLoc,
            .ambientStrengthLoc = ambientStrengthLoc,
        };

        try scene.initPBRShader();

        return scene;
    }

    pub fn initPBRShader(self: *Self) !void {
        std.debug.print("Initializing PBR Shader...\n", .{});

        // Create PBR shader program
        self.pbrShaderProgram = try createShaderProgram("shaders/pbr_vertex.glsl", "shaders/pbr_fragment.glsl");

        // Store uniform locations
        glad.glUseProgram(self.pbrShaderProgram);

        // Standard MVP uniforms
        self.pbr_uModelLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "uModel");
        self.pbr_uViewLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "uView");
        self.pbr_uProjectionLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "uProjection");

        // PBR material uniforms
        self.pbr_baseColorFactorLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "baseColorFactor");
        self.pbr_metallicFactorLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "metallicFactor");
        self.pbr_roughnessFactorLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "roughnessFactor");

        // PBR texture uniforms
        self.pbr_useTextureLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "useTexture");
        self.pbr_hasBaseColorTextureLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "hasBaseColorTexture");
        self.pbr_hasMetallicRoughnessTextureLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "hasMetallicRoughnessTexture");
        self.pbr_hasNormalTextureLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "hasNormalTexture");
        self.pbr_hasOcclusionTextureLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "hasOcclusionTexture");
        self.pbr_hasEmissiveTextureLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "hasEmissiveTexture");

        // PBR texture samplers
        self.pbr_baseColorTextureLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "baseColorTexture");
        self.pbr_metallicRoughnessTextureLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "metallicRoughnessTexture");
        self.pbr_normalTextureLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "normalTexture");
        self.pbr_occlusionTextureLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "occlusionTexture");
        self.pbr_emissiveTextureLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "emissiveTexture");

        glad.glUniform1i(self.pbr_baseColorTextureLoc, TextureUnit.BaseColor.index());
        glad.glUniform1i(self.pbr_normalTextureLoc, TextureUnit.NormalMap.index());
        glad.glUniform1i(self.pbr_metallicRoughnessTextureLoc, TextureUnit.MetallicRoughness.index());
        glad.glUniform1i(self.pbr_occlusionTextureLoc, TextureUnit.Occlusion.index());
        glad.glUniform1i(self.pbr_emissiveTextureLoc, TextureUnit.Emissive.index());

        // Specular-Glossiness extension uniforms
        self.pbr_useSpecularGlossinessLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "useSpecularGlossiness");
        self.pbr_diffuseFactorLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "diffuseFactor");
        self.pbr_specularFactorLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "specularFactor");
        self.pbr_glossinessFactorLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "glossinessFactor");

        // Specular extension uniforms
        self.pbr_useSpecularExtensionLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "useSpecularExtension");
        self.pbr_specularStrengthLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "specularStrength");
        self.pbr_specularColorFactorLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "specularColorFactor");

        // Other PBR uniforms
        self.pbr_emissiveFactorLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "emissiveFactor");
        self.pbr_emissiveStrengthLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "emissiveStrength");
        self.pbr_alphaCutoffLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "alphaCutoff");
        self.pbr_alphaModeLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "alphaMode");

        // Lighting uniforms
        self.pbr_viewPosLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "viewPos");
        self.pbr_lightPositionLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "lightPosition");
        self.pbr_lightColorLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "lightColor");
        self.pbr_lightIntensityLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "lightIntensity");

        // Ambient lighting (reusing from standard shader)
        self.pbr_ambientColorLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "ambientColor");
        self.pbr_ambientStrengthLoc = glad.glGetUniformLocation(self.pbrShaderProgram, "ambientStrength");

        // Set default values
        if (self.pbr_ambientColorLoc != -1) {
            glad.glUniform3fv(self.pbr_ambientColorLoc, 1, &self.ambient_color);
        }
        if (self.pbr_ambientStrengthLoc != -1) {
            glad.glUniform1f(self.pbr_ambientStrengthLoc, self.ambient_strength);
        }

        glad.glUseProgram(0);
    }

    pub fn deinit(self: *Self) void {
        var it = self.nodes.iterator();
        while (it.next()) |entry| {
            const curr_mesh = entry.value_ptr.*.mesh;
            if (curr_mesh) |mesh| {
                glad.glDeleteVertexArrays(1, mesh.meta.VAO);
                glad.glDeleteBuffers(1, mesh.meta.VBO);

                if (mesh.indices) |indices| {
                    _ = indices;
                    glad.glDeleteBuffers(1, mesh.meta.IBO);
                }
            }
        }

        var viewport_it = self.camera_manager.viewports.iterator();
        while (viewport_it.next()) |entry| {
            var viewport = entry.value_ptr;
            viewport.deinit();
        }
        self.camera_manager.viewports.deinit();

        self.nodes.deinit();
        glad.glDeleteProgram(self.shaderProgram);
        cleanupImGui();
    }

    pub fn setupCallbacks(self: *Self, window: ?*glfw.struct_GLFWwindow) void {
        if (window == null) return;

        glfw.glfwSetWindowUserPointer(window, @ptrCast(self));

        const ptr = glfw.glfwGetWindowUserPointer(window);
        if (ptr == null) {
            std.debug.print("Failed to set window user pointer\n", .{});
            return;
        }

        _ = glfw.glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
        _ = glfw.glfwSetCursorPosCallback(window, mouseCallback);
        _ = glfw.glfwSetCharCallback(window, charCallback);
        _ = glfw.glfwSetKeyCallback(window, keyCallback);
        _ = glfw.glfwSetScrollCallback(window, scrollCallback);
        _ = glfw.glfwSetMouseButtonCallback(window, mouseButtonCallback);
    }

    pub fn getSceneGraph(self: Self) void {
        var it = self.nodes.iterator();

        std.debug.print("\nGetting Nodes...\n", .{});

        while (it.next()) |entry| {
            const curr_node = entry.value_ptr.*;

            std.debug.print("\n=================================\nNode: {s}", .{entry.key_ptr.*});
            curr_node.debug();
        }
    }

    pub fn addNode(self: *Self, name: []const u8, node: *Node) !void {
        node.addSceneRecursively(self);
        try self.nodes.put(name, node);
    }

    pub fn render(self: *Self, window: ?*glfw.struct_GLFWwindow) void {
        // Start frame timing
        const frame_start = glfw.glfwGetTime();

        // Update viewports if needed
        if (self.update_viewports) {
            self.update_viewports = self.camera_manager.update_viewports() catch |err| blk: {
                std.debug.print("Failed to update viewports: {}\n", .{err});
                break :blk true;
            };
        }

        if (self.ambientColorLoc != -1) {
            glad.glUniform3fv(self.ambientColorLoc, 1, &self.ambient_color);
        }
        if (self.ambientStrengthLoc != -1) {
            glad.glUniform1f(self.ambientStrengthLoc, self.ambient_strength);
        }

        //TODO: Split updating nodes transformations & draw calls into separate functions, call transformation updates once here, then do draw calls for every FBO
        // --- Render each active camera to its FBO ---
        var vp_it = self.camera_manager.viewports.iterator();
        while (vp_it.next()) |vp_entry| {
            const cam_name = vp_entry.key_ptr.*;
            const viewport = vp_entry.value_ptr;

            // Skip disabled viewports
            if (!viewport.enabled) continue;

            // Retrieve the actual Camera* from camera_manager:
            const cam_ptr = self.camera_manager.cameras.get(cam_name) orelse continue;

            // Bind FBO
            glad.glBindFramebuffer(glad.GL_FRAMEBUFFER, viewport.fbo.fbo);
            glad.glViewport(0, 0, viewport.fbo.width, viewport.fbo.height);

            // Clear
            glad.glClearColor(0.1, 0.1, 0.1, 1.0);
            glad.glClear(glad.GL_COLOR_BUFFER_BIT | glad.GL_DEPTH_BUFFER_BIT);

            // Get view and projection matrices once (used by both shaders)
            const view_mat = cam_ptr.get_view_matrix();
            const view_arr = view_mat.to_array();
            const proj_mat = cam_ptr.get_projection_matrix();
            const proj_arr = proj_mat.to_array();

            // FIRST PASS: PBR objects
            glad.glUseProgram(self.pbrShaderProgram);

            // Set ambient lighting uniform for PBR shader
            if (self.pbr_ambientColorLoc != -1) {
                glad.glUniform3fv(self.pbr_ambientColorLoc, 1, &self.ambient_color);
            }
            if (self.pbr_ambientStrengthLoc != -1) {
                glad.glUniform1f(self.pbr_ambientStrengthLoc, self.ambient_strength);
            }

            // Set camera position for PBR specular calculations
            const camera_pos = cam_ptr.get_base().position;
            if (self.pbr_viewPosLoc != -1) {
                glad.glUniform3f(self.pbr_viewPosLoc, camera_pos.x(), camera_pos.y(), camera_pos.z());
            }

            // Set light properties for PBR shader
            if (self.pbr_lightPositionLoc != -1) {
                glad.glUniform3fv(self.pbr_lightPositionLoc, 1, &self.lightPosition);
            }
            if (self.pbr_lightColorLoc != -1) {
                glad.glUniform3fv(self.pbr_lightColorLoc, 1, &self.lightColor);
            }
            if (self.pbr_lightIntensityLoc != -1) {
                glad.glUniform1f(self.pbr_lightIntensityLoc, self.lightIntensity);
            }

            // Set camera's view and projection for PBR shader
            if (self.pbr_uViewLoc != -1) {
                glad.glUniformMatrix4fv(self.pbr_uViewLoc, 1, glad.GL_FALSE, &view_arr);
            }
            if (self.pbr_uProjectionLoc != -1) {
                glad.glUniformMatrix4fv(self.pbr_uProjectionLoc, 1, glad.GL_FALSE, &proj_arr);
            }

            // Set rendering mode to PBR
            self.rendering_mode = .PBR;

            // Do a PBR pass through all nodes
            var node_it = self.nodes.iterator();
            while (node_it.next()) |node_entry| {
                node_entry.value_ptr.*.update();
            }

            // SECOND PASS: Standard objects
            glad.glUseProgram(self.shaderProgram);

            // Set ambient lighting uniform for standard shader
            if (self.ambientColorLoc != -1) {
                glad.glUniform3fv(self.ambientColorLoc, 1, &self.ambient_color);
            }
            if (self.ambientStrengthLoc != -1) {
                glad.glUniform1f(self.ambientStrengthLoc, self.ambient_strength);
            }

            // Set camera's view and projection for standard shader
            if (self.uViewLoc != -1) {
                glad.glUniformMatrix4fv(self.uViewLoc, 1, glad.GL_FALSE, &view_arr);
            }
            if (self.uProjectionLoc != -1) {
                glad.glUniformMatrix4fv(self.uProjectionLoc, 1, glad.GL_FALSE, &proj_arr);
            }

            // Set rendering mode to Standard
            self.rendering_mode = .Standard;

            // Do another pass through all nodes, but with standard shader active
            node_it = self.nodes.iterator();
            while (node_it.next()) |node_entry| {
                node_entry.value_ptr.*.update();
            }
        }

        // Now bind back to default framebuffer, just so we have a blank background for ImGui
        glad.glBindFramebuffer(glad.GL_FRAMEBUFFER, 0);
        glad.glViewport(0, 0, @intFromFloat(self.width), @intFromFloat(self.height));
        glad.glClearColor(0.15, 0.15, 0.15, 1.0);
        glad.glClear(glad.GL_COLOR_BUFFER_BIT | glad.GL_DEPTH_BUFFER_BIT);

        // The rest: ImGui rendering
        imgui.igRender();
        imgui.ImGui_ImplOpenGL3_RenderDrawData(imgui.igGetDrawData());

        glfw.glfwSwapBuffers(window);
        glfw.glfwPollEvents();

        // Frame timing and FPS calculation
        const frame_end = glfw.glfwGetTime();
        const frame_time = frame_end - frame_start;

        // Store frame time in circular buffer
        self.frame_times[self.frame_time_index] = frame_time * 1000.0; // Convert to ms
        self.frame_time_index = (self.frame_time_index + 1) % 120;

        // Calculate and print average frame time every second
        self.frame_count += 1;
        if (frame_end - self.last_fps_time >= 1.0) {
            var sum: f64 = 0;
            for (self.frame_times) |time| {
                sum += time;
            }
            const avg_frame_time = sum / @as(f64, @floatFromInt(self.frame_times.len));
            std.debug.print("Avg Frame Time: {d:.2}ms, FPS: {d:.1}\n", .{ avg_frame_time, 1000.0 / avg_frame_time });

            self.frame_count = 0;
            self.avg_frame_time = avg_frame_time;
            self.last_fps_time = frame_end;
        }
    }

    pub fn toggleSimulationMode(self: *Self) void {
        self.appState.simulation_mode = !self.appState.simulation_mode;

        if (self.appState.simulation_mode) {
            self.camera_manager.set_active("simulation_third_person");
        } else {
            self.camera_manager.set_active("free");
        }
    }

    pub fn processInput(self: *Self, debug: bool) void {
        _ = debug;

        const io = imgui.igGetIO();
        if (self.appState.menu) {
            if (io.*.WantCaptureKeyboard) return;
        }

        self.camera_manager.main_camera.?.process_key_input();
    }

    pub fn setAmbientColor(self: *Self, r: f32, g: f32, b: f32) void {
        self.ambient_color = .{ r, g, b };
    }

    pub fn setAmbientStrength(self: *Self, strength: f32) void {
        self.ambient_strength = std.math.clamp(strength, 0.0, 1.0);
    }
};

fn readShaderSource(comptime path: []const u8) ![]const u8 {
    const file: []const u8 = @embedFile(path);
    std.debug.print("Source Length: {}\n", .{file.len});
    return file;
}

fn compileShader(shaderType: u32, source: []const u8) !u32 {
    const shader = glad.glCreateShader(shaderType);
    if (shader == 0) {
        std.debug.print("Failed to compile shader\n", .{});
        return ShaderError.UnableToCreateShader;
    }

    const src_ptr: [*c]const u8 = @ptrCast(@alignCast(source.ptr));
    const src_len = source.len;

    std.debug.print("Compiling Shader...\n", .{});
    glad.glShaderSource(shader, 1, @ptrCast(&src_ptr), @ptrCast(&src_len));
    glad.glCompileShader(shader);

    // Check for compilation errors
    var success: u32 = 0;
    glad.glGetShaderiv(shader, glad.GL_COMPILE_STATUS, @alignCast(@ptrCast(&success)));
    if (success == 0) {
        var infoLog: [512]u8 = undefined;
        glad.glGetShaderInfoLog(shader, 512, null, &infoLog);
        std.debug.print("ERROR::SHADER::COMPILATION_FAILED\n{any}\n", .{infoLog});
        return ShaderError.ShaderCompilationFailed;
    }

    return shader;
}

pub fn createShaderProgram(comptime vertexPath: []const u8, comptime fragmentPath: []const u8) !u32 {
    std.debug.print("Initializing Vertex Shader...\n", .{});
    std.debug.print("Reading Vertex Shader from Source...\n", .{});
    const vertexSource = try readShaderSource(vertexPath);
    const vertexShader = compileShader(glad.GL_VERTEX_SHADER, vertexSource) catch |err| {
        std.debug.print("Failed to read vertex shader '{s}': {any}\n", .{ vertexPath, err });
        return ShaderError.ShaderCompilationFailed;
    };
    std.debug.print("\n", .{});

    std.debug.print("Initializing Fragment Shader...\n", .{});
    std.debug.print("Reading Fragment Shader from Source...\n", .{});
    const fragmentSource = try readShaderSource(fragmentPath);
    const fragmentShader = compileShader(glad.GL_FRAGMENT_SHADER, fragmentSource) catch |err| {
        std.debug.print("Failed to read fragment shader '{s}': {any}\n", .{ fragmentPath, err });
        return ShaderError.ShaderCompilationFailed;
    };
    std.debug.print("\n", .{});

    std.debug.print("Creating ShaderProgram...\n", .{});
    const shaderProgram = glad.glCreateProgram();
    if (shaderProgram == 0) {
        std.debug.print("Failed to create Shader Program\n", .{});
        return ShaderError.UnableToCreateProgram;
    }

    std.debug.print("Attaching ShaderProgram to openGL...\n", .{});
    glad.glAttachShader(shaderProgram, vertexShader);
    glad.glAttachShader(shaderProgram, fragmentShader);
    glad.glLinkProgram(shaderProgram);

    // Check for linking errors
    var success: u32 = 0;
    glad.glGetProgramiv(shaderProgram, glad.GL_LINK_STATUS, @ptrCast(@alignCast(&success)));

    if (success == 0) {
        var infoLog: [512]u8 = undefined;
        glad.glGetProgramInfoLog(shaderProgram, 512, null, &infoLog);
        std.debug.print("ERROR::PROGRAM::LINKING_FAILED\n{any}\n", .{infoLog});
        return ShaderError.ShaderLinkingFailed;
    }

    std.debug.print("Running cleanup on Shaders...\n", .{});
    // Shaders can be deleted after linking
    glad.glDeleteShader(vertexShader);
    glad.glDeleteShader(fragmentShader);

    return shaderProgram;
}

fn checkOpenGLError(caller: []const u8) void {
    var err: u32 = glad.glGetError();
    while (err != glad.GL_NO_ERROR) {
        std.debug.print("OpenGL Error: {x} from Caller: {s}\n", .{ err, caller });
        err = glad.glGetError();
    }
}

fn getCurrentMonitor(window: ?*glfw.struct_GLFWwindow) ?*glfw.GLFWmonitor {
    if (window == null) return null;

    var monitor_count: i32 = undefined;
    const monitors = glfw.glfwGetMonitors(&monitor_count);
    if (monitors == null or monitor_count == 0) return null;

    // Get window position
    var win_x: i32 = undefined;
    var win_y: i32 = undefined;
    glfw.glfwGetWindowPos(window, &win_x, &win_y);

    // Get window size
    var win_width: i32 = undefined;
    var win_height: i32 = undefined;
    glfw.glfwGetWindowSize(window, &win_width, &win_height);

    // Find the monitor that contains the window center
    const center_x = win_x + @divTrunc(win_width, 2);
    const center_y = win_y + @divTrunc(win_height, 2);

    var i: usize = 0;
    while (i < @as(i32, @intCast(monitor_count))) : (i += 1) {
        const mon = monitors[i];
        var mx: i32 = undefined;
        var my: i32 = undefined;
        var mw: i32 = undefined;
        var mh: i32 = undefined;
        glfw.glfwGetMonitorWorkarea(mon, &mx, &my, &mw, &mh);

        if (center_x >= mx and center_x < mx + mw and
            center_y >= my and center_y < my + mh)
        {
            return mon;
        }
    }

    // Default to primary monitor if no match
    return glfw.glfwGetPrimaryMonitor();
}

pub fn createWindow() !?*glfw.GLFWwindow {
    glfw.glfwWindowHint(glfw.GLFW_FOCUSED, glfw.GLFW_TRUE);
    glfw.glfwWindowHint(glfw.GLFW_FOCUS_ON_SHOW, glfw.GLFW_TRUE);
    glfw.glfwWindowHint(glfw.GLFW_CLIENT_API, glfw.GLFW_OPENGL_API);
    glfw.glfwWindowHint(glfw.GLFW_CONTEXT_CREATION_API, glfw.GLFW_NATIVE_CONTEXT_API);
    glfw.glfwWindowHint(glfw.GLFW_DOUBLEBUFFER, glfw.GLFW_TRUE);
    glfw.glfwWindowHint(glfw.GLFW_SRGB_CAPABLE, glfw.GLFW_TRUE);
    glfw.glfwWindowHint(glfw.GLFW_DECORATED, glfw.GLFW_TRUE); // Remove window decorations
    glfw.glfwWindowHint(glfw.GLFW_MAXIMIZED, glfw.GLFW_TRUE); // Start maximized

    const monitor = glfw.glfwGetPrimaryMonitor();
    const video_mode = glfw.glfwGetVideoMode(monitor);

    // Create window with monitor's resolution
    const width = video_mode.*.width;
    const height = video_mode.*.height;

    const window = glfw.glfwCreateWindow(width, height, "Drone Studio", null, null) orelse return null;

    // Position the window at the monitor's position
    var xpos: i32 = undefined;
    var ypos: i32 = undefined;
    glfw.glfwGetMonitorPos(monitor, &xpos, &ypos);
    glfw.glfwSetWindowPos(window, xpos, ypos);

    // Rest of your initialization code...
    glfw.glfwMakeContextCurrent(window);
    glfw.glfwSwapInterval(0);

    // Initialize OpenGL Loader
    std.debug.print("Loading Glad...\n", .{});
    if (glad.gladLoadGLLoader(@ptrCast(&glfw.glfwGetProcAddress)) == 0) {
        std.debug.print("Failed to initialize GLAD\n", .{});
        return GSLWError.FailedToInitialize;
    }

    std.debug.print("Initializing viewport...\n\n", .{});
    glad.glViewport(0, 0, width, height);
    glad.glEnable(glad.GL_DEPTH_TEST);
    glad.glDepthFunc(glad.GL_LESS);

    // Initialize ImGui
    _ = imgui.igCreateContext(null);
    const io = imgui.igGetIO();
    io.*.ConfigFlags |= imgui.ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.*.ConfigFlags |= imgui.ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls

    const optional_window: ?*imgui.GLFWwindow = @ptrCast(window);
    if (!imgui.ImGui_ImplGlfw_InitForOpenGL(optional_window, false)) {
        return null;
    }
    if (!imgui.ImGui_ImplOpenGL3_Init("#version 330")) {
        return null;
    }

    // Set up cursor mode BEFORE changing monitor settings
    glfw.glfwSetInputMode(window, glfw.GLFW_CURSOR, glfw.GLFW_CURSOR_DISABLED);
    if (glfw.glfwRawMouseMotionSupported() == glfw.GLFW_TRUE) {
        glfw.glfwSetInputMode(window, glfw.GLFW_RAW_MOUSE_MOTION, glfw.GLFW_TRUE);
    }

    // Force focus and raise window
    glfw.glfwFocusWindow(window);
    glfw.glfwShowWindow(window);

    return window;
}

pub fn cleanupImGui() void {
    imgui.ImGui_ImplOpenGL3_Shutdown();
    imgui.ImGui_ImplGlfw_Shutdown();
    imgui.igDestroyContext(null);
}

fn framebufferSizeCallback(window: ?*glfw.GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
    if (window == null) return;

    const user_ptr = glfw.glfwGetWindowUserPointer(window);
    if (user_ptr == null) {
        std.debug.print("Error: Window user pointer is null in framebufferSizeCallback\n", .{});
        return;
    }

    const scene = @as(*Scene, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

    glad.glViewport(0, 0, width, height);

    // Update Scene's width and height
    scene.width = @floatFromInt(width);
    scene.height = @floatFromInt(height);
}

fn mouseCallback(window: ?*glfw.struct_GLFWwindow, xpos: f64, ypos: f64) callconv(.C) void {
    if (window == null) return;

    // Only process mouse input if window is focused
    if (glfw.glfwGetWindowAttrib(window, glfw.GLFW_FOCUSED) != glfw.GLFW_TRUE) {
        return;
    }

    const scene = @as(*Scene, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));
    const io = imgui.igGetIO();

    if (scene.appState.first_mouse) {
        scene.appState.last_mouse_x = xpos;
        scene.appState.last_mouse_y = ypos;
        scene.appState.first_mouse = false;
        return;
    }

    const xoffset = xpos - scene.appState.last_mouse_x;
    const yoffset = scene.appState.last_mouse_y - ypos; // Reversed Y

    scene.appState.last_mouse_x = xpos;
    scene.appState.last_mouse_y = ypos;

    if (scene.appState.menu) {
        if (io.*.WantCaptureMouse) {
            // Let ImGui handle the mouse if it wants it
            scene.appState.last_mouse_x = xpos;
            scene.appState.last_mouse_y = ypos;
            return;
        }
    }

    if (scene.camera_manager.main_camera) |camera| {
        camera.process_mouse_input(xoffset, yoffset);
    }
}

// New method to handle drone mouse movement for yaw and pitch

fn mouseButtonCallback(window: ?*glfw.struct_GLFWwindow, button: c_int, action: c_int, mods: c_int) callconv(.C) void {
    if (window == null) return;

    const scene = @as(*Scene, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

    if (scene.appState.menu) {
        imgui.ImGui_ImplGlfw_MouseButtonCallback(@ptrCast(window), button, action, mods);

        const io = imgui.igGetIO();
        if (io.*.WantCaptureMouse) {
            return;
        }
    }
}

fn charCallback(window: ?*glfw.struct_GLFWwindow, character: c_uint) callconv(.C) void {
    if (window == null) return;

    const scene = @as(*Scene, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

    if (scene.appState.menu) {
        imgui.ImGui_ImplGlfw_CharCallback(@ptrCast(window), character);
    }
}

fn keyCallback(window: ?*glfw.struct_GLFWwindow, key: c_int, scancode: c_int, action: c_int, mods: c_int) callconv(.C) void {
    if (window == null) return;

    const scene = @as(*Scene, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));
    // const io = imgui.igGetIO();
    if (scene.appState.menu) {
        imgui.ImGui_ImplGlfw_KeyCallback(@ptrCast(window), key, scancode, action, mods);
    }

    if (key < 0 or key >= 1024) return;

    if (action == glfw.GLFW_PRESS) {
        scene.appState.keys[@intCast(key)] = true;
    } else if (action == glfw.GLFW_RELEASE) {
        scene.appState.keys[@intCast(key)] = false;
    }

    if (action == glfw.GLFW_PRESS or action == glfw.GLFW_REPEAT) {
        switch (key) {
            glfw.GLFW_KEY_ESCAPE => {
                scene.appState.menu = !scene.appState.menu;
                var cursor_mode = glfw.glfwGetInputMode(window, glfw.GLFW_CURSOR);

                switch (cursor_mode) {
                    glfw.GLFW_CURSOR_NORMAL => {
                        cursor_mode = glfw.GLFW_CURSOR_DISABLED;
                    },
                    glfw.GLFW_CURSOR_HIDDEN => {
                        cursor_mode = glfw.GLFW_CURSOR_NORMAL;
                    },
                    glfw.GLFW_CURSOR_DISABLED => {
                        cursor_mode = glfw.GLFW_CURSOR_NORMAL;
                    },
                    else => {},
                }

                glfw.glfwSetInputMode(window, glfw.GLFW_CURSOR, cursor_mode);
            },
            glfw.GLFW_KEY_M => {
                glfw.glfwIconifyWindow(window);
            },
            glfw.GLFW_KEY_RIGHT_BRACKET => {
                glfw.glfwDestroyWindow(window);
            },
            glfw.GLFW_KEY_P => {
                scene.appState.paused = !scene.appState.paused;
            },
            glfw.GLFW_KEY_F => {
                scene.appState.fly = !scene.appState.fly;
            },

            // New key bindings for simulation mode
            glfw.GLFW_KEY_V => {
                // Toggle simulation mode
                scene.toggleSimulationMode();
            },
            glfw.GLFW_KEY_C => {
                // Toggle first/third-person view in simulation mode
                if (scene.appState.simulation_mode) {
                    std.debug.print("Toggling from {s} => {s}\n", .{
                        if (scene.appState.first_person_view) "simulation_first_person" else "simulation_third_person",
                        if (scene.appState.first_person_view) "simulation_third_person" else "simulation_first_person",
                    });

                    if (scene.appState.first_person_view)
                        scene.camera_manager.set_active("simulation_third_person")
                    else
                        scene.camera_manager.set_active("simulation_first_person");

                    scene.appState.first_person_view = !scene.appState.first_person_view;
                }
            },

            else => {},
        }
    }
}

fn scrollCallback(window: ?*glfw.struct_GLFWwindow, xoffset: f64, yoffset: f64) callconv(.C) void {
    if (window == null) return;

    const scene = @as(*Scene, @ptrCast(@alignCast(glfw.glfwGetWindowUserPointer(window))));

    if (scene.appState.menu and imgui.igGetIO().*.WantCaptureMouse) {
        imgui.ImGui_ImplGlfw_ScrollCallback(@ptrCast(window), xoffset, yoffset);
        return;
    }

    const zoomSensitivity: f32 = 0.1;
    const newZoom = scene.appState.zoom - @as(f32, @floatCast(yoffset)) * zoomSensitivity * scene.appState.zoom;

    // Clamp the zoom level to prevent weird behavior at larger FOV's
    if (newZoom < 1.0) {
        scene.appState.zoom = 1.0;
    } else if (newZoom >= 120) {
        scene.appState.zoom = 120;
    } else {
        scene.appState.zoom = newZoom;
    }

    scene.camera_manager.main_camera.?.process_scroll_wheel(newZoom);
}

pub const TextureGenerator = struct {
    const Self = @This();
    count: c_int = 0,

    pub fn generateID(self: *Self) c_int {
        defer self.count += 1;
        if (self.count >= 32) {
            std.debug.print("Warning: Texture unit limit exceeded\n", .{});
            return @mod(self.count, 32);
        }
        return self.count;
    }
};

pub const FrameBuffer = struct {
    fbo: c_uint,
    texture: c_uint,
    depth_buffer: c_uint,
    width: c_int,
    height: c_int,

    pub fn init(width: c_int, height: c_int) !FrameBuffer {
        var fb: FrameBuffer = undefined;
        fb.width = width;
        fb.height = height;

        // Create framebuffer
        glad.glGenFramebuffers(1, &fb.fbo);
        glad.glBindFramebuffer(glad.GL_FRAMEBUFFER, fb.fbo);

        // Create texture to render to
        glad.glGenTextures(1, &fb.texture);
        glad.glBindTexture(glad.GL_TEXTURE_2D, fb.texture);
        glad.glTexImage2D(glad.GL_TEXTURE_2D, 0, glad.GL_RGB, width, height, 0, glad.GL_RGB, glad.GL_UNSIGNED_BYTE, null);
        glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MIN_FILTER, glad.GL_LINEAR);
        glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MAG_FILTER, glad.GL_LINEAR);
        glad.glBindTexture(glad.GL_TEXTURE_2D, 0);

        // Attach texture to framebuffer
        glad.glFramebufferTexture2D(glad.GL_FRAMEBUFFER, glad.GL_COLOR_ATTACHMENT0, glad.GL_TEXTURE_2D, fb.texture, 0);

        // Create depth buffer
        glad.glGenRenderbuffers(1, &fb.depth_buffer);
        glad.glBindRenderbuffer(glad.GL_RENDERBUFFER, fb.depth_buffer);
        glad.glRenderbufferStorage(glad.GL_RENDERBUFFER, glad.GL_DEPTH_COMPONENT, width, height);
        glad.glBindRenderbuffer(glad.GL_RENDERBUFFER, 0);

        // Attach depth buffer to framebuffer
        glad.glFramebufferRenderbuffer(glad.GL_FRAMEBUFFER, glad.GL_DEPTH_ATTACHMENT, glad.GL_RENDERBUFFER, fb.depth_buffer);

        // Check if framebuffer is complete
        if (glad.glCheckFramebufferStatus(glad.GL_FRAMEBUFFER) != glad.GL_FRAMEBUFFER_COMPLETE) {
            std.debug.print("Framebuffer is not complete!\n", .{});
            return error.FramebufferIncomplete;
        }

        // Unbind framebuffer
        glad.glBindFramebuffer(glad.GL_FRAMEBUFFER, 0);

        return fb;
    }

    pub fn deinit(self: *FrameBuffer) void {
        glad.glDeleteFramebuffers(1, &self.fbo);
        glad.glDeleteTextures(1, &self.texture);
        glad.glDeleteRenderbuffers(1, &self.depth_buffer);
    }
};

pub const Viewport = struct {
    const Self = @This();

    name: []const u8, // Name of the viewport
    fbo: FrameBuffer, // Framebuffer for rendering this viewport
    shader_program: c_uint,
    mesh: *Mesh, // Use the Mesh struct instead of raw VAO/VBO
    enabled: bool,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, width: i32, height: i32) !Self {
        // Create a copy of the name string
        const name_copy = try allocator.dupe(u8, name);

        // Initialize the framebuffer
        const fbo = try FrameBuffer.init(width, height);

        // Create shader program for rendering viewport
        const shader_program = try createShaderProgram("shaders/miniview_vertex.glsl", "shaders/miniview_fragment.glsl");

        // Create vertices for a quad using the Mesh.Vertex struct
        const vertices = [_]Mesh.Vertex{
            // Position                  Color                 Texture
            .{ .position = .{ -1.0, -1.0, 0.0 }, .color = .{ 1.0, 1.0, 1.0 }, .texture = .{ 0.0, 0.0 } },
            .{ .position = .{ 1.0, -1.0, 0.0 }, .color = .{ 1.0, 1.0, 1.0 }, .texture = .{ 1.0, 0.0 } },
            .{ .position = .{ 1.0, 1.0, 0.0 }, .color = .{ 1.0, 1.0, 1.0 }, .texture = .{ 1.0, 1.0 } },
            .{ .position = .{ -1.0, 1.0, 0.0 }, .color = .{ 1.0, 1.0, 1.0 }, .texture = .{ 0.0, 1.0 } },
        };

        // Create indices for the quad (two triangles)
        const indices = [_]u32{
            0, 1, 2, // First triangle
            2, 3, 0, // Second triangle
        };

        // Create a copy of the vertices and indices for the mesh
        const vertices_copy = try allocator.dupe(Mesh.Vertex, &vertices);
        const indices_copy = try allocator.dupe(u32, &indices);

        // Create a mesh for the quad
        const mesh = try Mesh.init(allocator, vertices_copy, indices_copy, Mesh.gen_draw(glad.GL_TRIANGLES));

        const viewport = Self{
            .name = name_copy,
            .fbo = fbo,
            .shader_program = shader_program,
            .mesh = mesh,
            .enabled = true,
            .allocator = allocator,
        };

        return viewport;
    }

    pub fn deinit(self: *Self) void {
        // Free OpenGL resources
        self.fbo.deinit();
        glad.glDeleteProgram(self.shader_program);

        // Free the mesh
        self.mesh.deinit();
        self.allocator.destroy(self.mesh);

        // Free the name
        self.allocator.free(self.name);
    }

    pub fn render(self: *Self, window_width: f32, window_height: f32) void {
        if (!self.enabled) return;

        // Calculate viewport dimensions in pixels
        const x = self.position[0] * window_width;
        const y = self.position[1] * window_height;
        const width = self.size[0] * window_width;
        const height = self.size[1] * window_height;

        // Save current OpenGL state
        var last_viewport: [4]c_int = undefined;
        glad.glGetIntegerv(glad.GL_VIEWPORT, &last_viewport);

        var last_program: c_int = 0;
        glad.glGetIntegerv(glad.GL_CURRENT_PROGRAM, &last_program);

        var last_blend_enabled: c_int = 0;
        glad.glGetIntegerv(glad.GL_BLEND, &last_blend_enabled);

        var last_depth_test_enabled: c_int = 0;
        glad.glGetIntegerv(glad.GL_DEPTH_TEST, &last_depth_test_enabled);

        // Disable depth test for 2D rendering
        glad.glDisable(glad.GL_DEPTH_TEST);

        // Enable blending for transparency
        glad.glEnable(glad.GL_BLEND);
        glad.glBlendFunc(glad.GL_SRC_ALPHA, glad.GL_ONE_MINUS_SRC_ALPHA);

        // Set viewport
        glad.glViewport(
            @intFromFloat(x),
            @intFromFloat(y),
            @intFromFloat(width),
            @intFromFloat(height),
        );

        // Use the shader program
        glad.glUseProgram(self.shader_program);

        // Bind texture
        glad.glActiveTexture(glad.GL_TEXTURE0);
        glad.glBindTexture(glad.GL_TEXTURE_2D, self.fbo.texture);

        // Set texture uniform
        const textureLoc = glad.glGetUniformLocation(self.shader_program, "viewTexture");
        if (textureLoc != -1) {
            glad.glUniform1i(textureLoc, 0);
        }

        // Draw the quad using the mesh's draw function
        self.mesh._draw(self.mesh);

        // Add border
        const border_width: f32 = 2.0;
        const border_color = imgui.igColorConvertFloat4ToU32(.{ .x = 1.0, .y = 1.0, .z = 1.0, .w = 0.8 });

        const draw_list = imgui.igGetWindowDrawList();
        imgui.ImDrawList_AddRect(
            draw_list,
            .{ .x = x, .y = y },
            .{ .x = x + width, .y = y + height },
            border_color,
            0.0,
            imgui.ImDrawFlags_None,
            border_width,
        );

        // Draw the Viewport name in the top-left corner
        const text_padding = 5.0;
        imgui.ImDrawList_AddText_Vec2(
            draw_list,
            .{ .x = x + text_padding, .y = y + text_padding },
            imgui.igColorConvertFloat4ToU32(.{ .x = 1.0, .y = 1.0, .z = 1.0, .w = 1.0 }),
            self.name.ptr,
            null,
        );

        // Restore previous OpenGL state
        if (last_depth_test_enabled == glad.GL_TRUE) {
            glad.glEnable(glad.GL_DEPTH_TEST);
        } else {
            glad.glDisable(glad.GL_DEPTH_TEST);
        }

        if (last_blend_enabled == glad.GL_TRUE) {
            glad.glEnable(glad.GL_BLEND);
        } else {
            glad.glDisable(glad.GL_BLEND);
        }

        glad.glViewport(
            last_viewport[0],
            last_viewport[1],
            last_viewport[2],
            last_viewport[3],
        );
        glad.glUseProgram(@intCast(last_program));
    }
};
