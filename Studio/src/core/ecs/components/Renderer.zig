// src/ecs/components/Renderer.zig
const std = @import("std");
const Core = @import("../Core.zig");
const Math = @import("../../Math.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const Transform = @import("../components/Transform.zig");
const ResourceManager = @import("../ResourceManager.zig");
const Mesh = @import("../../Mesh.zig");
const gl = @import("../../bindings/gl.zig");
const Globals = @import("Globals.zig");
const ECSManager = @import("../ECSManager.zig");
const Viewports = @import("Viewports.zig");
const Camera = @import("Camera.zig");
const RenderProfiler = @import("../../debug/RenderProfiler.zig");

const glad = gl.glad;
const Mat4 = Math.Mat4;
const Vec3 = Math.Vec3;
const Material = ResourceManager.MaterialVariant;
const ViewportComponent = Viewports.ViewportComponent;
const TransformComponent = Transform.TransformComponent;
const GlobalsComponent = Globals.GlobalsComponent;
const CameraComponent = Camera.CameraComponent;

/// Visibility layer constants for filtering renderables per viewport
pub const VisibilityLayer = struct {
    pub const DEFAULT: u32 = 1 << 0; // Regular scene geometry
    pub const DEBUG: u32 = 1 << 1; // Debug visualizations (frustums, paths, landmarks)
    pub const UI: u32 = 1 << 2; // UI elements
    pub const ALL: u32 = 0xFFFFFFFF; // Sees everything
};

pub const Renderable = struct {
    const Self = @This();

    mesh_name: [:0]const u8,
    material_name: ?[:0]const u8 = null,
    is_visible: bool = true,
    render_order: i32 = 0,
    cast_shadows: bool = false,
    receive_shadows: bool = false,
    visibility_mask: u32 = VisibilityLayer.ALL, // Which layers this renderable belongs to

    pub fn init(allocator: std.mem.Allocator, mesh_name: []const u8) !Self {
        return .{
            .mesh_name = try allocator.dupeZ(u8, mesh_name),
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.mesh_name);

        if (self.material_name) |mat_name| {
            allocator.free(mat_name);
        }
    }

    pub fn setMaterial(self: *Self, allocator: std.mem.Allocator, material_name: ?[:0]const u8) !void {
        if (self.material_name) |old_mat| {
            allocator.free(old_mat);
        }

        self.material_name = try allocator.dupeZ(u8, material_name orelse "default");
    }

    pub fn attach(self: *Renderable, ecs: *ECSManager, eid: Core.EntityID) !void {
        try ecs.renderer_components.add(eid, self.*);
    }

    pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        const mat_str = if (self.material_name) |mat| mat else "null";

        try writer.print("Renderable(\n" ++
            "\tmesh_name=\"{s}\",\n" ++
            "\tmaterial_name=\"{s}\",\n" ++
            "\tis_visible={any},\n" ++
            "\trender_order={d},\n" ++
            "\tcast_shadows={any},\n" ++
            "\treceive_shadows={any},\n" ++
            ")", .{
            self.mesh_name,
            mat_str,
            self.is_visible,
            self.render_order,
            self.cast_shadows,
            self.receive_shadows,
        });
    }
};

pub const RenderSystem = struct {
    const Self = @This();

    world: *Core.World,
    globals: *GlobalsComponent,
    transform_components: *SparseSet(TransformComponent),
    camera_components: *SparseSet(CameraComponent),
    viewport_components: *SparseSet(ViewportComponent),
    renderables: *SparseSet(Renderable),
    render_queue: std.ArrayList(RenderCommand),
    profiler: RenderProfiler.Profiler,

    const RenderCommand = struct {
        shader_id: c_uint,
        shader_name: ?[:0]const u8,
        entity_id: Core.EntityID,
        transform: TransformComponent,
        renderer: Renderable,

        pub fn lessThan(_: void, a: RenderCommand, b: RenderCommand) bool {
            // First sort by material/shader to minimize state changes
            if (a.shader_name == null and b.shader_name != null) {
                return true;
            }
            if (a.shader_name != null and b.shader_name == null) {
                return false;
            }
            if (a.shader_name != null and b.shader_name != null) {
                // Compare the shader resource strings directly
                const a_shader = a.shader_name.?;
                const b_shader = b.shader_name.?;

                const cmp = std.mem.lessThan(u8, a_shader, b_shader);
                if (!std.mem.eql(u8, a_shader, b_shader)) {
                    return cmp;
                }
            }

            // Then sort by material
            if (a.renderer.material_name == null and b.renderer.material_name != null) {
                return true;
            }
            if (a.renderer.material_name != null and b.renderer.material_name == null) {
                return false;
            }
            if (a.renderer.material_name != null and b.renderer.material_name != null) {
                // Compare the material resource strings directly
                const a_material = a.renderer.material_name.?;
                const b_material = b.renderer.material_name.?;

                const cmp = std.mem.lessThan(u8, a_material, b_material);
                if (!std.mem.eql(u8, a_material, b_material)) {
                    return cmp;
                }
            }

            // Sort by render order
            if (a.renderer.render_order != b.renderer.render_order) {
                return a.renderer.render_order < b.renderer.render_order;
            }

            // Sort by mesh
            const cmp = std.mem.lessThan(u8, a.renderer.mesh_name, b.renderer.mesh_name);

            return cmp;
        }
    };

    pub fn init(
        allocator: std.mem.Allocator,
        world: *Core.World,
        globals: *GlobalsComponent,
        transform_components: *SparseSet(TransformComponent),
        camera_components: *SparseSet(CameraComponent),
        viewport_components: *SparseSet(ViewportComponent),
        renderables: *SparseSet(Renderable),
    ) !Self {
        return .{
            .world = world,
            .globals = globals,
            .transform_components = transform_components,
            .camera_components = camera_components,
            .viewport_components = viewport_components,
            .renderables = renderables,
            .render_queue = try std.ArrayList(RenderCommand).initCapacity(allocator, 64),
            .profiler = RenderProfiler.Profiler.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.render_queue.deinit();
    }

    pub fn update(self: *Self) !void {
        self.profiler.beginFrame();
        defer self.profiler.endFrame();

        self.render_queue.clearRetainingCapacity();

        {
            var gather_scope = self.profiler.sectionScope(.gather_commands);
            defer gather_scope.end();

            var renderer_iter = self.renderables.iterator();
            while (renderer_iter.next()) |tuple| {
                const entity_id = tuple.entity_id;
                const renderer = tuple.component.*;

                // Skip invisible objects
                if (!renderer.is_visible) continue;

                if (self.transform_components.get(entity_id)) |transform| {
                    var material_resource: ?*ResourceManager.MaterialResource = null;
                    if (renderer.material_name) |material_name| {
                        material_resource = self.world.resource_manager.materials.getPtr(material_name);
                    }

                    var shader_id: c_uint = 0;
                    var shader_name: ?[:0]const u8 = null;

                    // Check for material-specific shader
                    if (material_resource != null and material_resource.?.shader_ref != null) {
                        shader_name = material_resource.?.shader_ref;
                        if (shader_name) |name| {
                            if (self.world.resource_manager.shaders.get(name)) |shader| {
                                shader_id = shader.program_id;
                            }
                        }
                    }

                    try self.render_queue.append(.{
                        .entity_id = entity_id,
                        .renderer = renderer,
                        .transform = transform.*,
                        .shader_id = shader_id,
                        .shader_name = shader_name,
                    });

                    self.profiler.trackCommand();
                }
            }
        }

        // Sort render queue to minimize state changes
        // std.sort.insertion(RenderCommand, self.render_queue.items, {}, RenderCommand.lessThan);

        // Upload material buffer to GPU (if dirty) and bind SSBOs
        self.world.resource_manager.material_buffer.upload();
        self.world.resource_manager.material_buffer.bind();

        var vp_it = self.viewport_components.iterator();
        while (vp_it.next()) |vp_entry| {
            const entity_id = vp_entry.entity_id;
            const viewport = vp_entry.component.vp;

            const cam = self.camera_components.get(entity_id) orelse continue;
            const tf = self.transform_components.get(entity_id) orelse continue;

            // Bind FBO and clear
            glad.glBindFramebuffer(glad.GL_FRAMEBUFFER, viewport.fbo.fbo);
            glad.glViewport(0, 0, viewport.fbo.width, viewport.fbo.height);
            glad.glClearColor(0.1, 0.1, 0.1, 1.0);
            glad.glClear(glad.GL_COLOR_BUFFER_BIT | glad.GL_DEPTH_BUFFER_BIT);

            const view_matrix = cam.view(tf);
            const projection_matrix = cam.projection();
            const camera_position = tf.position;

            var prev_material: ?[]const u8 = null;
            const viewport_visibility = vp_entry.component.visibility_mask;

            for (self.render_queue.items) |*cmd| {
                // Skip if renderable not visible to this viewport's layer mask
                if (cmd.renderer.visibility_mask & viewport_visibility == 0) continue;

                const mesh_resource = self.world.resource_manager.meshes.get(cmd.renderer.mesh_name) orelse continue;
                const mesh = mesh_resource.mesh;

                // Setup material/shader
                const material_changed = blk: {
                    if (prev_material) |prev| {
                        if (cmd.renderer.material_name) |curr| {
                            break :blk !std.mem.eql(u8, prev, curr);
                        }
                    }
                    break :blk true;
                };

                if (cmd.renderer.material_name) |material_name| {
                    const material_resource = self.world.resource_manager.materials.getPtr(material_name) orelse continue;

                    if (material_changed) {
                        var material_scope = self.profiler.sectionScope(.material_setup);
                        defer material_scope.end();
                        _ = self.applyMaterial(material_resource, cmd.transform, &view_matrix, &projection_matrix, &camera_position);
                        prev_material = material_name;
                    } else {
                        // Same material - just update model matrix
                        self.updateModelMatrix(material_resource, &cmd.transform.world_transform);
                    }
                } else {
                    // No material - use default shader
                    self.applyDefaultShader(&cmd.transform.world_transform, &view_matrix, &projection_matrix, mesh);
                }

                // Draw
                self.profiler.trackVaoBind(mesh.meta.VAO);
                {
                    var draw_scope = self.profiler.sectionScope(.draw_submission);
                    defer draw_scope.end();
                    self.profiler.trackDraw();

                    mesh._draw(mesh);
                }

                // Reset state
                glad.glDisable(glad.GL_BLEND);
                glad.glEnable(glad.GL_CULL_FACE);
                glad.glCullFace(glad.GL_BACK);
            }
        }

        // Now bind back to default framebuffer, just so we have a blank background for ImGui
        glad.glBindFramebuffer(glad.GL_FRAMEBUFFER, 0);
        glad.glViewport(0, 0, 1920, 1080);
        glad.glClearColor(0.15, 0.15, 0.15, 1.0);
        glad.glClear(glad.GL_COLOR_BUFFER_BIT | glad.GL_DEPTH_BUFFER_BIT);
    }

    /// Apply material for rendering. With SSBOs, this now only sets:
    /// - Shader program
    /// - Transform matrices (uModel, uView, uProjection)
    /// - Camera position (viewPos)
    /// - Material index (uMaterialIndex) for SSBO lookup
    /// - Render state (blend, cull face)
    pub fn applyMaterial(
        self: *Self,
        material_resource: *ResourceManager.MaterialResource,
        transform: TransformComponent,
        view_matrix: *const Mat4,
        projection_matrix: *const Mat4,
        view_position: *const [3]f32,
    ) c_uint {
        const shader_name = material_resource.shader_ref orelse switch (material_resource.material) {
            .PBR => "pbr_shader",
            .Phong => "standard_shader",
        };
        const shader = self.world.resource_manager.getShader(shader_name) orelse return 0;

        self.profiler.trackMaterialBind(@intFromPtr(material_resource));
        self.profiler.trackShaderBind(@intCast(shader.program_id));

        // Activate shader
        glad.glUseProgram(shader.program_id);

        // Set transformation matrices
        const uModelLoc = shader.uniforms.get("uModel") orelse -1;
        const uViewLoc = shader.uniforms.get("uView") orelse -1;
        const uProjectionLoc = shader.uniforms.get("uProjection") orelse -1;
        const viewPosLoc = shader.uniforms.get("viewPos") orelse -1;
        const uMaterialIndexLoc = shader.uniforms.get("uMaterialIndex") orelse -1;

        if (uModelLoc != -1) {
            glad.glUniformMatrix4fv(uModelLoc, 1, glad.GL_FALSE, &transform.world_transform.to_array());
        }
        if (uViewLoc != -1) {
            glad.glUniformMatrix4fv(uViewLoc, 1, glad.GL_FALSE, &view_matrix.to_array());
        }
        if (uProjectionLoc != -1) {
            glad.glUniformMatrix4fv(uProjectionLoc, 1, glad.GL_FALSE, &projection_matrix.to_array());
        }
        if (viewPosLoc != -1) {
            glad.glUniform3f(viewPosLoc, view_position[0], view_position[1], view_position[2]);
        }

        // Set material index for SSBO lookup
        if (uMaterialIndexLoc != -1) {
            if (material_resource.gpu_index) |idx| {
                glad.glUniform1ui(uMaterialIndexLoc, idx);
            }
        }

        // Disable vertex color mode (we're using material)
        const uUseVertexColorLoc = shader.uniforms.get("uUseVertexColor") orelse -1;
        if (uUseVertexColorLoc != -1) glad.glUniform1i(uUseVertexColorLoc, 0);

        // Apply render state based on material type
        switch (material_resource.material) {
            .PBR => |pbr| {
                // Blend state
                if (pbr.data.alphaMode == .BLEND) {
                    glad.glEnable(glad.GL_BLEND);
                    glad.glBlendFunc(glad.GL_SRC_ALPHA, glad.GL_ONE_MINUS_SRC_ALPHA);
                } else {
                    glad.glDisable(glad.GL_BLEND);
                }

                // Cull state
                if (pbr.data.doubleSided) {
                    glad.glDisable(glad.GL_CULL_FACE);
                } else {
                    glad.glEnable(glad.GL_CULL_FACE);
                    glad.glCullFace(glad.GL_BACK);
                }
            },
            .Phong => |phong| {
                // Cull state
                glad.glEnable(glad.GL_CULL_FACE);
                glad.glCullFace(glad.GL_BACK);

                // Blend state
                if (phong.data.diffuseColor[3] < 1.0) {
                    glad.glEnable(glad.GL_BLEND);
                    glad.glBlendFunc(glad.GL_SRC_ALPHA, glad.GL_ONE_MINUS_SRC_ALPHA);
                } else {
                    glad.glDisable(glad.GL_BLEND);
                }
            },
        }

        return shader.program_id;
    }

    /// Update only the model matrix for same-material draws
    fn updateModelMatrix(self: *Self, material_resource: *ResourceManager.MaterialResource, world_transform: *const Mat4) void {
        const shader_name = material_resource.shader_ref orelse switch (material_resource.material) {
            .PBR => "pbr_shader",
            .Phong => "standard_shader",
        };
        const shader = self.world.resource_manager.getShader(shader_name) orelse return;
        const uModelLoc = shader.uniforms.get("uModel") orelse return;
        glad.glUniformMatrix4fv(uModelLoc, 1, glad.GL_FALSE, &world_transform.to_array());
    }

    /// Apply default shader for entities without materials (uses vertex colors)
    fn applyDefaultShader(self: *Self, world_transform: *const Mat4, view_matrix: *const Mat4, projection_matrix: *const Mat4, mesh: *Mesh) void {
        const shader = self.world.resource_manager.getShader("standard_shader") orelse return;

        self.profiler.trackShaderBind(@intCast(shader.program_id));
        glad.glUseProgram(shader.program_id);

        const uModelLoc = shader.uniforms.get("uModel") orelse -1;
        const uViewLoc = shader.uniforms.get("uView") orelse -1;
        const uProjectionLoc = shader.uniforms.get("uProjection") orelse -1;
        const uUseVertexColorLoc = shader.uniforms.get("uUseVertexColor") orelse -1;

        if (uModelLoc != -1) glad.glUniformMatrix4fv(uModelLoc, 1, glad.GL_FALSE, &world_transform.to_array());
        if (uViewLoc != -1) glad.glUniformMatrix4fv(uViewLoc, 1, glad.GL_FALSE, &view_matrix.to_array());
        if (uProjectionLoc != -1) glad.glUniformMatrix4fv(uProjectionLoc, 1, glad.GL_FALSE, &projection_matrix.to_array());
        if (uUseVertexColorLoc != -1) glad.glUniform1i(uUseVertexColorLoc, 1);

        // Set instancing flags based on mesh draw function
        const is_points: c_int = if (mesh._draw == Mesh.instanced_point_draw) 1 else 0;
        const is_lines: c_int = if (mesh._draw == Mesh.instanced_line_draw) 1 else 0;
        const uInstancedKeypointsLoc = shader.uniforms.get("uInstancedKeypoints") orelse -1;
        const uInstancedLinesLoc = shader.uniforms.get("uInstancedLines") orelse -1;
        if (uInstancedKeypointsLoc != -1) glad.glUniform1i(uInstancedKeypointsLoc, is_points);
        if (uInstancedLinesLoc != -1) glad.glUniform1i(uInstancedLinesLoc, is_lines);
    }

    pub fn debug(self: *Self) void {
        var it = self.renderables.iterator();
        while (it.next()) |entry| {
            std.debug.print("{any}\n", .{entry.component});
        }
    }

    /// Set visibility for an entity and all its children recursively
    pub fn setVisibility(self: *Self, entity_id: Core.EntityID, visible: bool) void {
        // Set visibility for this entity if it has a renderable
        if (self.renderables.get(entity_id)) |renderable_ptr| {
            renderable_ptr.is_visible = visible;
        }

        // Set visibility for all children if this entity has a transform
        if (self.transform_components.get(entity_id)) |transform| {
            for (transform.children.items) |child_id| {
                self.setVisibility(child_id, visible);
            }
        }
    }
};
