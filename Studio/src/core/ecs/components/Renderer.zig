// src/ecs/components/Renderer.zig
const std = @import("std");
const Core = @import("../Core.zig");
const Math = @import("../../Math.zig");
const SparseSet = @import("../SparseSet.zig").SparseSet;
const Transform = @import("../components/Transform.zig");
const ResourceManager = @import("../ResourceManager.zig");
const Mesh = @import("../../Mesh.zig");
const TextureUnit = @import("../../Node.zig").TextureUnit;
const gl = @import("../../bindings/gl.zig");
const Globals = @import("Globals.zig");
const ECSManager = @import("../ECSManager.zig");
const Viewports = @import("Viewports.zig");
const Camera = @import("Camera.zig");

const glad = gl.glad;
const Mat4 = Math.Mat4;
const Vec3 = Math.Vec3;
const Material = ResourceManager.MaterialVariant;
const ViewportComponent = Viewports.ViewportComponent;
const TransformComponent = Transform.TransformComponent;
const GlobalsComponent = Globals.GlobalsComponent;
const CameraComponent = Camera.CameraComponent;

pub const Renderable = struct {
    const Self = @This();

    mesh_name: [:0]const u8,
    material_name: ?[:0]const u8 = null,
    is_visible: bool = true,
    render_order: i32 = 0,
    cast_shadows: bool = false,
    receive_shadows: bool = false,

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
        };
    }

    pub fn deinit(self: *Self) void {
        self.render_queue.deinit();
    }

    pub fn update(self: *Self) !void {
        self.render_queue.clearRetainingCapacity();

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
            }
        }

        // Sort render queue to minimize state changes
        // std.sort.insertion(RenderCommand, self.render_queue.items, {}, RenderCommand.lessThan);

        var vp_it = self.viewport_components.iterator();
        while (vp_it.next()) |vp_entry| {
            const entity_id = vp_entry.entity_id;
            const viewport = vp_entry.component.vp;

            // Retrieve the actual Camera* from camera_manager:
            const camera = self.camera_components.get(entity_id);
            const transform = self.transform_components.get(entity_id);

            if (camera != null and transform != null) {
                const cam = camera.?;
                const tf = transform.?;

                // Skip disabled viewports
                if (!cam.active) continue;

                // Bind FBO
                glad.glBindFramebuffer(glad.GL_FRAMEBUFFER, viewport.fbo.fbo);
                glad.glViewport(0, 0, viewport.fbo.width, viewport.fbo.height);

                // Clear
                glad.glClearColor(0.1, 0.1, 0.1, 1.0);
                glad.glClear(glad.GL_COLOR_BUFFER_BIT | glad.GL_DEPTH_BUFFER_BIT);

                const view_matrix = cam.view(tf);
                const projection_matrix = cam.projection();
                const camera_position = tf.position;

                // std.debug.print("View Mat {any}\nProj Mat: {any}\nCamera Position: {any}\n", .{ view_matrix, projection_matrix, camera_position });

                // Process render queue
                const current_shader_id: c_uint = 0;
                var prev_material: ?[]const u8 = null;

                // std.debug.print("Render Queue Size: {d}\n", .{self.render_queue.items.len});

                for (self.render_queue.items, 0..) |*cmd, idx| {
                    _ = idx;
                    // std.debug.print("Render Command {d}:\n\tShaderID: {d}\n\tShaderName: {s}\n\tTransform: {any}\n\tEntityID: {d}\n\tRenderable: {any}\n", .{
                    //     idx,
                    //     cmd.shader_id,
                    //     cmd.shader_name orelse "null",
                    //     cmd.transform.world_transform,
                    //     cmd.entity_id.id,
                    //     cmd.renderer,
                    // });

                    // Get mesh from resource manager
                    if (self.world.resource_manager.meshes.get(cmd.renderer.mesh_name)) |mesh_resource| {
                        const mesh = mesh_resource.mesh;

                        // Get material if available
                        if (cmd.renderer.material_name) |material_name| {
                            if (prev_material == null or !std.mem.eql(u8, prev_material.?, material_name)) {
                                if (self.world.resource_manager.materials.getPtr(material_name)) |material_resource| {
                                    self.applyMaterial(material_resource, cmd.transform, &view_matrix, &projection_matrix, &camera_position);
                                }

                                prev_material = material_name;
                            }
                        } else {
                            // No material - use a default shader if needed
                            if (current_shader_id == 0) {
                                var shader = self.world.resource_manager.getShader("standard_shader") orelse return;

                                glad.glUseProgram(shader.program_id);

                                // Set transformation matrices
                                const uModelLoc = shader.getorPutUniformLocation("uModel");
                                const uViewLoc = shader.getorPutUniformLocation("uView");
                                const uProjectionLoc = shader.getorPutUniformLocation("uProjection");

                                if (uModelLoc != -1) {
                                    glad.glUniformMatrix4fv(uModelLoc, 1, glad.GL_FALSE, &cmd.transform.world_transform.to_array());
                                }

                                if (uViewLoc != -1) {
                                    glad.glUniformMatrix4fv(uViewLoc, 1, glad.GL_FALSE, &view_matrix.to_array());
                                }

                                if (uProjectionLoc != -1) {
                                    glad.glUniformMatrix4fv(uProjectionLoc, 1, glad.GL_FALSE, &projection_matrix.to_array());
                                }
                            }
                        }

                        // Draw the mesh
                        mesh._draw(mesh);

                        // Reset per-material state if needed
                        glad.glDisable(glad.GL_BLEND);
                        glad.glEnable(glad.GL_CULL_FACE);
                        glad.glCullFace(glad.GL_BACK);
                    }
                }
            }

            // std.debug.print("{s}\n{s}\n", .{ "=" ** 80, "=" ** 80 });
        }

        // Now bind back to default framebuffer, just so we have a blank background for ImGui
        glad.glBindFramebuffer(glad.GL_FRAMEBUFFER, 0);
        glad.glViewport(0, 0, 1920, 1080);
        glad.glClearColor(0.15, 0.15, 0.15, 1.0);
        glad.glClear(glad.GL_COLOR_BUFFER_BIT | glad.GL_DEPTH_BUFFER_BIT);
    }

    pub fn applyMaterial(self: *Self, material_resource: *ResourceManager.MaterialResource, transform: TransformComponent, view_matrix: *const Mat4, projection_matrix: *const Mat4, view_position: *const [3]f32) void {
        const shader_name = material_resource.shader_ref orelse switch (material_resource.material) {
            .PBR => "pbr_shader",
            .Phong => "standard_shader",
        };
        var shader = self.world.resource_manager.getShader(shader_name) orelse return;

        // Activate the shader program
        glad.glUseProgram(shader.program_id);

        // Set transformation matrices
        const uModelLoc = shader.uniforms.get("uModel") orelse -1;
        const uViewLoc = shader.uniforms.get("uView") orelse -1;
        const uProjectionLoc = shader.uniforms.get("uProjection") orelse -1;
        const viewPosLoc = shader.uniforms.get("viewPos") orelse -1;
        const useTexureLoc = shader.uniforms.get("useTexture") orelse -1;
        // const uNormalMatrixLoc = shader.uniforms.get("uNormalMatrix") orelse -1;

        // std.debug.print("View Mat {any}\nProj Mat: {any}\nCamera Position: {any}\n", .{ view_matrix, projection_matrix, view_position });

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

        if (useTexureLoc != -1) {
            glad.glUniform1i(useTexureLoc, @intFromBool(true));
        }

        // if (uNormalMatrixLoc != -1) {
        //     // Calculate normal matrix (inverse transpose of the model matrix's upper 3x3 part)
        //     var normal_matrix = transform.world_transform.to_mat3().inverse().transpose();
        //     glad.glUniformMatrix3fv(uNormalMatrixLoc, 1, glad.GL_FALSE, &normal_matrix.to_array()[0]);
        // }

        // Apply material based on type
        switch (material_resource.material) {
            .PBR => |pbr| {
                // Apply PBR material properties using cached uniform locations
                const baseColorFactorLoc = shader.uniforms.get("baseColorFactor") orelse -1;
                const metallicFactorLoc = shader.uniforms.get("metallicFactor") orelse -1;
                const roughnessFactorLoc = shader.uniforms.get("roughnessFactor") orelse -1;
                const emissiveFactorLoc = shader.uniforms.get("emissiveFactor") orelse -1;
                const emissiveStrengthLoc = shader.uniforms.get("emissiveStrength") orelse -1;
                const alphaCutoffLoc = shader.uniforms.get("alphaCutoff") orelse -1;
                const alphaModeEnumLoc = shader.uniforms.get("alphaModeEnum") orelse -1;
                const doubleSidedLoc = shader.uniforms.get("doubleSided") orelse -1;
                const specularColorLoc = shader.uniforms.get("specularColor") orelse -1;
                const specularStrengthLoc = shader.uniforms.get("specularStrength") orelse -1;

                // Apply all PBR parameters
                if (baseColorFactorLoc != -1) glad.glUniform4fv(baseColorFactorLoc, 1, &pbr.data.baseColorFactor);
                if (metallicFactorLoc != -1) glad.glUniform1f(metallicFactorLoc, pbr.data.metallicFactor);
                if (roughnessFactorLoc != -1) glad.glUniform1f(roughnessFactorLoc, pbr.data.roughnessFactor);
                if (emissiveFactorLoc != -1) glad.glUniform3fv(emissiveFactorLoc, 1, &pbr.data.emissiveFactor);
                if (emissiveStrengthLoc != -1) glad.glUniform1f(emissiveStrengthLoc, pbr.data.emissiveStrength);
                if (alphaCutoffLoc != -1) glad.glUniform1f(alphaCutoffLoc, pbr.data.alphaCutoff);
                if (alphaModeEnumLoc != -1) glad.glUniform1i(alphaModeEnumLoc, @intFromEnum(pbr.data.alphaMode));
                if (doubleSidedLoc != -1) glad.glUniform1i(doubleSidedLoc, @intFromBool(pbr.data.doubleSided));
                if (specularColorLoc != -1) glad.glUniform3fv(specularColorLoc, 1, &pbr.data.specularColor);
                if (specularStrengthLoc != -1) glad.glUniform1f(specularStrengthLoc, pbr.data.specularStrength);

                // Texture presence flags
                const hasBaseColorTexLoc = shader.uniforms.get("hasBaseColorTexture") orelse -1;
                const hasNormalTexLoc = shader.uniforms.get("hasNormalTexture") orelse -1;
                const hasMetallicRoughnessTexLoc = shader.uniforms.get("hasMetallicRoughnessTexture") orelse -1;
                const hasOcclusionTexLoc = shader.uniforms.get("hasOcclusionTexture") orelse -1;
                const hasEmissiveTexLoc = shader.uniforms.get("hasEmissiveTexture") orelse -1;
                const hasSpecularTexLoc = shader.uniforms.get("hasSpecularTexture") orelse -1;

                // Set texture presence flags
                if (hasBaseColorTexLoc != -1) glad.glUniform1i(hasBaseColorTexLoc, @intFromBool(pbr.textures.baseColor != null));
                if (hasNormalTexLoc != -1) glad.glUniform1i(hasNormalTexLoc, @intFromBool(pbr.textures.normal != null));
                if (hasMetallicRoughnessTexLoc != -1) glad.glUniform1i(hasMetallicRoughnessTexLoc, @intFromBool(pbr.textures.metallicRoughness != null));
                if (hasOcclusionTexLoc != -1) glad.glUniform1i(hasOcclusionTexLoc, @intFromBool(pbr.textures.occlusion != null));
                if (hasEmissiveTexLoc != -1) glad.glUniform1i(hasEmissiveTexLoc, @intFromBool(pbr.textures.emissive != null));
                if (hasSpecularTexLoc != -1) glad.glUniform1i(hasSpecularTexLoc, @intFromBool(pbr.textures.specular != null));

                // Texture sampler locations
                const baseColorTexLoc = shader.uniforms.get("baseColorTexture") orelse -1;
                const normalTexLoc = shader.uniforms.get("normalTexture") orelse -1;
                const metallicRoughnessTexLoc = shader.uniforms.get("metallicRoughnessTexture") orelse -1;
                const occlusionTexLoc = shader.uniforms.get("occlusionTexture") orelse -1;
                const emissiveTexLoc = shader.uniforms.get("emissiveTexture") orelse -1;
                const specularTexLoc = shader.uniforms.get("specularTexture") orelse -1;

                // Bind PBR textures
                if (pbr.textures.baseColor) |tex_id| {
                    glad.glActiveTexture(TextureUnit.BaseColor.glValue());
                    glad.glBindTexture(glad.GL_TEXTURE_2D, tex_id);
                    if (baseColorTexLoc != -1) glad.glUniform1i(baseColorTexLoc, TextureUnit.BaseColor.index());
                }

                if (pbr.textures.normal) |tex_id| {
                    glad.glActiveTexture(TextureUnit.NormalMap.glValue());
                    glad.glBindTexture(glad.GL_TEXTURE_2D, tex_id);
                    if (normalTexLoc != -1) glad.glUniform1i(normalTexLoc, TextureUnit.NormalMap.index());
                }

                if (pbr.textures.metallicRoughness) |tex_id| {
                    glad.glActiveTexture(TextureUnit.MetallicRoughness.glValue());
                    glad.glBindTexture(glad.GL_TEXTURE_2D, tex_id);
                    if (metallicRoughnessTexLoc != -1) glad.glUniform1i(metallicRoughnessTexLoc, TextureUnit.MetallicRoughness.index());
                }

                if (pbr.textures.occlusion) |tex_id| {
                    glad.glActiveTexture(TextureUnit.Occlusion.glValue());
                    glad.glBindTexture(glad.GL_TEXTURE_2D, tex_id);
                    if (occlusionTexLoc != -1) glad.glUniform1i(occlusionTexLoc, TextureUnit.Occlusion.index());
                }

                if (pbr.textures.emissive) |tex_id| {
                    glad.glActiveTexture(TextureUnit.Emissive.glValue());
                    glad.glBindTexture(glad.GL_TEXTURE_2D, tex_id);
                    if (emissiveTexLoc != -1) glad.glUniform1i(emissiveTexLoc, TextureUnit.Emissive.index());
                }

                if (pbr.textures.specular) |tex_id| {
                    glad.glActiveTexture(TextureUnit.Specular.glValue());
                    glad.glBindTexture(glad.GL_TEXTURE_2D, tex_id);
                    if (specularTexLoc != -1) glad.glUniform1i(specularTexLoc, TextureUnit.Specular.index());
                }

                // Apply rendering state based on material properties
                if (pbr.data.alphaMode == .BLEND) {
                    glad.glEnable(glad.GL_BLEND);
                    glad.glBlendFunc(glad.GL_SRC_ALPHA, glad.GL_ONE_MINUS_SRC_ALPHA);
                } else if (pbr.data.alphaMode == .MASK) {
                    glad.glEnable(glad.GL_BLEND);
                    glad.glBlendFunc(glad.GL_SRC_ALPHA, glad.GL_ONE_MINUS_SRC_ALPHA);
                    glad.glUniform1f(shader.getorPutUniformLocation("alphaCutoff"), pbr.data.alphaCutoff);
                } else {
                    glad.glDisable(glad.GL_BLEND);
                }

                if (pbr.data.doubleSided) {
                    glad.glDisable(glad.GL_CULL_FACE);
                } else {
                    glad.glEnable(glad.GL_CULL_FACE);
                    glad.glCullFace(glad.GL_BACK);
                }
            },
            .Phong => |phong| {
                // Apply Phong material properties
                const ambientColorLoc = shader.uniforms.get("ambientColor") orelse -1;
                const diffuseColorLoc = shader.uniforms.get("diffuseColor") orelse -1;
                const specularColorLoc = shader.uniforms.get("specularColor") orelse -1;
                const shininessLoc = shader.uniforms.get("shininess") orelse -1;

                if (ambientColorLoc != -1) glad.glUniform3fv(ambientColorLoc, 1, &phong.data.ambientColor);
                if (diffuseColorLoc != -1) glad.glUniform4fv(diffuseColorLoc, 1, &phong.data.diffuseColor);
                if (specularColorLoc != -1) glad.glUniform3fv(specularColorLoc, 1, &phong.data.specularColor);
                if (shininessLoc != -1) glad.glUniform1f(shininessLoc, phong.data.shininess);

                // Texture presence flags
                const hasDiffuseTexLoc = shader.uniforms.get("hasDiffuseTexture") orelse -1;
                const hasSpecularTexLoc = shader.uniforms.get("hasSpecularTexture") orelse -1;
                const hasNormalTexLoc = shader.uniforms.get("hasNormalTexture") orelse -1;

                if (hasDiffuseTexLoc != -1) glad.glUniform1i(hasDiffuseTexLoc, @intFromBool(phong.data.diffuseTexture != null));
                if (hasSpecularTexLoc != -1) glad.glUniform1i(hasSpecularTexLoc, @intFromBool(phong.data.specularTexture != null));
                if (hasNormalTexLoc != -1) glad.glUniform1i(hasNormalTexLoc, @intFromBool(phong.data.normalTexture != null));

                // Texture sampler locations
                const diffuseTexLoc = shader.uniforms.get("diffuseTexture") orelse -1;
                const specularTexLoc = shader.uniforms.get("specularTexture") orelse -1;
                const normalTexLoc = shader.uniforms.get("normalTexture") orelse -1;

                // Bind Phong textures
                if (phong.data.diffuseTexture) |tex_id| {
                    glad.glActiveTexture(glad.GL_TEXTURE0);
                    glad.glBindTexture(glad.GL_TEXTURE_2D, tex_id);
                    if (diffuseTexLoc != -1) glad.glUniform1i(diffuseTexLoc, 0);
                }

                if (phong.data.specularTexture) |tex_id| {
                    glad.glActiveTexture(glad.GL_TEXTURE1);
                    glad.glBindTexture(glad.GL_TEXTURE_2D, tex_id);
                    if (specularTexLoc != -1) glad.glUniform1i(specularTexLoc, 1);
                }

                if (phong.data.normalTexture) |tex_id| {
                    glad.glActiveTexture(glad.GL_TEXTURE2);
                    glad.glBindTexture(glad.GL_TEXTURE_2D, tex_id);
                    if (normalTexLoc != -1) glad.glUniform1i(normalTexLoc, 2);
                }

                // Apply default rendering state for Phong
                glad.glEnable(glad.GL_CULL_FACE);
                glad.glCullFace(glad.GL_BACK);

                // Enable blending only if alpha < 1.0
                if (phong.data.diffuseColor[3] < 1.0) {
                    glad.glEnable(glad.GL_BLEND);
                    glad.glBlendFunc(glad.GL_SRC_ALPHA, glad.GL_ONE_MINUS_SRC_ALPHA);
                } else {
                    glad.glDisable(glad.GL_BLEND);
                }
            },
        }
    }

    pub fn debug(self: *Self) void {
        var it = self.renderables.iterator();
        while (it.next()) |entry| {
            std.debug.print("{any}\n", .{entry.component});
        }
    }
};
