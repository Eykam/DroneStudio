// src/GLTF.zig
const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const ImageLoader = @import("Image.zig");
const Mesh = @import("Mesh.zig");
const Math = @import("Math.zig");
const gl = @import("bindings/gl.zig");

const glad = gl.glad;
const Vec3 = Math.Vec3;
const Mat4 = Math.Mat4;
const Quaternion = Math.Quaternion;

pub const GLTFError = error{
    InvalidFile,
    InvalidFormat,
    MissingRequiredField,
    UnsupportedFeature,
    DataError,
    ResourceNotFound,
    BufferCreationFailed,
};

pub const Document = struct {
    asset: Asset,
    scene: ?usize = null,
    scenes: ?[]Scene = null,
    nodes: ?[]GLTF_Node = null,
    meshes: ?[]GLTF_Mesh = null,
    accessors: ?[]Accessor = null,
    bufferViews: ?[]BufferView = null,
    buffers: ?[]Buffer = null,
    materials: ?[]Material = null,
    textures: ?[]Texture = null,
    images: ?[]Image = null,
    samplers: ?[]Sampler = null,
};

pub const Asset = struct {
    version: []const u8,
    generator: ?[]const u8 = null,
    copyright: ?[]const u8 = null,
    minVersion: ?[]const u8 = null,
};

pub const Scene = struct {
    name: ?[]const u8 = null,
    nodes: ?[]usize = null,
};

pub const GLTF_Node = struct {
    name: ?[]const u8 = null,
    mesh: ?usize = null,
    children: ?[]usize = null,
    translation: ?[3]f32 = null,
    rotation: ?[4]f32 = null,
    scale: ?[3]f32 = null,
    matrix: ?[16]f32 = null,
    camera: ?usize = null,
    skin: ?usize = null,
    weights: ?[]f32 = null,
};

pub const GLTF_Mesh = struct {
    name: ?[]const u8 = null,
    primitives: []Primitive,
    weights: ?[]f32 = null,
};

pub const Primitive = struct {
    attributes: Attributes,
    indices: ?usize = null,
    material: ?usize = null,
    mode: ?u32 = 4, // Default is TRIANGLES (4)
    targets: ?[]json.Value = null,
};

pub const Attributes = struct {
    POSITION: ?usize = null,
    NORMAL: ?usize = null,
    TANGENT: ?usize = null,
    TEXCOORD_0: ?usize = null,
    TEXCOORD_1: ?usize = null,
    COLOR_0: ?usize = null,
    JOINTS_0: ?usize = null,
    WEIGHTS_0: ?usize = null,
};

pub const Accessor = struct {
    name: ?[]const u8 = null,
    bufferView: ?usize = null,
    byteOffset: ?usize = 0,
    componentType: u32,
    normalized: ?bool = false,
    count: usize,
    type: ?[]const u8 = null,
    max: ?[]f32 = null,
    min: ?[]f32 = null,
    sparse: ?json.Value = null,
};

pub const BufferView = struct {
    name: ?[]const u8 = null,
    buffer: usize,
    byteOffset: ?usize = 0,
    byteLength: usize,
    byteStride: ?usize = null,
    target: ?u32 = null,
};

pub const Buffer = struct {
    name: ?[]const u8 = null,
    uri: ?[]const u8 = null,
    byteLength: usize,
};

pub const PbrSpecularGlossiness = struct {
    diffuseFactor: ?[4]f32 = null,
    diffuseTexture: ?TextureInfo = null,
    specularFactor: ?[3]f32 = null,
    glossinessFactor: ?f32 = null,
    specularGlossinessTexture: ?TextureInfo = null,
};

pub const EmissiveStrength = struct {
    emissiveStrength: ?f32 = null,
};

pub const Specular = struct {
    specularFactor: ?f32 = null,
    specularColorFactor: ?[3]f32 = null,
    specularTexture: ?TextureInfo = null,
};

pub const MaterialExtensions = struct {
    KHR_materials_pbrSpecularGlossiness: ?PbrSpecularGlossiness = null,
    KHR_materials_emissive_strength: ?EmissiveStrength = null,
    KHR_materials_specular: ?Specular = null,
};

pub const Material = struct {
    name: ?[]const u8 = null,
    pbrMetallicRoughness: ?PbrMetallicRoughness = null,
    normalTexture: ?TextureInfo = null,
    occlusionTexture: ?TextureInfo = null,
    emissiveTexture: ?TextureInfo = null,
    emissiveFactor: ?[3]f32 = null,
    alphaMode: ?[]const u8 = null,
    alphaCutoff: ?f32 = null,
    doubleSided: ?bool = null,
    extensions: ?MaterialExtensions = null,
};

pub const PbrMetallicRoughness = struct {
    baseColorFactor: ?[4]f32 = null,
    baseColorTexture: ?TextureInfo = null,
    metallicFactor: ?f32 = null,
    roughnessFactor: ?f32 = null,
    metallicRoughnessTexture: ?TextureInfo = null,
};

pub const TextureInfo = struct {
    index: usize,
    texCoord: ?usize = 0,
};

pub const Texture = struct {
    name: ?[]const u8 = null,
    sampler: ?usize = null,
    source: ?usize = null,
};

pub const TextureTypes = enum {
    baseColor,
    normal,
    metallicRoughness,
    occlusion,
    emissive,
    specular,
};

pub const Image = struct {
    name: ?[]const u8 = null,
    uri: ?[]const u8 = null,
    mimeType: ?[]const u8 = null,
    bufferView: ?usize = null,
};

pub const Sampler = struct {
    name: ?[]const u8 = null,
    magFilter: ?u32 = null,
    minFilter: ?u32 = null,
    wrapS: ?u32 = null,
    wrapT: ?u32 = null,
};

pub const ModelResource = struct {
    model_id: [:0]const u8,
    entities: []EntityInfo,
    allocator: std.mem.Allocator,

    pub const EntityInfo = struct {
        name: ?[:0]const u8,
        mesh_name: ?[:0]const u8,
        material_name: ?[:0]const u8,
        local_transformation: ?Mat4,
        translation: ?[3]f32,
        rotation: ?[4]f32,
        scale: ?[3]f32,
        parent_idx: ?usize,
        children: []usize,
    };

    pub fn deinit(self: *ModelResource) void {
        self.allocator.free(self.model_id);
        for (self.entities) |entity| {
            if (entity.name) |name| self.allocator.free(name);
            if (entity.mesh_name) |name| self.allocator.free(name);
            if (entity.material_name) |name| self.allocator.free(name);
            self.allocator.free(entity.children);
        }
        self.allocator.free(self.entities);
        self.allocator.destroy(self);
    }
};

pub fn createModelResource(
    allocator: std.mem.Allocator,
    model_id: [:0]const u8,
    gltf: *GLTF,
) !*ModelResource {
    const scene_idx = gltf.document.value.scene orelse 0;
    if (gltf.document.value.scenes == null or gltf.document.value.scenes.?.len == 0 or gltf.document.value.nodes == null) {
        const empty_model = try allocator.create(ModelResource);
        empty_model.* = .{
            .model_id = try allocator.dupeZ(u8, model_id),
            .entities = &.{},
            .allocator = allocator,
        };
        return empty_model;
    }

    const gltf_scene = gltf.document.value.scenes.?[scene_idx];
    var entity_list = std.ArrayList(ModelResource.EntityInfo).init(allocator);

    // Recursively build up node/primitive entities
    if (gltf_scene.nodes) |top_level_nodes| {
        for (top_level_nodes) |node_idx| {
            try processNodeAndChildren(allocator, gltf, model_id, node_idx, null, &entity_list);
        }
    }

    const model_resource = try allocator.create(ModelResource);
    model_resource.model_id = try allocator.dupeZ(u8, model_id);
    model_resource.allocator = allocator;
    model_resource.entities = try allocator.alloc(ModelResource.EntityInfo, entity_list.items.len);

    for (entity_list.items, 0..) |info, i| {
        model_resource.entities[i] = info;
    }

    entity_list.deinit();

    return model_resource;
}

fn processNodeAndChildren(
    allocator: std.mem.Allocator,
    gltf: *GLTF,
    model_id: []const u8,
    node_idx: usize,
    parent_idx: ?usize,
    entity_list: *std.ArrayList(ModelResource.EntityInfo),
) !void {
    if (gltf.document.value.nodes == null or node_idx >= gltf.document.value.nodes.?.len) {
        return;
    }

    const gltf_node = gltf.document.value.nodes.?[node_idx];

    // -------------------------------
    // 1) Create an entity for this node
    // -------------------------------
    var node_entity_info = ModelResource.EntityInfo{
        .name = null,
        .mesh_name = null,
        .material_name = null,
        .local_transformation = null,
        .translation = null,
        .rotation = null,
        .scale = null,
        .parent_idx = parent_idx,
        .children = &.{},
    };

    // Copy the node's name if any
    if (gltf_node.name) |node_name| {
        node_entity_info.name = try allocator.dupeZ(u8, node_name);
    }

    // If the node has a matrix, use it directly
    if (gltf_node.matrix) |mat_array| {
        // TODO: Investigate matrix format - GLTF spec uses column-major matrices,
        // need to verify if our Mat4 expects row-major or column-major format.
        // Models appearing upside-down might be related to coordinate system differences
        // (Y-up vs Z-up) or matrix interpretation issues.
        node_entity_info.local_transformation = Mat4.from_array(mat_array);
    } else {
        // Otherwise, store TRS
        if (gltf_node.translation) |t| {
            node_entity_info.translation = t;
        }
        if (gltf_node.rotation) |r| {
            node_entity_info.rotation = r;
        }
        if (gltf_node.scale) |s| {
            node_entity_info.scale = s;
        }
    }

    // Add this node-entity to the list
    const this_node_entity_idx = entity_list.items.len; // index of the new entity
    try entity_list.append(node_entity_info);

    // -------------------------------
    // 2) If the node has a mesh, create child-entities for each primitive
    // -------------------------------
    if (gltf_node.mesh) |mesh_idx| {
        if (gltf.document.value.meshes) |all_meshes| {
            if (mesh_idx < all_meshes.len) {
                const gltf_mesh = all_meshes[mesh_idx];
                // The glTF mesh can have multiple primitives each with its own material
                for (gltf_mesh.primitives, 0..) |primitive, prim_i| {
                    // We'll create a child-entity that references the parent's transform
                    // (so the child is effectively "in the same spot"),
                    // but each primitive can have a unique mesh_name + material_name.

                    var prim_entity_info = ModelResource.EntityInfo{
                        .name = null,
                        .mesh_name = null,
                        .material_name = null,
                        .local_transformation = Mat4.identity(),
                        .translation = null,
                        .rotation = null,
                        .scale = null,
                        .parent_idx = this_node_entity_idx, // parent is the node entity
                        .children = &.{},
                    };

                    // Build a unique mesh name for the ECS or ResourceManager
                    const prim_mesh_name = if (gltf_mesh.name) |mesh_name|
                        try std.fmt.allocPrintZ(allocator, "{s}_{s}_prim_{d}", .{ model_id, mesh_name, prim_i })
                    else
                        try std.fmt.allocPrintZ(allocator, "{s}_mesh_{d}_prim_{d}", .{ model_id, mesh_idx, prim_i });
                    prim_entity_info.mesh_name = prim_mesh_name;

                    // Material (if any)
                    if (primitive.material) |material_idx| {
                        if (gltf.document.value.materials) |all_mats| {
                            if (material_idx < all_mats.len) {
                                const mat_def = all_mats[material_idx];
                                const mat_name = if (mat_def.name) |mat_n|
                                    try std.fmt.allocPrintZ(allocator, "{s}_{s}", .{ model_id, mat_n })
                                else
                                    try std.fmt.allocPrintZ(allocator, "{s}_material_{d}", .{ model_id, material_idx });

                                prim_entity_info.material_name = mat_name;
                            }
                        }
                    }

                    // Append the child-entity
                    try entity_list.append(prim_entity_info);
                }
            }
        }
    }

    if (gltf_node.children) |child_indices| {
        for (child_indices) |child_idx| {
            try processNodeAndChildren(allocator, gltf, model_id, child_idx, this_node_entity_idx, entity_list);
        }
    }
}

/// Basic type definitions for glTF
pub const GLTF = struct {
    allocator: Allocator,
    raw_json: []const u8,
    document: json.Parsed(Document),
    buffers: std.ArrayList([]const u8),
    base_path: []const u8,
    textures: std.ArrayList(Mesh.TextureID),
    materials: std.ArrayList(Mesh.Material),

    pub fn init(allocator: Allocator, filepath: []const u8) !*GLTF {
        // Determine base path for resolving relative paths
        var base_path = try allocator.dupe(u8, filepath);
        // Find the last slash
        var last_slash: ?usize = null;
        for (base_path, 0..) |c, i| {
            if (c == '/' or c == '\\') {
                last_slash = i;
            }
        }
        if (last_slash) |idx| {
            base_path = base_path[0 .. idx + 1];
        } else {
            base_path = "";
        }

        // Load the file
        const file = try std.fs.cwd().openFile(filepath, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const raw_json = try allocator.alloc(u8, file_size);
        const bytes_read = try file.readAll(raw_json);
        if (bytes_read != file_size) {
            return GLTFError.InvalidFile;
        }

        // Parse JSON
        const document = try json.parseFromSlice(
            Document,
            allocator,
            raw_json,
            .{
                .ignore_unknown_fields = true,
            },
        );

        // Create the GLTF instance
        const gltf = try allocator.create(GLTF);
        gltf.* = GLTF{
            .allocator = allocator,
            .raw_json = raw_json,
            .document = document,
            .buffers = std.ArrayList([]const u8).init(allocator),
            .base_path = base_path,
            .textures = std.ArrayList(Mesh.TextureID).init(allocator),
            .materials = std.ArrayList(Mesh.Material).init(allocator),
        };

        // Load buffers
        try gltf.loadBuffers();

        return gltf;
    }

    pub fn deinit(self: *GLTF) void {
        // Free buffer data
        for (self.buffers.items) |buffer| {
            self.allocator.free(buffer);
        }

        self.document.deinit();
        self.buffers.deinit();
        self.textures.deinit();
        self.materials.deinit();

        // Free raw JSON and base path
        self.allocator.free(self.raw_json);
        self.allocator.free(self.base_path);

        // Free self
        self.allocator.destroy(self);
    }

    fn loadBuffers(self: *GLTF) !void {
        if (self.document.value.buffers) |buffers| {
            try self.buffers.ensureTotalCapacity(buffers.len);

            for (buffers) |buffer_def| {
                if (buffer_def.uri) |uri| {
                    if (std.mem.startsWith(u8, uri, "data:")) {
                        // Handle data URI
                        // For simplicity, we're only supporting base64 encoded data
                        const data_start = std.mem.indexOf(u8, uri, "base64,") orelse {
                            return GLTFError.InvalidFormat;
                        };
                        const base64_data = uri[data_start + 7 ..];

                        // Allocate buffer for decoded data
                        const base64_standard = std.base64.standard;
                        const decoder = base64_standard.Decoder;
                        const decoded_size = try decoder.calcSizeForSlice(base64_data);
                        const decoded = try self.allocator.alloc(u8, decoded_size);

                        // Decode base64 data
                        try decoder.decode(decoded, base64_data);
                        try self.buffers.append(decoded);
                    } else {
                        // Handle file URI
                        const full_path = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ self.base_path, uri });
                        defer self.allocator.free(full_path);

                        const buffer_file = std.fs.cwd().openFile(full_path, .{}) catch {
                            return GLTFError.ResourceNotFound;
                        };
                        defer buffer_file.close();

                        const buffer_size = try buffer_file.getEndPos();
                        const buffer_data = try self.allocator.alloc(u8, buffer_size);
                        const bytes_read = try buffer_file.readAll(buffer_data);

                        if (bytes_read != buffer_size) {
                            self.allocator.free(buffer_data);
                            return GLTFError.DataError;
                        }

                        try self.buffers.append(buffer_data);
                    }
                } else {
                    // GLB-stored buffer (not implemented in this basic version)
                    return GLTFError.UnsupportedFeature;
                }
            }
        }
    }

    pub fn loadMesh(self: *GLTF, allocator: Allocator, mesh_idx: usize) !?*Mesh {
        if (self.document.value.meshes == null or mesh_idx >= self.document.value.meshes.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const mesh_def = self.document.value.meshes.?[mesh_idx];

        // For simplicity, we only load the first primitive
        if (mesh_def.primitives.len == 0) {
            return null;
        }

        const primitive = mesh_def.primitives[0];

        // Check for positions attribute
        if (primitive.attributes.POSITION == null) {
            return GLTFError.MissingRequiredField;
        }

        // Load positions
        const positions = try self.loadAccessorVec3(primitive.attributes.POSITION.?);

        // Load colors (if available)
        const colors = if (primitive.attributes.COLOR_0) |color_accessor|
            try self.loadAccessorVec3(color_accessor)
        else
            null;

        // Load texture coordinates (if available)
        const texcoords = if (primitive.attributes.TEXCOORD_0) |texcoord_accessor|
            try self.loadAccessorVec2(texcoord_accessor)
        else
            null;

        // Load indices (if available)
        const indices = if (primitive.indices) |indices_accessor|
            try self.loadAccessorIndices(indices_accessor)
        else
            null;

        const normals = if (primitive.attributes.NORMAL) |normal_accessor|
            try self.loadAccessorVec3(normal_accessor)
        else
            null;

        // Load tangents (if available) - note: glTF tangents are vec4 with w component indicating handedness
        const tangents = if (primitive.attributes.TANGENT) |tangent_accessor|
            try self.loadAccessorVec4(tangent_accessor)
        else
            null;

        // Create vertices array
        const vertex_count = positions.len;
        var vertices = try allocator.alloc(Mesh.Vertex, vertex_count);

        for (0..vertex_count) |i| {
            vertices[i] = Mesh.Vertex{
                .position = positions[i],
                .color = if (colors) |c| c[i] else .{ 1.0, 1.0, 1.0 },
                .texture = if (texcoords) |t| t[i] else null,
                .alpha = null,
                .normal = if (normals) |n| n[i] else null,
                .tangent = if (tangents) |t| .{ t[i][0], t[i][1], t[i][2] } else null,
                .bitangent = null,
            };
        }

        if (normals != null and texcoords != null and tangents == null) {
            Mesh.calculateTangents(vertices, indices);
        } else if (tangents != null) {
            // If we have tangents but no bitangents, calculate bitangents
            // Bitangent = cross(Normal, Tangent) * sign
            for (0..vertex_count) |i| {
                if (vertices[i].normal != null and vertices[i].tangent != null) {
                    const n = vertices[i].normal.?;
                    const t = vertices[i].tangent.?;

                    // For glTF, the tangent's W component indicates handedness
                    // We use the last element of the tangent vec4 for this
                    const handedness = if (tangents) |tans| tans[i][3] else 1.0;

                    // B = cross(N, T) * handedness
                    vertices[i].bitangent = .{
                        (n[1] * t[2] - n[2] * t[1]) * handedness,
                        (n[2] * t[0] - n[0] * t[2]) * handedness,
                        (n[0] * t[1] - n[1] * t[0]) * handedness,
                    };
                }
            }
        }

        // Create mesh
        const mesh = try Mesh.init(allocator, vertices, indices, Mesh.gen_draw(.triangles));

        // Load texture if present
        if (primitive.material != null) {
            const material_idx = primitive.material.?;
            if (material_idx < self.materials.items.len) {
                mesh.material = self.materials.items[material_idx];

                // Set up flags based on material
                if (mesh.flags == null) {
                    mesh.flags = Mesh.MeshFlags{};
                }

                // Check if we have any textures
                const has_textures = (mesh.material.textures.baseColor != null or
                    mesh.material.textures.normal != null or
                    mesh.material.textures.metallicRoughness != null or
                    mesh.material.textures.occlusion != null or
                    mesh.material.textures.emissive != null or
                    mesh.material.textures.specular != null);

                mesh.flags.?.use_texture = has_textures;
                mesh.flags.?.use_pbr = true;
            }
        }

        return mesh;
    }

    pub fn loadBufferViewImage(self: *GLTF, allocator: Allocator, buffer_view_idx: usize) !*ImageLoader.Image {
        if (self.document.value.bufferViews == null or buffer_view_idx >= self.document.value.bufferViews.?.len) {
            return error.ImageErrorResourceNotFound;
        }

        const buffer_view = self.document.value.bufferViews.?[buffer_view_idx];

        const buffer_idx = buffer_view.buffer;
        if (buffer_idx >= self.buffers.items.len) {
            return error.ImageErrorResourceNotFound;
        }

        const buffer = self.buffers.items[buffer_idx];
        const offset = buffer_view.byteOffset orelse 0;
        const length = buffer_view.byteLength;

        const image_data = buffer[offset .. offset + length];

        return try ImageLoader.Image.loadFromMemory(allocator, image_data);
    }

    fn loadAccessorVec4(self: *GLTF, accessor_idx: usize) ![][4]f32 {
        if (self.document.value.accessors == null or accessor_idx >= self.document.value.accessors.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const accessor = self.document.value.accessors.?[accessor_idx];

        // Check type compatibility
        if (accessor.type == null or !std.mem.eql(u8, accessor.type.?, "VEC4") or
            accessor.componentType != 5126)
        { // 5126 is GL_FLOAT
            return GLTFError.UnsupportedFeature;
        }

        // Get buffer view
        if (accessor.bufferView == null) {
            return GLTFError.MissingRequiredField;
        }

        const buffer_view_idx = accessor.bufferView.?;
        if (self.document.value.bufferViews == null or buffer_view_idx >= self.document.value.bufferViews.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const buffer_view = self.document.value.bufferViews.?[buffer_view_idx];

        // Get buffer
        const buffer_idx = buffer_view.buffer;
        if (buffer_idx >= self.buffers.items.len) {
            return GLTFError.ResourceNotFound;
        }

        const buffer = self.buffers.items[buffer_idx];
        const offset = buffer_view.byteOffset orelse 0;
        const accessor_offset = accessor.byteOffset orelse 0;
        const stride = buffer_view.byteStride orelse (4 * @sizeOf(f32));

        // Read data
        const count = accessor.count;
        var result = try self.allocator.alloc([4]f32, count);

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const base_pos = offset + accessor_offset + (i * stride);

            // Safely read each float component individually
            var j: usize = 0;
            while (j < 4) : (j += 1) {
                const float_pos = base_pos + (j * @sizeOf(f32));

                // Ensure we don't read beyond the buffer
                if (float_pos + @sizeOf(f32) > buffer.len) {
                    self.allocator.free(result);
                    return error.InvalidData;
                }

                // Read 4 bytes and convert to f32
                const bytes = buffer[float_pos .. float_pos + @sizeOf(f32)];
                result[i][j] = std.mem.bytesToValue(f32, bytes[0..4]);
            }
        }

        return result;
    }

    fn loadAccessorVec3(self: *GLTF, accessor_idx: usize) ![][3]f32 {
        if (self.document.value.accessors == null or accessor_idx >= self.document.value.accessors.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const accessor = self.document.value.accessors.?[accessor_idx];

        // Check type compatibility
        if (accessor.type == null or !std.mem.eql(u8, accessor.type.?, "VEC3") or
            accessor.componentType != 5126)
        { // 5126 is GL_FLOAT
            return GLTFError.UnsupportedFeature;
        }

        // Get buffer view
        if (accessor.bufferView == null) {
            return GLTFError.MissingRequiredField;
        }

        const buffer_view_idx = accessor.bufferView.?;
        if (self.document.value.bufferViews == null or buffer_view_idx >= self.document.value.bufferViews.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const buffer_view = self.document.value.bufferViews.?[buffer_view_idx];

        // Get buffer
        const buffer_idx = buffer_view.buffer;
        if (buffer_idx >= self.buffers.items.len) {
            return GLTFError.ResourceNotFound;
        }

        const buffer = self.buffers.items[buffer_idx];
        const offset = buffer_view.byteOffset orelse 0;
        const accessor_offset = accessor.byteOffset orelse 0;
        const stride = buffer_view.byteStride orelse (3 * @sizeOf(f32));

        // Read data
        const count = accessor.count;
        var result = try self.allocator.alloc([3]f32, count);

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const base_pos = offset + accessor_offset + (i * stride);

            // Safely read each float component individually
            var j: usize = 0;
            while (j < 3) : (j += 1) {
                const float_pos = base_pos + (j * @sizeOf(f32));

                // Ensure we don't read beyond the buffer
                if (float_pos + @sizeOf(f32) > buffer.len) {
                    self.allocator.free(result);
                    return error.InvalidData;
                }

                // Read 4 bytes and convert to f32
                const bytes = buffer[float_pos .. float_pos + @sizeOf(f32)];
                result[i][j] = std.mem.bytesToValue(f32, bytes[0..4]);
            }
        }

        return result;
    }

    fn loadAccessorVec2(self: *GLTF, accessor_idx: usize) ![][2]f32 {
        if (self.document.value.accessors == null or accessor_idx >= self.document.value.accessors.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const accessor = self.document.value.accessors.?[accessor_idx];

        // Check type compatibility
        if (accessor.type == null or !std.mem.eql(u8, accessor.type.?, "VEC2") or
            accessor.componentType != 5126)
        { // 5126 is GL_FLOAT
            return GLTFError.UnsupportedFeature;
        }

        // Get buffer view
        if (accessor.bufferView == null) {
            return GLTFError.MissingRequiredField;
        }

        const buffer_view_idx = accessor.bufferView.?;
        if (self.document.value.bufferViews == null or buffer_view_idx >= self.document.value.bufferViews.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const buffer_view = self.document.value.bufferViews.?[buffer_view_idx];

        // Get buffer
        const buffer_idx = buffer_view.buffer;
        if (buffer_idx >= self.buffers.items.len) {
            return GLTFError.ResourceNotFound;
        }

        const buffer = self.buffers.items[buffer_idx];
        const offset = buffer_view.byteOffset orelse 0;
        const accessor_offset = accessor.byteOffset orelse 0;
        const stride = buffer_view.byteStride orelse (2 * @sizeOf(f32));

        // Read data
        const count = accessor.count;
        var result = try self.allocator.alloc([2]f32, count);

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const base_pos = offset + accessor_offset + (i * stride);

            // Safely read each float component individually
            var j: usize = 0;
            while (j < 2) : (j += 1) {
                const float_pos = base_pos + (j * @sizeOf(f32));

                // Ensure we don't read beyond the buffer
                if (float_pos + @sizeOf(f32) > buffer.len) {
                    self.allocator.free(result);
                    return error.InvalidData;
                }

                // Read 4 bytes and convert to f32
                const bytes = buffer[float_pos .. float_pos + @sizeOf(f32)];
                result[i][j] = std.mem.bytesToValue(f32, bytes[0..4]);
            }
        }

        return result;
    }

    fn loadAccessorIndices(self: *GLTF, accessor_idx: usize) ![]u32 {
        if (self.document.value.accessors == null or accessor_idx >= self.document.value.accessors.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const accessor = self.document.value.accessors.?[accessor_idx];

        // Check type compatibility
        if (accessor.type == null or !std.mem.eql(u8, accessor.type.?, "SCALAR")) {
            return GLTFError.UnsupportedFeature;
        }

        // Get buffer view
        if (accessor.bufferView == null) {
            return GLTFError.MissingRequiredField;
        }

        const buffer_view_idx = accessor.bufferView.?;
        if (self.document.value.bufferViews == null or buffer_view_idx >= self.document.value.bufferViews.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const buffer_view = self.document.value.bufferViews.?[buffer_view_idx];

        // Get buffer
        const buffer_idx = buffer_view.buffer;
        if (buffer_idx >= self.buffers.items.len) {
            return GLTFError.ResourceNotFound;
        }

        const buffer = self.buffers.items[buffer_idx];
        const offset = buffer_view.byteOffset orelse 0;
        const accessor_offset = accessor.byteOffset orelse 0;

        // Read data
        const count = accessor.count;
        var result = try self.allocator.alloc(u32, count);

        var i: usize = 0;
        var byte_pos = offset + accessor_offset;

        switch (accessor.componentType) {
            5121 => { // GL_UNSIGNED_BYTE
                while (i < count) : (i += 1) {
                    if (byte_pos >= buffer.len) {
                        self.allocator.free(result);
                        return error.InvalidData;
                    }

                    result[i] = @intCast(buffer[byte_pos]);
                    byte_pos += 1;
                }
            },
            5123 => { // GL_UNSIGNED_SHORT
                while (i < count) : (i += 1) {
                    if (byte_pos + @sizeOf(u16) > buffer.len) {
                        self.allocator.free(result);
                        return error.InvalidData;
                    }

                    // Read 2 bytes and convert to u16
                    const bytes = buffer[byte_pos .. byte_pos + @sizeOf(u16)];
                    result[i] = @intCast(std.mem.bytesToValue(u16, bytes[0..2]));
                    byte_pos += 2;
                }
            },
            5125 => { // GL_UNSIGNED_INT
                while (i < count) : (i += 1) {
                    if (byte_pos + @sizeOf(u32) > buffer.len) {
                        self.allocator.free(result);
                        return error.InvalidData;
                    }

                    // Read 4 bytes and convert to u32
                    const bytes = buffer[byte_pos .. byte_pos + @sizeOf(u32)];
                    result[i] = std.mem.bytesToValue(u32, bytes[0..4]);
                    byte_pos += 4;
                }
            },
            else => return GLTFError.UnsupportedFeature,
        }

        return result;
    }
};
