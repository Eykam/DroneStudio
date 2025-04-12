// src/GLTF.zig
const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const ImageLoader = @import("Image.zig");
const Node = @import("Node.zig");
const Mesh = @import("Mesh.zig");
const Math = @import("Math.zig");
const Pipeline = @import("Pipeline.zig");
const Shape = @import("Shape.zig");
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

// Public helper function to load a glTF file and add it to a scene
pub fn loadToScene(allocator: Allocator, scene: *Pipeline.Scene, filepath: []const u8, name: []const u8) !void {
    std.debug.print("Loading glTF file: {s}\n", .{filepath});

    // Parse the glTF file
    var gltf = try GLTF.init(allocator, filepath);
    defer gltf.deinit();

    // Load the default scene from the glTF file
    const root_node = try gltf.loadScene(allocator, null);

    // Add the loaded model to the scene
    try scene.addNode(name, root_node);

    std.debug.print("glTF file loaded successfully\n", .{});
}

pub fn loadAsNode(allocator: Allocator, filepath: []const u8, name: []const u8) !*Node {
    std.debug.print("Loading glTF file: {s}\n", .{filepath});

    // Parse the glTF file
    var gltf = try GLTF.init(allocator, filepath);
    defer gltf.deinit();

    // Load the default scene from the glTF file
    const root_node = try gltf.loadScene(allocator, null);
    std.debug.print("glTF : {s} loaded successfully\n", .{name});
    return root_node;
}

/// Basic type definitions for glTF
pub const GLTF = struct {
    allocator: Allocator,
    raw_json: []const u8,
    document: Document,
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
        defer document.deinit();

        // Create the GLTF instance
        const gltf = try allocator.create(GLTF);
        gltf.* = GLTF{
            .allocator = allocator,
            .raw_json = raw_json,
            .document = document.value,
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
        if (self.document.buffers) |buffers| {
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

    pub fn loadScene(self: *GLTF, allocator: Allocator, scene_index: ?usize) !*Node {
        // Get the scene to load (default or specified)
        const scene_idx = scene_index orelse self.document.scene orelse 0;
        if (self.document.scenes == null or scene_idx >= self.document.scenes.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const scene = self.document.scenes.?[scene_idx];

        try self.preloadImages();
        try self.preloadMaterials();

        // Create root node for this scene
        const root_node = try Node.init(allocator, null, null, null);

        // Process all top-level nodes in the scene
        if (scene.nodes) |node_indices| {
            for (node_indices) |node_idx| {
                const child_node = try self.loadNode(allocator, node_idx);
                try root_node.addChild(child_node);
            }
        }

        return root_node;
    }

    fn preloadImages(self: *GLTF) !void {
        if (self.document.images == null) return;

        const images = self.document.images.?;
        try self.materials.ensureTotalCapacity(images.len);

        for (images) |image| {
            var texture: ?Mesh.TextureID = null;

            if (image.uri) |uri| {
                const full_path = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ self.base_path, uri });
                defer self.allocator.free(full_path);

                const img = ImageLoader.Image.loadFromFile(self.allocator, full_path) catch |err| {
                    std.debug.print("Failed to load image: {}\n", .{err});
                    return error.FailedToLoadImage;
                };
                defer img.deinit();

                texture = try img.createGLTexture();
            }

            if (image.uri == null and image.bufferView != null) {
                // This is a buffer-embedded image
                if (image.mimeType) |_| {
                    // Load the image from buffer view
                    var img = try self.loadBufferViewImage(self.allocator, image.bufferView.?);

                    // Create GL textures
                    texture = try img.createGLTexture();
                }
            }

            if (texture) |_texture|
                try self.textures.append(_texture);
        }
    }

    fn preloadMaterials(self: *GLTF) !void {
        if (self.document.materials == null) {
            return;
        }

        // Ensure capacity for all materials
        try self.materials.ensureTotalCapacity(self.document.materials.?.len);

        // Process each material
        for (self.document.materials.?, 0..) |material_def, idx| {
            var material = Mesh.Material{};

            // Set basic material properties
            if (material_def.doubleSided) |double_sided| {
                material.doubleSided = double_sided;
            }

            if (material_def.alphaMode) |alpha_mode_str| {
                if (std.mem.eql(u8, alpha_mode_str, "MASK")) {
                    material.alphaMode = .MASK;
                } else if (std.mem.eql(u8, alpha_mode_str, "BLEND")) {
                    material.alphaMode = .BLEND;
                } else {
                    material.alphaMode = .OPAQUE;
                }
            }

            if (material_def.alphaCutoff) |alpha_cutoff| {
                material.alphaCutoff = alpha_cutoff;
            }

            // Set emissive factor if present
            if (material_def.emissiveFactor) |emissive| {
                material.emissiveFactor = emissive;
            }

            // Process PBR Metallic-Roughness parameters
            if (material_def.pbrMetallicRoughness) |pbr_mr| {
                if (pbr_mr.baseColorFactor) |base_color| {
                    material.baseColorFactor = base_color;
                }

                if (pbr_mr.metallicFactor) |metallic| {
                    material.metallicFactor = metallic;
                }

                if (pbr_mr.roughnessFactor) |roughness| {
                    material.roughnessFactor = roughness;
                }

                // Load base color texture
                if (pbr_mr.baseColorTexture) |tex_info| {
                    try self.loadTextureForMaterial(tex_info.index, &material, .baseColor);
                }

                // Load metallic-roughness texture
                if (pbr_mr.metallicRoughnessTexture) |tex_info| {
                    try self.loadTextureForMaterial(tex_info.index, &material, .metallicRoughness);
                }
            }

            // Load normal texture if present
            if (material_def.normalTexture) |tex_info| {
                try self.loadTextureForMaterial(tex_info.index, &material, .normal);
            }

            // Load occlusion texture if present
            if (material_def.occlusionTexture) |tex_info| {
                try self.loadTextureForMaterial(tex_info.index, &material, .occlusion);
            }

            // Load emissive texture if present
            if (material_def.emissiveTexture) |tex_info| {
                try self.loadTextureForMaterial(tex_info.index, &material, .emissive);
            }

            // Process extensions
            if (material_def.extensions) |extensions| {
                // KHR_materials_pbrSpecularGlossiness extension
                if (extensions.KHR_materials_pbrSpecularGlossiness) |sg| {
                    std.debug.print("Found KHR_materials_pbrSpecularGlossiness extension\n", .{});

                    // Set specular-glossiness parameters
                    if (sg.diffuseFactor) |diffuse| {
                        material.diffuseFactor = diffuse;
                    }

                    if (sg.specularFactor) |specular| {
                        material.specularFactor = specular;
                    }

                    if (sg.glossinessFactor) |glossiness| {
                        material.glossinessFactor = glossiness;
                    }

                    // Load diffuse texture
                    if (sg.diffuseTexture) |tex_info| {
                        try self.loadTextureForMaterial(
                            tex_info.index,
                            &material,
                            .baseColor,
                        );
                    }

                    // Load specular-glossiness texture if present
                    if (sg.specularGlossinessTexture) |tex_info| {
                        try self.loadTextureForMaterial(
                            tex_info.index,
                            &material,
                            .metallicRoughness,
                        );
                    }
                }

                // KHR_materials_emissive_strength extension
                if (extensions.KHR_materials_emissive_strength) |es| {
                    std.debug.print("Found KHR_materials_emissive_strength extension\n", .{});

                    if (es.emissiveStrength) |strength| {
                        material.emissiveStrength = strength;
                    }
                }

                // KHR_materials_specular extension
                if (extensions.KHR_materials_specular) |spec| {
                    std.debug.print("Found KHR_materials_specular extension\n", .{});

                    if (spec.specularFactor) |factor| {
                        material.specularStrength = factor;
                    }

                    if (spec.specularColorFactor) |color| {
                        material.specularColor = color;
                    }

                    // Load specular texture if present
                    if (spec.specularTexture) |tex_info| {
                        try self.loadTextureForMaterial(tex_info.index, &material, .specular);
                    }
                }
            }

            // Add the material to our cache
            try self.materials.append(material);

            std.debug.print("Preloaded Material IDX: {d}\n", .{idx});
            std.debug.print("{any}\n", .{material});
        }
    }

    fn loadTextureForMaterial(
        self: *GLTF,
        texture_idx: usize,
        material: *Mesh.Material,
        texture_type: TextureTypes,
    ) !void {
        std.debug.print("Loading Texture for: {s}...\n", .{@tagName(texture_type)});

        if (self.document.textures == null or texture_idx >= self.document.textures.?.len) {
            return;
        }

        const texture = self.document.textures.?[texture_idx];

        if (texture.source == null) return;
        const image_idx = texture.source.?;

        if (self.document.images == null or image_idx >= self.document.images.?.len) {
            return;
        }

        // Check if we've already loaded this texture
        if (self.textures.items.len > texture_idx) {
            const cached_texture = self.textures.items[texture_idx];

            switch (texture_type) {
                .baseColor => material.textures.baseColor = cached_texture.y,
                .normal => material.textures.normal = cached_texture.y,
                .metallicRoughness => material.textures.metallicRoughness = cached_texture.y,
                .occlusion => material.textures.occlusion = cached_texture.y,
                .emissive => material.textures.emissive = cached_texture.y,
                .specular => material.textures.specular = cached_texture.y,
            }

            return;
        }
    }

    pub fn loadNode(self: *GLTF, allocator: Allocator, node_idx: usize) !*Node {
        if (self.document.nodes == null or node_idx >= self.document.nodes.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const node_def = self.document.nodes.?[node_idx];

        var node = try Node.init(allocator, null, null, null);

        // Initialize node (we'll add mesh data if present)
        // Load mesh if present
        if (node_def.mesh) |mesh_idx| {
            if (try self.loadMesh(allocator, mesh_idx)) |mesh|
                node.mesh = mesh;
        }

        // Apply transformations
        if (node_def.matrix) |transformation| {
            // If matrix is provided, use it directly
            const mat = Mat4.from_array(transformation);
            node.local_transform = mat;
        } else {
            // Otherwise, apply TRS properties
            if (node_def.translation) |translation| {
                node.setPosition(
                    translation[0],
                    translation[1],
                    translation[2],
                );
            }

            if (node_def.rotation) |rotation| {
                const q = Quaternion.init(
                    rotation[0],
                    rotation[1],
                    rotation[2],
                    rotation[3],
                );
                node.setRotation(q);
            }

            if (node_def.scale) |scale| {
                node.setScale(scale[0], scale[1], scale[2]);
            }
        }

        // Process children nodes
        if (node_def.children) |children| {
            for (children) |child_idx| {
                const child_node = try self.loadNode(allocator, child_idx);
                try node.addChild(child_node);
            }
        }

        return node;
    }

    pub fn loadMesh(self: *GLTF, allocator: Allocator, mesh_idx: usize) !?*Mesh {
        if (self.document.meshes == null or mesh_idx >= self.document.meshes.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const mesh_def = self.document.meshes.?[mesh_idx];

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
        const mesh = try Mesh.init(allocator, vertices, indices, Mesh.gen_draw(glad.GL_TRIANGLES));

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

    fn loadBufferViewImage(self: *GLTF, allocator: Allocator, buffer_view_idx: usize) !*ImageLoader.Image {
        if (self.document.bufferViews == null or buffer_view_idx >= self.document.bufferViews.?.len) {
            return error.ImageErrorResourceNotFound;
        }

        const buffer_view = self.document.bufferViews.?[buffer_view_idx];

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
        if (self.document.accessors == null or accessor_idx >= self.document.accessors.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const accessor = self.document.accessors.?[accessor_idx];

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
        if (self.document.bufferViews == null or buffer_view_idx >= self.document.bufferViews.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const buffer_view = self.document.bufferViews.?[buffer_view_idx];

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
            const pos = offset + accessor_offset + (i * stride);
            const vec4_ptr: *[4]f32 = @constCast(@ptrCast(@alignCast(&buffer[pos])));
            result[i] = vec4_ptr.*;
        }

        return result;
    }

    fn loadAccessorVec3(self: *GLTF, accessor_idx: usize) ![][3]f32 {
        if (self.document.accessors == null or accessor_idx >= self.document.accessors.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const accessor = self.document.accessors.?[accessor_idx];

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
        if (self.document.bufferViews == null or buffer_view_idx >= self.document.bufferViews.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const buffer_view = self.document.bufferViews.?[buffer_view_idx];

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
            const pos = offset + accessor_offset + (i * stride);
            const vec3_ptr: *[3]f32 = @constCast(@ptrCast(@alignCast(&buffer[pos])));
            result[i] = vec3_ptr.*;
        }

        return result;
    }

    fn loadAccessorVec2(self: *GLTF, accessor_idx: usize) ![][2]f32 {
        if (self.document.accessors == null or accessor_idx >= self.document.accessors.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const accessor = self.document.accessors.?[accessor_idx];

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
        if (self.document.bufferViews == null or buffer_view_idx >= self.document.bufferViews.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const buffer_view = self.document.bufferViews.?[buffer_view_idx];

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
            const pos = offset + accessor_offset + (i * stride);
            const vec2_ptr: *[2]f32 = @constCast(@ptrCast(@alignCast(&buffer[pos])));
            result[i] = vec2_ptr.*;
        }

        return result;
    }

    fn loadAccessorIndices(self: *GLTF, accessor_idx: usize) ![]u32 {
        if (self.document.accessors == null or accessor_idx >= self.document.accessors.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const accessor = self.document.accessors.?[accessor_idx];

        // Check type compatibility
        if (accessor.type == null or !std.mem.eql(u8, accessor.type.?, "SCALAR")) {
            return GLTFError.UnsupportedFeature;
        }

        // Get buffer view
        if (accessor.bufferView == null) {
            return GLTFError.MissingRequiredField;
        }

        const buffer_view_idx = accessor.bufferView.?;
        if (self.document.bufferViews == null or buffer_view_idx >= self.document.bufferViews.?.len) {
            return GLTFError.ResourceNotFound;
        }

        const buffer_view = self.document.bufferViews.?[buffer_view_idx];

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
                    result[i] = @intCast(buffer[byte_pos]);
                    byte_pos += 1;
                }
            },
            5123 => { // GL_UNSIGNED_SHORT
                while (i < count) : (i += 1) {
                    const short_ptr: *u16 = @constCast(@ptrCast(@alignCast(&buffer[byte_pos])));
                    result[i] = @intCast(short_ptr.*);
                    byte_pos += 2;
                }
            },
            5125 => { // GL_UNSIGNED_INT
                while (i < count) : (i += 1) {
                    const int_ptr: *u32 = @constCast(@ptrCast(@alignCast(&buffer[byte_pos])));
                    result[i] = int_ptr.*;
                    byte_pos += 4;
                }
            },
            else => return GLTFError.UnsupportedFeature,
        }

        return result;
    }
};
