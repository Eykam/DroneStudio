//src/core/ecs/graphics/OpenGL.zig
const std = @import("std");
const GLTF = @import("../../GLTF.zig");
const gl = @import("../../bindings/gl.zig");
const glad = gl.glad;

// ============================================================================
// Texture Unit Management
// ============================================================================

pub const TextureUnit = enum(c_uint) {
    // Material property textures
    BaseColor = 0,
    NormalMap = 1,
    MetallicRoughness = 2,
    Occlusion = 3,
    Emissive = 4,
    Specular = 5,

    // Reserved for other uses
    Shadow = 6,
    Environment = 7,
    Irradiance = 8,
    LUT = 9,

    pub const SlotCount = @typeInfo(TextureUnit).@"enum".fields.len;

    pub inline fn glValue(self: TextureUnit) c_uint {
        return @as(c_uint, @intCast(glad.GL_TEXTURE0)) + @intFromEnum(self);
    }

    pub inline fn idx(self: TextureUnit) usize {
        return @intFromEnum(self);
    }

    pub inline fn index(self: TextureUnit) c_int {
        return @intCast(@intFromEnum(self));
    }
};

/// GL sampler parameters (OpenGL constants)
pub const SamplerParams = struct {
    mag_filter: c_int = glad.GL_LINEAR,
    min_filter: c_int = glad.GL_LINEAR_MIPMAP_LINEAR,
    wrap_s: c_int = glad.GL_REPEAT,
    wrap_t: c_int = glad.GL_REPEAT,

    pub const default: SamplerParams = .{};
};

/// Sampler utilities for glTF -> OpenGL conversion.
/// For bindless textures, params are applied directly to the texture before
/// calling glGetTextureHandleARB(), which bakes the sampling state into the handle.
pub const Sampler = struct {
    /// Converts a GLTF.Sampler into OpenGL sampler params.
    pub inline fn paramsFromGltf(gltf_sampler: GLTF.Sampler) SamplerParams {
        return .{
            .mag_filter = mapMagFilter(gltf_sampler.magFilter),
            .min_filter = mapMinFilter(gltf_sampler.minFilter),
            .wrap_s = mapWrap(gltf_sampler.wrapS),
            .wrap_t = mapWrap(gltf_sampler.wrapT),
        };
    }

    /// Apply sampler params directly onto a texture object.
    /// Call this BEFORE makeBindless() to bake params into the bindless handle.
    pub fn applyToTexture2D(texture_id: c_uint, params: SamplerParams) void {
        glad.glBindTexture(glad.GL_TEXTURE_2D, texture_id);
        defer glad.glBindTexture(glad.GL_TEXTURE_2D, 0);
        glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MAG_FILTER, params.mag_filter);
        glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_MIN_FILTER, params.min_filter);
        glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_WRAP_S, params.wrap_s);
        glad.glTexParameteri(glad.GL_TEXTURE_2D, glad.GL_TEXTURE_WRAP_T, params.wrap_t);
    }

    // glTF -> OpenGL mappings

    inline fn mapMagFilter(mag: ?GLTF.MagFilter) c_int {
        const v = mag orelse return glad.GL_LINEAR;
        return switch (v) {
            .nearest => glad.GL_NEAREST,
            .linear => glad.GL_LINEAR,
        };
    }

    inline fn mapMinFilter(min: ?GLTF.MinFilter) c_int {
        const v = min orelse return glad.GL_LINEAR;
        return switch (v) {
            .nearest => glad.GL_NEAREST,
            .linear => glad.GL_LINEAR,
            .nearest_mipmap_nearest => glad.GL_NEAREST_MIPMAP_NEAREST,
            .linear_mipmap_nearest => glad.GL_LINEAR_MIPMAP_NEAREST,
            .nearest_mipmap_linear => glad.GL_NEAREST_MIPMAP_LINEAR,
            .linear_mipmap_linear => glad.GL_LINEAR_MIPMAP_LINEAR,
        };
    }

    inline fn mapWrap(wrap: ?GLTF.WrapMode) c_int {
        const v = wrap orelse return glad.GL_REPEAT;
        return switch (v) {
            .clamp_to_edge => glad.GL_CLAMP_TO_EDGE,
            .mirrored_repeat => glad.GL_MIRRORED_REPEAT,
            .repeat => glad.GL_REPEAT,
        };
    }
};

// ============================================================================
// Material Buffer System (SSBOs + Bindless Textures)
// ============================================================================

/// Maximum number of materials supported in the buffer
pub const MAX_MATERIALS: usize = 2048;

/// GPU-compatible material header (std430 layout)
/// Each material has bindless texture handles + metadata
pub const MaterialGPU = extern struct {
    /// Bindless texture handles for each slot (uvec2 in GLSL = u64)
    texture_handles: [TextureUnit.SlotCount]u64 align(8),
    /// Bitmask of which texture slots are valid
    texture_mask: u32,
    /// 0 = Phong, 1 = PBR
    material_type: u32,
    /// Index into type-specific data array (PBR or Phong)
    data_index: u32,
    /// Packed flags: bits 0-1 = alphaMode, bit 2 = doubleSided
    flags: u32,

    pub const FLAG_DOUBLE_SIDED: u32 = 1 << 2;
    pub const ALPHA_OPAQUE: u32 = 0;
    pub const ALPHA_MASK: u32 = 1;
    pub const ALPHA_BLEND: u32 = 2;
};

/// GPU-compatible PBR material data (std430 layout)
pub const PBRDataGPU = extern struct {
    baseColorFactor: [4]f32 align(16),
    emissiveFactor: [3]f32,
    emissiveStrength: f32,
    specularColor: [3]f32,
    specularStrength: f32,
    metallicFactor: f32,
    roughnessFactor: f32,
    alphaCutoff: f32,
    _pad: f32 = 0,
};

/// GPU-compatible Phong material data (std430 layout)
pub const PhongDataGPU = extern struct {
    ambientColor: [3]f32 align(16),
    shininess: f32,
    diffuseColor: [4]f32,
    specularColor: [3]f32,
    _pad: f32 = 0,
};

/// SSBO binding points
pub const SSBO_BINDING_MATERIALS: c_uint = 0;
pub const SSBO_BINDING_PBR_DATA: c_uint = 1;
pub const SSBO_BINDING_PHONG_DATA: c_uint = 2;

/// Manages material SSBOs for GPU-driven rendering
pub const MaterialBuffer = struct {
    const Self = @This();

    // SSBO handles
    materials_ssbo: c_uint = 0,
    pbr_data_ssbo: c_uint = 0,
    phong_data_ssbo: c_uint = 0,

    // CPU-side data (dynamically allocated)
    materials: []MaterialGPU,
    pbr_data: []PBRDataGPU,
    phong_data: []PhongDataGPU,
    allocator: std.mem.Allocator,

    // Counts
    material_count: u32 = 0,
    pbr_count: u32 = 0,
    phong_count: u32 = 0,

    // Dirty flag for lazy upload
    dirty: bool = false,

    pub fn init(allocator: std.mem.Allocator) !Self {
        var self = Self{
            .allocator = allocator,
            .materials = try allocator.alloc(MaterialGPU, MAX_MATERIALS),
            .pbr_data = try allocator.alloc(PBRDataGPU, MAX_MATERIALS),
            .phong_data = try allocator.alloc(PhongDataGPU, MAX_MATERIALS),
        };

        // Generate SSBOs
        glad.glGenBuffers(1, &self.materials_ssbo);
        glad.glGenBuffers(1, &self.pbr_data_ssbo);
        glad.glGenBuffers(1, &self.phong_data_ssbo);

        // Allocate GPU storage
        self.allocateBuffers();

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.materials_ssbo != 0) glad.glDeleteBuffers(1, &self.materials_ssbo);
        if (self.pbr_data_ssbo != 0) glad.glDeleteBuffers(1, &self.pbr_data_ssbo);
        if (self.phong_data_ssbo != 0) glad.glDeleteBuffers(1, &self.phong_data_ssbo);

        self.allocator.free(self.materials);
        self.allocator.free(self.pbr_data);
        self.allocator.free(self.phong_data);
    }

    fn allocateBuffers(self: *Self) void {
        // Materials SSBO
        glad.glBindBuffer(glad.GL_SHADER_STORAGE_BUFFER, self.materials_ssbo);
        glad.glBufferData(
            glad.GL_SHADER_STORAGE_BUFFER,
            @intCast(MAX_MATERIALS * @sizeOf(MaterialGPU)),
            null,
            glad.GL_DYNAMIC_DRAW,
        );

        // PBR data SSBO
        glad.glBindBuffer(glad.GL_SHADER_STORAGE_BUFFER, self.pbr_data_ssbo);
        glad.glBufferData(
            glad.GL_SHADER_STORAGE_BUFFER,
            @intCast(MAX_MATERIALS * @sizeOf(PBRDataGPU)),
            null,
            glad.GL_DYNAMIC_DRAW,
        );

        // Phong data SSBO
        glad.glBindBuffer(glad.GL_SHADER_STORAGE_BUFFER, self.phong_data_ssbo);
        glad.glBufferData(
            glad.GL_SHADER_STORAGE_BUFFER,
            @intCast(MAX_MATERIALS * @sizeOf(PhongDataGPU)),
            null,
            glad.GL_DYNAMIC_DRAW,
        );

        glad.glBindBuffer(glad.GL_SHADER_STORAGE_BUFFER, 0);
    }

    /// Add a PBR material, returns the material index
    pub fn addPBRMaterial(
        self: *Self,
        texture_handles: [TextureUnit.SlotCount]u64,
        texture_mask: u32,
        data: PBRDataGPU,
        flags: u32,
    ) ?u32 {
        if (self.material_count >= MAX_MATERIALS or self.pbr_count >= MAX_MATERIALS) return null;

        const mat_idx = self.material_count;
        const data_idx = self.pbr_count;

        self.materials[mat_idx] = .{
            .texture_handles = texture_handles,
            .texture_mask = texture_mask,
            .material_type = 1, // PBR
            .data_index = data_idx,
            .flags = flags,
        };
        self.pbr_data[data_idx] = data;

        self.material_count += 1;
        self.pbr_count += 1;
        self.dirty = true;

        return mat_idx;
    }

    /// Add a Phong material, returns the material index
    pub fn addPhongMaterial(
        self: *Self,
        texture_handles: [TextureUnit.SlotCount]u64,
        texture_mask: u32,
        data: PhongDataGPU,
    ) ?u32 {
        if (self.material_count >= MAX_MATERIALS or self.phong_count >= MAX_MATERIALS) return null;

        const mat_idx = self.material_count;
        const data_idx = self.phong_count;

        self.materials[mat_idx] = .{
            .texture_handles = texture_handles,
            .texture_mask = texture_mask,
            .material_type = 0, // Phong
            .data_index = data_idx,
            .flags = 0,
        };
        self.phong_data[data_idx] = data;

        self.material_count += 1;
        self.phong_count += 1;
        self.dirty = true;

        return mat_idx;
    }

    /// Upload all material data to GPU (call once per frame if dirty)
    pub fn upload(self: *Self) void {
        if (!self.dirty) return;

        // Upload materials
        glad.glBindBuffer(glad.GL_SHADER_STORAGE_BUFFER, self.materials_ssbo);
        glad.glBufferSubData(
            glad.GL_SHADER_STORAGE_BUFFER,
            0,
            @intCast(self.material_count * @sizeOf(MaterialGPU)),
            @ptrCast(self.materials.ptr),
        );

        // Upload PBR data
        if (self.pbr_count > 0) {
            glad.glBindBuffer(glad.GL_SHADER_STORAGE_BUFFER, self.pbr_data_ssbo);
            glad.glBufferSubData(
                glad.GL_SHADER_STORAGE_BUFFER,
                0,
                @intCast(self.pbr_count * @sizeOf(PBRDataGPU)),
                @ptrCast(self.pbr_data.ptr),
            );
        }

        // Upload Phong data
        if (self.phong_count > 0) {
            glad.glBindBuffer(glad.GL_SHADER_STORAGE_BUFFER, self.phong_data_ssbo);
            glad.glBufferSubData(
                glad.GL_SHADER_STORAGE_BUFFER,
                0,
                @intCast(self.phong_count * @sizeOf(PhongDataGPU)),
                @ptrCast(self.phong_data.ptr),
            );
        }

        glad.glBindBuffer(glad.GL_SHADER_STORAGE_BUFFER, 0);
        self.dirty = false;
    }

    /// Bind SSBOs to their respective binding points (call before rendering)
    pub fn bind(self: *Self) void {
        glad.glBindBufferBase(glad.GL_SHADER_STORAGE_BUFFER, SSBO_BINDING_MATERIALS, self.materials_ssbo);
        glad.glBindBufferBase(glad.GL_SHADER_STORAGE_BUFFER, SSBO_BINDING_PBR_DATA, self.pbr_data_ssbo);
        glad.glBindBufferBase(glad.GL_SHADER_STORAGE_BUFFER, SSBO_BINDING_PHONG_DATA, self.phong_data_ssbo);
    }

    /// Clear all materials (for reloading)
    pub fn clear(self: *Self) void {
        self.material_count = 0;
        self.pbr_count = 0;
        self.phong_count = 0;
        self.dirty = true;
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
        glad.glTexImage2D(glad.GL_TEXTURE_2D, 0, glad.GL_RGBA8, width, height, 0, glad.GL_RGBA, glad.GL_UNSIGNED_BYTE, null);
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

        const status = glad.glCheckFramebufferStatus(glad.GL_FRAMEBUFFER);
        if (status != glad.GL_FRAMEBUFFER_COMPLETE) {
            std.debug.print("Framebuffer is not complete! Status: 0x{X}\n", .{status});
            switch (status) {
                glad.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT => std.debug.print("Incomplete attachment\n", .{}),
                glad.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => std.debug.print("Missing attachment\n", .{}),
                glad.GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER => std.debug.print("Incomplete draw buffer\n", .{}),
                glad.GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER => std.debug.print("Incomplete read buffer\n", .{}),
                glad.GL_FRAMEBUFFER_UNSUPPORTED => std.debug.print("Unsupported format combination\n", .{}),
                else => std.debug.print("Unknown framebuffer status error\n", .{}),
            }
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

    name: [:0]const u8,
    fbo: FrameBuffer,
    shader_program: c_uint,
    enabled: bool,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, width: i32, height: i32) !Self {
        const name_copy = try allocator.dupeZ(u8, name);
        const fbo = try FrameBuffer.init(width, height);

        const shader_program = try createShaderProgram("../../shaders/miniview_vertex.glsl", "../../shaders/miniview_fragment.glsl");
        const viewport = Self{
            .name = name_copy,
            .fbo = fbo,
            .shader_program = shader_program,
            .enabled = true,
            .allocator = allocator,
        };

        return viewport;
    }

    pub fn deinit(self: *Self) void {
        self.fbo.deinit();
        glad.glDeleteProgram(self.shader_program);

        self.allocator.free(self.name);
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
        return error.UnableToCreateShader;
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
        return error.ShaderCompilationFailed;
    }

    return shader;
}

pub fn createShaderProgram(comptime vertexPath: []const u8, comptime fragmentPath: []const u8) !u32 {
    std.debug.print("Initializing Vertex Shader...\n", .{});
    std.debug.print("Reading Vertex Shader from Source...\n", .{});
    const vertexSource = try readShaderSource(vertexPath);
    const vertexShader = compileShader(glad.GL_VERTEX_SHADER, vertexSource) catch |err| {
        std.debug.print("Failed to read vertex shader '{s}': {any}\n", .{ vertexPath, err });
        return error.ShaderCompilationFailed;
    };
    std.debug.print("\n", .{});

    std.debug.print("Initializing Fragment Shader...\n", .{});
    std.debug.print("Reading Fragment Shader from Source...\n", .{});
    const fragmentSource = try readShaderSource(fragmentPath);
    const fragmentShader = compileShader(glad.GL_FRAGMENT_SHADER, fragmentSource) catch |err| {
        std.debug.print("Failed to read fragment shader '{s}': {any}\n", .{ fragmentPath, err });
        return error.ShaderCompilationFailed;
    };
    std.debug.print("\n", .{});

    std.debug.print("Creating ShaderProgram...\n", .{});
    const shaderProgram = glad.glCreateProgram();
    if (shaderProgram == 0) {
        std.debug.print("Failed to create Shader Program\n", .{});
        return error.UnableToCreateProgram;
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
        return error.ShaderLinkingFailed;
    }

    std.debug.print("Running cleanup on Shaders...\n", .{});
    // Shaders can be deleted after linking
    glad.glDeleteShader(vertexShader);
    glad.glDeleteShader(fragmentShader);

    return shaderProgram;
}
