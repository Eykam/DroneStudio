const std = @import("std");
const c = @import("bindings/c.zig");
const glad = @import("bindings/gl.zig").glad;
pub const cuda = c.cuda;

pub const StereoParams = cuda.StereoParams;
pub const ImageParams = cuda.ImageParams;

pub const BufferResource = struct {
    resource: cuda.cudaGraphicsResource_t,
    d_buffer: *cuda.float4,

    pub fn init(buffer_id: glad.GLuint) !BufferResource {
        var resource: BufferResource = BufferResource{};
        const err = cuda.cudaGraphicsGLRegisterBuffer(
            &resource.resource,
            buffer_id,
            cuda.cudaGraphicsRegisterFlagsWriteDiscard,
        );

        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to register buffer resource: {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });

            return error.FailedToRegisterBuffer;
        }

        return resource;
    }

    pub fn deinit(self: *BufferResource) void {
        self.unmap();
        _ = cuda.cudaGraphicsUnregisterResource(self.resource);
    }

    pub fn map(self: *BufferResource) !void {
        const err = cuda.cudaGraphicsMapResources(1, &self.resource, null);
        if (err != cuda.cudaSuccess) {
            return error.FailedToMapBUfferResource;
        }

        var bytes: usize = undefined;
        const ptr_err = cuda.cudaGraphicsResourceGetMappedPointer(
            @ptrCast(&self.d_buffer),
            &bytes,
            self.resource,
        );

        if (ptr_err != cuda.cudaSuccess) {
            return error.FailedToGetMappedBufferPointer;
        }
    }

    pub fn unmap(self: *BufferResource) void {
        _ = cuda.cudaGraphicsUnmapResources(1, &self.resource, null);
    }
};

pub const TextureResource = struct {
    tex_resource: cuda.cudaGraphicsResource_t,
    array: cuda.cudaArray_t,
    surface: cuda.cudaSurfaceObject_t,

    pub fn init(texture_id: glad.GLuint) !TextureResource {
        var resource: TextureResource = undefined;
        const err = cuda.cudaGraphicsGLRegisterImage(
            &resource.tex_resource,
            texture_id,
            cuda.GL_TEXTURE_2D,
            cuda.cudaGraphicsRegisterFlagsSurfaceLoadStore | cuda.cudaGraphicsRegisterFlagsWriteDiscard,
        );

        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to register GL texture with id: {}!\n", .{texture_id});
            return error.FailedToRegisterGLTexture;
        }
        return resource;
    }

    pub fn deinit(self: *TextureResource) void {
        self.unmap();
        _ = cuda.cudaGraphicsUnregisterResource(self.tex_resource);
    }

    pub fn map(self: *TextureResource) !void {
        // Map the texture resource
        var err = cuda.cudaGraphicsMapResources(1, &self.tex_resource, null);
        if (err != cuda.cudaSuccess) {
            return error.FailedToMapTextureResource;
        }

        // Get the mapped array
        err = cuda.cudaGraphicsSubResourceGetMappedArray(
            &self.array,
            self.tex_resource,
            0,
            0,
        );
        if (err != cuda.cudaSuccess) {
            return error.FailedToGetMappedTextureArray;
        }

        // Create surface object
        var res_desc: cuda.cudaResourceDesc = std.mem.zeroes(cuda.cudaResourceDesc);
        res_desc.resType = cuda.cudaResourceTypeArray;
        res_desc.res.array.array = self.array;

        err = cuda.cudaCreateSurfaceObject(&self.surface, &res_desc);
        if (err != cuda.cudaSuccess) {
            return error.FailedToCreateSurfaceObject;
        }
    }

    pub fn unmap(self: *TextureResource) void {
        _ = cuda.cudaDestroySurfaceObject(self.surface);
        _ = cuda.cudaGraphicsUnmapResources(1, &self.tex_resource, null);
    }
};

pub const BufferParams = struct {
    position: glad.GLuint,
    color: glad.GLuint,
    size: u32,
};

pub const Buffers = struct {
    positions: BufferResource,
    colors: BufferResource,
    buffer_size: u32,

    pub fn init(buffer_ids: BufferParams) !Buffers {
        return Buffers{
            .positions = try BufferResource.init(buffer_ids.position),
            .colors = try BufferResource.init(buffer_ids.color),
            .buffer_size = buffer_ids.size,
        };
    }

    pub fn deinit(self: *Buffers) void {
        self.positions.deinit();
        self.colors.deinit();
    }

    pub fn map(self: *Buffers) !void {
        try self.positions.map();
        try self.colors.map();
    }

    pub fn unmap(self: *Buffers) void {
        self.positions.unmap();
        self.colors.unmap();
    }
};

pub const ConnectionResources = struct {
    left: Buffers,
    right: Buffers,

    pub fn init(left: BufferParams, right: BufferParams) !ConnectionResources {
        return ConnectionResources{
            .left = try Buffers.init(left),
            .right = try Buffers.init(right),
        };
    }

    pub fn deinit(self: *ConnectionResources) void {
        self.left.deinit();
        self.right.deinit();
    }

    pub fn map(self: *ConnectionResources) !void {
        try self.left.map();
        try self.right.map();
    }

    pub fn unmap(self: *ConnectionResources) void {
        self.left.unmap();
        self.right.unmap();
    }
};

pub const KeypointResources = struct {
    connections: ?ConnectionResources,
    keypoints: Buffers,

    pub fn init(keypoint_buffers: BufferParams, left: ?BufferParams, right: ?BufferParams) !KeypointResources {
        var resources = KeypointResources{
            .keypoints = try Buffers.init(keypoint_buffers),
            .connections = null,
        };

        if (left != null and right != null) {
            resources.connections = try ConnectionResources.init(left.?, right.?);
        }

        return resources;
    }

    pub fn deinit(self: *KeypointResources) void {
        if (self.connections) |*conn| {
            conn.deinit();
        }
        self.keypoints.deinit();
    }

    pub fn map(self: *KeypointResources) !void {
        try self.keypoints.map();
        if (self.connections) |*conn| {
            try conn.map();
        }
    }

    pub fn unmap(self: *KeypointResources) void {
        self.keypoints.unmap();
        if (self.connections) |*conn| {
            conn.unmap();
        }
    }
};

pub const TextureIDs = struct {
    y: glad.GLuint,
    uv: glad.GLuint,
    depth: glad.GLuint,
};

pub const Textures = struct {
    y: ?TextureResource,
    uv: ?TextureResource,
    depth: ?TextureResource,
};

pub const DetectionResources = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    d_keypoint_count: [*c]u32,
    gl_texture_ids: TextureIDs,
    gl_textures: Textures,
    world_transform: [16]f32,
    d_world_transform: [*c]f32,
    keypoint_resources: ?KeypointResources,
    d_descriptors: [*c]cuda.BRIEFDescriptor,
    id: u32,
    initialized: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        id: u32,
        max_keypoints: u32,
        y_texture_id: glad.GLuint,
        uv_texture_id: glad.GLuint,
        depth_texture_id: glad.GLuint,
        buffer_ids: BufferParams,
        left_connection: ?BufferParams,
        right_connection: ?BufferParams,
    ) !*Self {
        const self = try allocator.create(Self);

        var d_keypoint_count: [*c]u32 = undefined;
        var d_descriptors: [*c]cuda.BRIEFDescriptor = undefined;
        var d_world_transform: [*c]f32 = undefined;

        var err = cuda.cudaMalloc(@ptrCast(&d_keypoint_count), @sizeOf(u32));
        err |= cuda.cudaMalloc(@ptrCast(&d_descriptors), max_keypoints * @sizeOf(cuda.BRIEFDescriptor));
        err |= cuda.cudaMalloc(@ptrCast(&d_world_transform), 16 * @sizeOf(f32));

        if (err != cuda.cudaSuccess) {
            return error.CudaAllocationFailed;
        }

        var identity = [_]f32{
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        };

        err = cuda.cudaMemcpy(
            d_world_transform,
            &identity,
            16 * @sizeOf(f32),
            cuda.cudaMemcpyHostToDevice,
        );

        if (err != cuda.cudaSuccess) {
            // Clean up already allocated memory
            std.debug.print("Error Copying Identity matrix! Err => {s}: {s}\n", .{ cuda.cudaGetErrorName(err), cuda.cudaGetErrorString(err) });

            _ = cuda.cudaFree(d_keypoint_count);
            _ = cuda.cudaFree(d_descriptors);
            _ = cuda.cudaFree(d_world_transform);

            return error.CudaMemcpyFailed;
        }

        self.* = .{
            .allocator = allocator,
            .initialized = true,
            .id = id,
            .gl_texture_ids = .{
                .y = y_texture_id,
                .uv = uv_texture_id,
                .depth = depth_texture_id,
            },
            .gl_textures = .{
                .y = null,
                .uv = null,
                .depth = null,
            },
            .world_transform = identity,
            .d_keypoint_count = d_keypoint_count,
            .d_world_transform = d_world_transform,
            .keypoint_resources = null,
            .d_descriptors = d_descriptors,
        };

        try self.register_textures();
        try self.register_buffers(buffer_ids, left_connection, right_connection);

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.gl_textures.y) |*tex| tex.deinit();
        if (self.gl_textures.uv) |*tex| tex.deinit();
        if (self.gl_textures.depth) |*tex| tex.deinit();

        if (self.keypoint_resources) |*res| res.deinit();

        _ = cuda.cudaFree(self.d_keypoint_count);
        _ = cuda.cudaFree(self.d_descriptors);
        _ = cuda.cudaFree(self.d_world_transform);

        self.allocator.destroy(self);
    }

    pub fn map_transformation(self: *Self, transformation: [16]f32) !void {
        const err = cuda.cudaMemcpy(
            self.d_world_transform,
            &transformation,
            transformation.len * @sizeOf(f32),
            cuda.cudaMemcpyHostToDevice,
        );

        if (err != cuda.cudaSuccess) {
            std.debug.print("Transform copy to device failed: {s}\n", .{cuda.cudaGetErrorString(err)});
            return error.MapTransformationFailed;
        }
    }

    pub fn register_textures(self: *Self) !void {
        self.gl_textures.y = try TextureResource.init(self.gl_texture_ids.y);
        self.gl_textures.uv = try TextureResource.init(self.gl_texture_ids.uv);
        self.gl_textures.depth = try TextureResource.init(self.gl_texture_ids.depth);
    }

    pub fn register_buffers(self: *Self, own: BufferParams, left: ?BufferParams, right: ?BufferParams) !void {
        self.keypoint_resources = try KeypointResources.init(own, left, right);
    }

    pub fn map_resources(self: *Self) !void {
        if (self.gl_textures.y) |*tex| try tex.map();
        if (self.gl_textures.uv) |*tex| try tex.map();
        if (self.gl_textures.depth) |*tex| try tex.map();

        if (self.keypoint_resources) |*res| try res.map();
    }

    pub fn unmap_resources(self: *Self) void {
        if (self.gl_textures.y) |*tex| tex.unmap();
        if (self.gl_textures.uv) |*tex| tex.unmap();
        if (self.gl_textures.depth) |*tex| tex.unmap();

        if (self.keypoint_resources) |*res| res.unmap();
    }
};
