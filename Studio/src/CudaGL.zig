const std = @import("std");
const c = @import("bindings/c.zig");
const glad = @import("bindings/gl.zig").glad;
pub const cuda = c.cuda;

pub const StereoParams = cuda.StereoParams;
pub const ImageParams = cuda.ImageParams;

pub const BufferResource = struct {
    const Self = @This();

    resource: cuda.cudaGraphicsResource_t,
    d_buffer: *cuda.float4,

    pub fn init(allocator: std.mem.Allocator, buffer_id: glad.GLuint) !*Self {
        var buffer_resource = try allocator.create(Self);

        buffer_resource.* = Self{
            .resource = undefined,
            .d_buffer = undefined,
        };

        const err = cuda.cudaGraphicsGLRegisterBuffer(
            &buffer_resource.resource,
            buffer_id,
            cuda.cudaGraphicsRegisterFlagsWriteDiscard,
        );

        if (err != cuda.cudaSuccess) {
            allocator.destroy(buffer_resource);

            std.debug.print("Failed to register buffer resource {d}: {s} => {s}\n", .{
                buffer_id,
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });

            return error.FailedToRegisterBuffer;
        }

        std.debug.print("Successfully registed Buffer with ID: {d}\n", .{buffer_id});

        return buffer_resource;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.unmap() catch |err| {
            std.debug.print("Failed To Deinitalize BufferResource! {any}\n", .{err});
        };
        _ = cuda.cudaGraphicsUnregisterResource(self.resource);
        allocator.destroy(self);
    }

    pub fn map(self: *Self) !void {
        // std.debug.print("Mapping Buffer Resource...\n", .{});

        const err = cuda.cudaGraphicsMapResources(1, &self.resource, null);
        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to map Buffer Resource! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });

            return error.FailedToMapBUfferResource;
        }

        var bytes: usize = undefined;
        const ptr_err = cuda.cudaGraphicsResourceGetMappedPointer(
            @ptrCast(&self.d_buffer),
            &bytes,
            self.resource,
        );

        if (ptr_err != cuda.cudaSuccess) {
            std.debug.print("Failed to get mapped Buffer Pointer! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
            return error.FailedToGetMappedBufferPointer;
        }
    }

    pub fn unmap(self: *Self) !void {
        const err = cuda.cudaGraphicsUnmapResources(1, &self.resource, null);

        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to unmap BufferResource! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });

            return error.FailedToUnmapBufferResource;
        }
    }
};

pub const TextureResource = struct {
    const Self = @This();

    tex_resource: cuda.cudaGraphicsResource_t,
    array: cuda.cudaArray_t,
    surface: cuda.cudaSurfaceObject_t,

    pub fn init(allocator: std.mem.Allocator, texture_id: glad.GLuint) !*Self {
        var texture_resource = try allocator.create(Self);
        texture_resource.* = Self{
            .tex_resource = undefined,
            .array = undefined,
            .surface = undefined,
        };

        const err = cuda.cudaGraphicsGLRegisterImage(
            &texture_resource.tex_resource,
            texture_id,
            cuda.GL_TEXTURE_2D,
            cuda.cudaGraphicsRegisterFlagsSurfaceLoadStore | cuda.cudaGraphicsRegisterFlagsWriteDiscard,
        );

        if (err != cuda.cudaSuccess) {
            allocator.destroy(texture_resource);

            std.debug.print("Failed to register Texture with ID {d}: {s} => {s}\n", .{
                texture_id,
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
            return error.FailedToRegisterGLTexture;
        }

        std.debug.print("Successfully registed Texture with ID: {d}\n", .{texture_id});

        return texture_resource;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.unmap() catch |err| {
            std.debug.print("Failed To Deinitalize TextureResource! {any}\n", .{err});
        };
        _ = cuda.cudaGraphicsUnregisterResource(self.tex_resource);
        allocator.destroy(self);
    }

    pub fn map(self: *Self) !void {
        // Map the texture resource
        // std.debug.print("Mapping Texture Resources...\n", .{});

        var err = cuda.cudaGraphicsMapResources(1, &self.tex_resource, null);
        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to map Texture Resources Object! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
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
            std.debug.print("Failed to get Texture Mapped Array Object! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
            return error.FailedToGetMappedTextureArray;
        }

        // Create surface object
        var res_desc: cuda.cudaResourceDesc = std.mem.zeroes(cuda.cudaResourceDesc);
        res_desc.resType = cuda.cudaResourceTypeArray;
        res_desc.res.array.array = self.array;

        err = cuda.cudaCreateSurfaceObject(&self.surface, &res_desc);
        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to create Surface Object! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });
            return error.FailedToCreateSurfaceObject;
        }
    }

    pub fn unmap(self: *Self) !void {
        var err = cuda.cudaDestroySurfaceObject(self.surface);
        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to Destroy Surface Object! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });

            return error.FailedToDestroySurfaceObject;
        }

        err = cuda.cudaGraphicsUnmapResources(1, &self.tex_resource, null);

        if (err != cuda.cudaSuccess) {
            std.debug.print("Failed to Unmap TextureResource! {s} => {s}\n", .{
                cuda.cudaGetErrorName(err),
                cuda.cudaGetErrorString(err),
            });

            return error.FailedToUnmapTextureResource;
        }
    }
};

pub const BufferParams = struct {
    position: glad.GLuint,
    color: glad.GLuint,
    size: u32,
};

pub const Buffers = struct {
    const Self = @This();

    positions: *BufferResource,
    colors: *BufferResource,
    buffer_size: u32,

    pub fn init(allocator: std.mem.Allocator, buffer_ids: BufferParams) !Self {
        return Self{
            .positions = try BufferResource.init(allocator, buffer_ids.position),
            .colors = try BufferResource.init(allocator, buffer_ids.color),
            .buffer_size = buffer_ids.size,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.positions.deinit(allocator);
        self.colors.deinit(allocator);
    }

    pub fn map(self: *Self) !void {
        try self.positions.map();
        try self.colors.map();
    }

    pub fn unmap(self: *Self) !void {
        try self.positions.unmap();
        try self.colors.unmap();
    }
};

pub const ConnectionResources = struct {
    const Self = @This();

    left: Buffers,
    right: Buffers,

    pub fn init(allocator: std.mem.Allocator, left: BufferParams, right: BufferParams) !Self {
        return Self{
            .left = try Buffers.init(allocator, left),
            .right = try Buffers.init(allocator, right),
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.left.deinit(allocator);
        self.right.deinit(allocator);
    }

    pub fn map(self: *Self) !void {
        try self.left.map();
        try self.right.map();
    }

    pub fn unmap(self: *Self) !void {
        try self.left.unmap();
        try self.right.unmap();
    }
};

pub const KeypointResources = struct {
    const Self = @This();

    connections: ?ConnectionResources,
    keypoints: Buffers,

    pub fn init(allocator: std.mem.Allocator, keypoint_buffers: BufferParams, left: ?BufferParams, right: ?BufferParams) !Self {
        var resources = Self{
            .keypoints = try Buffers.init(allocator, keypoint_buffers),
            .connections = null,
        };

        if (left != null and right != null) {
            resources.connections = try ConnectionResources.init(allocator, left.?, right.?);
        }

        return resources;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        if (self.connections) |*conn| {
            conn.deinit(allocator);
        }
        self.keypoints.deinit(allocator);
    }

    pub fn map(self: *Self) !void {
        try self.keypoints.map();
        if (self.connections) |*conn| {
            try conn.map();
        }
    }

    pub fn unmap(self: *Self) !void {
        try self.keypoints.unmap();
        if (self.connections) |*conn| {
            try conn.unmap();
        }
    }
};

pub const TextureIDs = struct {
    y: glad.GLuint,
    uv: glad.GLuint,
    depth: glad.GLuint,
};

pub const Textures = struct {
    y: ?*TextureResource,
    uv: ?*TextureResource,
    depth: ?*TextureResource,
};

pub const DetectionResources = struct {
    const Self = @This();

    id: u32,
    allocator: std.mem.Allocator,
    initialized: bool,

    keypoint_resources: ?KeypointResources,
    gl_textures: Textures,
    gl_texture_ids: TextureIDs,
    world_transform: [16]f32,

    d_keypoint_count: [*c]c_uint,
    d_world_transform: [*c]f32,
    d_descriptors: [*c]cuda.BRIEFDescriptor,

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

        var err = cuda.cudaMalloc(@ptrCast(&d_keypoint_count), @sizeOf(c_uint));
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
        if (self.gl_textures.y) |tex| tex.deinit(self.allocator);
        if (self.gl_textures.uv) |tex| tex.deinit(self.allocator);
        if (self.gl_textures.depth) |tex| tex.deinit(self.allocator);

        if (self.keypoint_resources) |*res| res.deinit(self.allocator);

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
        std.debug.print("Registering Resources for y texture plane!\n", .{});
        self.gl_textures.y = try TextureResource.init(self.allocator, self.gl_texture_ids.y);
        std.debug.print("Registering Resources for uv texture plane!\n", .{});
        self.gl_textures.uv = try TextureResource.init(self.allocator, self.gl_texture_ids.uv);
        std.debug.print("Registering Resources for depth texture plane!\n", .{});
        self.gl_textures.depth = try TextureResource.init(self.allocator, self.gl_texture_ids.depth);
    }

    pub fn register_buffers(self: *Self, own: BufferParams, left: ?BufferParams, right: ?BufferParams) !void {
        self.keypoint_resources = try KeypointResources.init(self.allocator, own, left, right);
    }

    pub fn map_resources(self: *Self) !void {
        // std.debug.print("Mapping Resources...\n", .{});

        if (self.gl_textures.y) |tex| try tex.map();
        if (self.gl_textures.uv) |tex| try tex.map();
        if (self.gl_textures.depth) |tex| try tex.map();

        if (self.keypoint_resources) |*res| try res.map();
    }

    pub fn unmap_resources(self: *Self) void {
        // std.debug.print("Unmapping Resources...\n", .{});

        if (self.gl_textures.y) |tex| tex.unmap() catch |err| {
            std.debug.print("Failed to unmap Y Texture plane!  {any}\n", .{err});
        };

        if (self.gl_textures.uv) |tex| tex.unmap() catch |err| {
            std.debug.print("Failed to unmap UV Texture plane!  {any}\n", .{err});
        };

        if (self.gl_textures.depth) |tex| tex.unmap() catch |err| {
            std.debug.print("Failed to unmap Depth Texture Plane!  {any}\n", .{err});
        };

        if (self.keypoint_resources) |*res| res.unmap() catch |err| {
            std.debug.print("Failed to unmap Keypoint Buffer Resources! {any}\n", .{err});
        };
    }
};

pub fn CudaKernelError(comptime prefix: []const u8) type {
    return struct {
        kernel_name: []const u8,
        duration_ms: f32,
        err: ?[*:0]const u8,

        pub fn format(
            self: @This(),
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;

            try writer.print("{s}[{s}] ", .{ prefix, self.kernel_name });
            if (self.err) |err| {
                try writer.print("failed: {s}", .{err});
            } else {
                try writer.print("took {d:.3} ms", .{self.duration_ms});
            }
        }
    };
}

pub fn track(
    comptime kernel_name: []const u8,
    comptime ResultType: type,
    kernel_fn: anytype,
    args: anytype,
) !ResultType {
    const tracked = try trackKernelExecution(kernel_name, ResultType, kernel_fn, args);
    std.debug.print("{}\n", .{tracked.stats});
    if (tracked.stats.err != null) {
        std.debug.print("Kernel Error Detected! {any}\n", .{tracked.stats.err});
        return error.CudaKernelError;
    }
    return tracked.result;
}

pub fn trackKernelExecution(
    comptime kernel_name: []const u8,
    comptime ResultType: type,
    kernel_fn: anytype,
    args: anytype,
) !struct { result: ResultType, stats: CudaKernelError("CUDA") } {
    var start: cuda.cudaEvent_t = undefined;
    var stop: cuda.cudaEvent_t = undefined;

    // Create timing events
    var err = cuda.cudaEventCreate(&start);
    if (err != cuda.cudaSuccess) {
        return error.CudaEventCreateFailed;
    }
    err = cuda.cudaEventCreate(&stop);
    if (err != cuda.cudaSuccess) {
        _ = cuda.cudaEventDestroy(start);
        return error.CudaEventCreateFailed;
    }
    defer _ = cuda.cudaEventDestroy(start);
    defer _ = cuda.cudaEventDestroy(stop);

    // Record start time
    err = cuda.cudaEventRecord(start, null);
    if (err != cuda.cudaSuccess) {
        return error.CudaEventRecordFailed;
    }

    const result = @call(.auto, kernel_fn, args);

    // Record stop time
    err = cuda.cudaEventRecord(stop, null);
    if (err != cuda.cudaSuccess) {
        return error.CudaEventRecordFailed;
    }

    err = cuda.cudaEventSynchronize(stop);
    if (err != cuda.cudaSuccess) {
        return error.CudaEventSyncFailed;
    }

    // Calculate elapsed time
    var ms: f32 = undefined;
    err = cuda.cudaEventElapsedTime(&ms, start, stop);
    if (err != cuda.cudaSuccess) {
        return error.CudaEventElapsedTimeFailed;
    }

    // Check for kernel errors
    err = cuda.cudaGetLastError();
    const error_str = if (err != cuda.cudaSuccess) cuda.cudaGetErrorString(err) else null;

    return .{
        .result = result,
        .stats = .{
            .kernel_name = kernel_name,
            .duration_ms = ms,
            .err = error_str,
        },
    };
}
