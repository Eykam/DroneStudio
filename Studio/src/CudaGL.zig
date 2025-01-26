const std = @import("std");
const c = @import("bindings/c.zig");
const glad = @import("bindings/gl.zig").glad;
const cuda = c.cuda;

pub const InstanceBuffer = struct {
    position_resource: cuda.cudaGraphicsResource_t,
    color_resource: cuda.cudaGraphicsResource_t,
    d_positions: *cuda.float4,
    d_colors: *cuda.float4,
};

pub const ConnectedKeypoint = struct {
    left: InstanceBuffer,
    right: InstanceBuffer,
};

pub const CudaGLResources = struct {
    connections: ConnectedKeypoint,
    keypoints: InstanceBuffer,
};

pub const CudaGLTextureResource = struct {
    tex_resource: cuda.cudaGraphicsResource_t,
    array: cuda.cudaArray_t,
    surface: cuda.cudaSurfaceObject_t,
};

pub const DetectorInstance = struct {
    const Self = @This();

    d_keypoint_count: [*c]u32,
    gl_ytexture: u32,
    gl_uvtexture: u32,
    gl_depthtexture: u32,
    world_transform: [16]f32,
    d_world_transform: [*c]f32,
    gl_resources: CudaGLResources,
    y_texture: ?*CudaGLTextureResource,
    uv_texture: ?*CudaGLTextureResource,
    depth_texture: ?*CudaGLTextureResource,
    d_descriptors: [*c]cuda.BRIEFDescriptor,
    id: u32,
    initialized: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        id: u32,
        max_keypoints: u32,
        y_texture: glad.GLuint,
        uv_texture: glad.GLuint,
        depth_texture: glad.GLuint,
    ) *Self {
        const detector = try allocator.create(Self);

        var err = cuda.cudaMalloc(detector.d_keypoint_count, @sizeOf(u32));
        err |= cuda.cudaMalloc(detector.d_descriptors, max_keypoints * @sizeOf(cuda.BRIEFDescriptor));
        err |= cuda.cudaMalloc(detector.d_world_transform, 16 * @sizeOf(f32));

        const identity = [16]f32{
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        };

        err |= cuda.cudaMemcpy(detector.d_world_transform, identity, identity.len * @sizeOf(f32), cuda.cudaMemcpyHostToDevice);

        detector.initialized = true;
        detector.id = id;
        detector.gl_ytexture = y_texture;
        detector.gl_uvtexture = uv_texture;
        detector.gl_depthtexture = depth_texture;

        return detector;
    }

    pub fn deinit(self: *Self) void {
        //         void cuda_cleanup_detector(int detector_id) {
        //     DetectorInstance* detector = get_detector_instance(detector_id);
        //     if (!detector) return;

        //     cuda_unregister_buffers(detector_id);

        //     if (detector->d_keypoint_count) cudaFree(detector->d_keypoint_count);
        //     if (detector->d_descriptors) cudaFree(detector->d_descriptors);
        //     detector->d_keypoint_count = nullptr;
        //     detector->d_descriptors = nullptr;

        //     if (detector->d_world_transform) {
        //         cudaFree(detector->d_world_transform);
        //         detector->d_world_transform = nullptr;
        //     }
        // }
        _ = self;
    }

    pub fn map_transformation(self: *Self, transformation: [16]f32) !void {
        const err = cuda.cudaMemcpy(
            self.world_transform,
            transformation,
            transformation.len * @sizeOf(f32),
            cuda.cudaMemcpyHostToDevice,
        );

        if (err != cuda.cudaSuccess) {
            std.debug.print("Transform copy to device failed: {s}\n", .{cuda.cudaGetErrorString(err)});
            return error.MapTransformationFailed;
        }
    }

    fn register_texture(self: *Self, texture: CudaGLTextureResource) void {
        self.y
    }

    pub fn register_textures(self: *Self) void {

        //         int cuda_register_gl_texture(int detector_id) {
        //     DetectorInstance* detector = get_detector_instance(detector_id);
        //     if (!detector) return -1;

        //     // Allocate texture resource
        //     detector->y_texture = (CudaGLTextureResource*)malloc(sizeof(CudaGLTextureResource));
        //     if (!detector->y_texture) {
        //         printf("Failed to allocate Y texture resource\n");
        //         return -1;
        //     }
        //     memset(detector->y_texture, 0, sizeof(CudaGLTextureResource));

        //     // Register Y texture
        //     cudaError_t err = cudaGraphicsGLRegisterImage(
        //         &detector->y_texture->tex_resource,
        //         detector->gl_ytexture,
        //         GL_TEXTURE_2D,
        //         cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard
        //     );

        //     if (err != cudaSuccess) {
        //         printf("Failed to register Y texture: %s\n", cudaGetErrorString(err));
        //         free(detector->y_texture);
        //         detector->y_texture = NULL;
        //         return -1;
        //     }

        //     // Allocate UV texture resource
        //     detector->uv_texture = (CudaGLTextureResource*)malloc(sizeof(CudaGLTextureResource));
        //     if (!detector->uv_texture) {
        //         printf("Failed to allocate UV texture resource\n");
        //         cudaGraphicsUnregisterResource(detector->y_texture->tex_resource);
        //         free(detector->y_texture);
        //         detector->y_texture = NULL;
        //         return -1;
        //     }
        //     memset(detector->uv_texture, 0, sizeof(CudaGLTextureResource));

        //     // Register UV texture
        //     err = cudaGraphicsGLRegisterImage(
        //         &detector->uv_texture->tex_resource,
        //         detector->gl_uvtexture,
        //         GL_TEXTURE_2D,
        //         cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard
        //     );

        //     if (err != cudaSuccess) {
        //         printf("Failed to register UV texture: %s\n", cudaGetErrorString(err));
        //         cudaGraphicsUnregisterResource(detector->y_texture->tex_resource);
        //         free(detector->y_texture);
        //         detector->y_texture = NULL;
        //         free(detector->uv_texture);
        //         detector->uv_texture = NULL;
        //         return -1;
        //     }

        //     // Allocate Depth texture resource
        //     detector->depth_texture = (CudaGLTextureResource*)malloc(sizeof(CudaGLTextureResource));
        //     if (!detector->depth_texture) {
        //         printf("Failed to allocate Depth texture resource\n");
        //         cudaGraphicsUnregisterResource(detector->y_texture->tex_resource);
        //         free(detector->y_texture);
        //         detector->y_texture = NULL;
        //         cudaGraphicsUnregisterResource(detector->uv_texture->tex_resource);
        //         free(detector->uv_texture);
        //         detector->uv_texture = NULL;
        //         return -1;
        //     }
        //     memset(detector->depth_texture, 0, sizeof(CudaGLTextureResource));

        //     // Register Depth texture
        //     err = cudaGraphicsGLRegisterImage(
        //         &detector->depth_texture->tex_resource,
        //         detector->gl_depthtexture,
        //         GL_TEXTURE_2D,
        //         cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard
        //     );

        //     if (err != cudaSuccess) {
        //         printf("Failed to register UV texture: %s\n", cudaGetErrorString(err));
        //         cudaGraphicsUnregisterResource(detector->y_texture->tex_resource);
        //         free(detector->y_texture);
        //         detector->y_texture = NULL;
        //         cudaGraphicsUnregisterResource(detector->uv_texture->tex_resource);
        //         free(detector->uv_texture);
        //         detector->uv_texture = NULL;
        //         free(detector->depth_texture);
        //         detector->depth_texture = NULL;
        //         return -1;
        //     }

        //     return 0;
        // }
    }

    pub fn unregister_texture(self: *Self) void {
        _ = self;
    }

    pub fn register_instance_buffer(self: *Self) void {
        _ = self;
    }

    pub fn register_buffers(self: *Self) void {
        _ = self;
    }

    pub fn unregister_buffers(self: *Self) void {
        _ = self;
    }

    pub fn map_resources(self: *Self) void {
        _ = self;
    }

    pub fn unmap_resources(self: *Self) void {
        _ = self;
    }
};
