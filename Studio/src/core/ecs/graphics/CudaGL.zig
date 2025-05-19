const std = @import("std");
const c = @import("../../bindings/c.zig");
const gl = @import("../../bindings/gl.zig");
const glad = gl.glad;
const cuda = c.cuda;

pub const CUDAGLTexture = struct {
    gl_tex: u32,
    graphics: cuda.cudaGraphicsResource_t, // ← keep the graphics‑interop handle
    cuda_arr: cuda.cudaArray_t, // cached CUDA array view of GL tex
    gpu_ptr: [*c]u8, // linear buffer we export to Python
    ipc: cuda.cudaIpcMemHandle_t,
    pitch: c_int,
    width: c_int,
    height: c_int,
    size: u64,
    frame: u64 = 0,

    pub fn copyFromGL(self: *CUDAGLTexture) void {
        checkCuda(cuda.cudaGraphicsMapResources(1, &self.graphics, null), "cuGraphicsMapResources");

        var arr: cuda.cudaArray_t = undefined;
        checkCuda(cuda.cudaGraphicsSubResourceGetMappedArray(&arr, self.graphics, 0, 0), "cuGraphicsSubResourceGetMappedArray");

        checkCuda(cuda.cudaMemcpy2DFromArray(self.gpu_ptr, // dst
            @intCast(self.pitch), // dpitch - destination pitch
            arr, // src - the CUDA array (GL texture)
            0, 0, // wOffset, hOffset - start at the beginning
            @intCast(self.width * 4), // width in bytes (RGBA = 4 bytes per pixel)
            @intCast(self.height), // height
            cuda.cudaMemcpyDeviceToDevice // both source and dest are on GPU
        ), "cudaMemcpy2DFromArray"); // checkCuda(cuda.cuMemcpy2DAsync(&copy, stream), "cuMemcpy2DAsync");

        checkCuda(cuda.cudaGraphicsUnmapResources(1, &self.graphics, null), "cuGraphicsUnmapResources");

        self.frame = (self.frame + 1) % @as(u64, @intCast(self.height));
    }

    pub fn deinit(self: *CUDAGLTexture) void {
        _ = cuda.cudaFree(self.gpu_ptr);
        _ = cuda.cudaGraphicsUnregisterResource(self.graphics);
        glad.glDeleteTextures(1, &self.gl_tex);
    }
};

inline fn checkCuda(res: cuda.CUresult, msg: []const u8) void {
    if (res != cuda.CUDA_SUCCESS) {
        std.debug.print("CUDA ‼ {s} failed (code {d})\nName: {s} => Message: {s}\n", .{
            msg,
            res,
            cuda.cudaGetErrorName(res),
            cuda.cudaGetErrorString(res),
        });
        @panic("CUDA call failed");
    } else {
        // std.debug.print("CUDA call: {s} executed successfully...\n", .{msg});
    }
}

var ctx_initialised: bool = false;
fn ensureCudaContext() void {
    if (@atomicLoad(bool, &ctx_initialised, .acquire)) return;
    checkCuda(cuda.cuInit(0), "cuInit");

    var dev: cuda.CUdevice = undefined;
    checkCuda(cuda.cuDeviceGet(&dev, 0), "cuDeviceGet(0)");

    var ctx: cuda.CUcontext = undefined;
    checkCuda(cuda.cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

    @atomicStore(bool, &ctx_initialised, true, .release);

    std.debug.print("CUDA Info => \n\tDevice: {d}\n\tContext: {any}\n", .{ dev, ctx });
}

pub fn createCUDAGLTexture(
    width: c_int,
    height: c_int,
    texture_id: glad.GLuint,
) CUDAGLTexture {
    ensureCudaContext();

    var tex: CUDAGLTexture = undefined;
    tex.width = width;
    tex.height = height;
    tex.pitch = width * 4; // RGBA8
    tex.gl_tex = texture_id;
    tex.size = (@as(u64, @intCast(tex.pitch))) *
        (@as(u64, @intCast(height)));

    std.debug.print("createCUDAGLTexture: FBO:{d} width={d}  height={d}  size={d}\n", .{
        tex.gl_tex,
        tex.width,
        tex.height,
        tex.size,
    });

    // Register with CUDA graphics interop
    checkCuda(
        cuda.cudaGraphicsGLRegisterImage(
            &tex.graphics,
            tex.gl_tex,
            glad.GL_TEXTURE_2D,
            cuda.cudaGraphicsRegisterFlagsReadOnly,
        ),
        "cuGraphicsGLRegisterImage",
    );

    checkCuda(cuda.cudaMalloc(&tex.gpu_ptr, tex.size), "cuMemAlloc");
    checkCuda(cuda.cudaIpcGetMemHandle(&tex.ipc, tex.gpu_ptr), "cuIpcGetMemHandle");

    return tex;
}

pub fn inspectFBO(fbo_id: c_uint, width: c_int, height: c_int) void {
    glad.glBindFramebuffer(glad.GL_READ_FRAMEBUFFER, fbo_id);

    const buffer_size: usize = @intCast(width * height * 4); // RGBA8
    const pixels = std.heap.c_allocator.alloc(u8, buffer_size) catch {
        std.debug.print("Failed to allocate memory for FBO inspection\n", .{});
        return;
    };
    defer std.heap.c_allocator.free(pixels);

    glad.glReadPixels(0, 0, width, height, glad.GL_RGBA, glad.GL_UNSIGNED_BYTE, pixels.ptr);

    const err = glad.glGetError();
    if (err != glad.GL_NO_ERROR) {
        std.debug.print("OpenGL error during glReadPixels: 0x{X}\n", .{err});
        return;
    }

    std.debug.print("FBO Contents (first few pixels):\n", .{});
    for (0..5) |y| {
        for (0..5) |x| {
            const idx = (y * @as(usize, @intCast(width)) + x) * 4;
            if (idx + 3 < pixels.len) {
                std.debug.print("Pixel ({d},{d}): RGBA = [{d}, {d}, {d}, {d}]\n", .{
                    x,
                    y,
                    pixels[idx],
                    pixels[idx + 1],
                    pixels[idx + 2],
                    pixels[idx + 3],
                });
            }
        }
    }

    var is_blank = true;
    for (pixels) |val| {
        if (val != 0) {
            is_blank = false;
            break;
        }
    }

    if (is_blank) {
        std.debug.print("WARNING: FBO appears to be completely blank!\n", .{});
    }
}

pub fn inspectCUDABuffer(gpu_ptr: [*c]u8, width: c_int, height: c_int, pitch: c_int) void {
    const buffer_size: usize = @intCast(height * pitch);
    const pixels = std.heap.c_allocator.alloc(u8, buffer_size) catch {
        std.debug.print("Failed to allocate memory for CUDA buffer inspection\n", .{});
        return;
    };
    defer std.heap.c_allocator.free(pixels);

    const result = cuda.cudaMemcpy2D(pixels.ptr, // dst
        @intCast(pitch), // dpitch
        gpu_ptr, // src
        @intCast(pitch), // spitch
        @intCast(width * 4), // width in bytes (RGBA)
        @intCast(height), // height
        cuda.cudaMemcpyDeviceToHost // GPU to CPU
    );

    if (result != cuda.cudaSuccess) {
        std.debug.print("CUDA error during cudaMemcpy2D: {s}\n", .{cuda.cudaGetErrorString(result)});
        return;
    }

    std.debug.print("CUDA Buffer Contents (first few pixels):\n", .{});
    for (0..5) |y| {
        for (0..5) |x| {
            const idx = (y * @as(usize, @intCast(pitch))) + (x * 4);
            if (idx + 3 < pixels.len) {
                std.debug.print("Pixel ({d},{d}): RGBA = [{d}, {d}, {d}, {d}]\n", .{
                    x,
                    y,
                    pixels[idx],
                    pixels[idx + 1],
                    pixels[idx + 2],
                    pixels[idx + 3],
                });
            }
        }
    }

    // Check if the buffer is just a solid color
    const first_pixel = [4]u8{ pixels[0], pixels[1], pixels[2], pixels[3] };
    var is_solid = true;

    pixel_check: for (0..@intCast(height)) |y| {
        for (0..@intCast(width)) |x| {
            const idx = (y * @as(usize, @intCast(pitch))) + (x * 4);
            if (idx + 3 < pixels.len) {
                for (0..4) |_c| {
                    if (pixels[idx + _c] != first_pixel[_c]) {
                        is_solid = false;
                        break :pixel_check;
                    }
                }
            }
        }
    }

    if (is_solid) {
        std.debug.print("NOTE: CUDA buffer appears to be a solid color: RGBA = [{d}, {d}, {d}, {d}]\n", .{ first_pixel[0], first_pixel[1], first_pixel[2], first_pixel[3] });
    }
}
