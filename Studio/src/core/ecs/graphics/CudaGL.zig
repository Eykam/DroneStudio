const std = @import("std");
const c = @import("../../bindings/c.zig");
const gl = @import("../../bindings/gl.zig");
const glad = gl.glad;
const cuda = c.cuda;

pub const CUDAGLTexture = struct {
    gl_tex: u32,
    graphics: cuda.cudaGraphicsResource_t,
    // Double buffering: main thread writes to buffers[write_idx], SLAM reads from buffers[1-write_idx]
    buffers: [2][*c]u8,
    ipc: [2]cuda.cudaIpcMemHandle_t,
    write_idx: std.atomic.Value(u8), // Which buffer main thread writes to
    queued_frame: std.atomic.Value(u64), // Frame number of the latest queued frame
    pitch: c_int,
    width: c_int,
    height: c_int,
    size: u64,
    frame: u64 = 0, // Track which frame we last copied

    const Self = @This();

    /// Initialize a CUDAGLTexture with double-buffering for async processing.
    pub fn init(width: c_int, height: c_int, texture_id: glad.GLuint) Self {
        ensureCudaContext();

        // Ensure GL commands are complete before registering with CUDA
        glad.glFinish();

        // Verify the texture exists in OpenGL
        if (glad.glIsTexture(texture_id) == glad.GL_FALSE) {
            std.debug.print("ERROR: texture_id {d} is not a valid OpenGL texture!\n", .{texture_id});
            @panic("Invalid OpenGL texture for CUDA-GL interop");
        }

        const pitch = width * 4; // RGBA8
        const size: u64 = (@as(u64, @intCast(pitch))) * (@as(u64, @intCast(height)));

        std.debug.print("CUDAGLTexture.init: texture_id={d} width={d} height={d} size={d}\n", .{
            texture_id,
            width,
            height,
            size,
        });

        var self = Self{
            .gl_tex = texture_id,
            .graphics = undefined,
            .buffers = .{ undefined, undefined },
            .ipc = .{ undefined, undefined },
            .write_idx = std.atomic.Value(u8).init(0),
            .queued_frame = std.atomic.Value(u64).init(0),
            .pitch = pitch,
            .width = width,
            .height = height,
            .size = size,
            .frame = 0,
        };

        // Register with CUDA graphics interop
        checkCuda(
            cuda.cudaGraphicsGLRegisterImage(
                &self.graphics,
                self.gl_tex,
                glad.GL_TEXTURE_2D,
                cuda.cudaGraphicsRegisterFlagsReadOnly,
            ),
            "cuGraphicsGLRegisterImage",
        );

        // Allocate double buffers
        checkCuda(cuda.cudaMalloc(&self.buffers[0], self.size), "cuMemAlloc buffer[0]");
        checkCuda(cuda.cudaMalloc(&self.buffers[1], self.size), "cuMemAlloc buffer[1]");
        checkCuda(cuda.cudaIpcGetMemHandle(&self.ipc[0], self.buffers[0]), "cuIpcGetMemHandle[0]");
        checkCuda(cuda.cudaIpcGetMemHandle(&self.ipc[1], self.buffers[1]), "cuIpcGetMemHandle[1]");

        return self;
    }

    /// Copy GL texture to the write buffer. Called by main thread.
    /// Returns true if copy was performed, false if already copied this frame.
    pub fn copyFromGL(self: *Self, current_frame: u64) bool {
        // Skip if already copied this frame
        if (self.frame == current_frame) {
            return false;
        }

        // Ensure all GL rendering is complete before CUDA access
        glad.glFlush();
        glad.glFinish();

        // Check OpenGL for errors
        const gl_err = glad.glGetError();
        if (gl_err != glad.GL_NO_ERROR) {
            std.debug.print("OpenGL error before CUDA map: 0x{x}\n", .{gl_err});
        }

        // Check for sticky errors from previous CUDA operations
        const prev_err = cuda.cudaGetLastError();
        if (prev_err != cuda.cudaSuccess) {
            std.debug.print("CUDA: Sticky error before copyFromGL: {s}\n", .{cuda.cudaGetErrorString(prev_err)});
        }

        // Get current write buffer
        const idx = self.write_idx.load(.acquire);

        std.debug.print("Copying Frame from GL => Cuda. Texture: {d}, buffer: {d}, ecs_frame: {d}\n", .{
            self.gl_tex,
            idx,
            current_frame,
        });

        checkCuda(cuda.cudaGraphicsMapResources(1, &self.graphics, null), "cuGraphicsMapResources");

        var arr: cuda.cudaArray_t = undefined;
        checkCuda(cuda.cudaGraphicsSubResourceGetMappedArray(
            &arr,
            self.graphics,
            0,
            0,
        ), "cuGraphicsSubResourceGetMappedArray");

        checkCuda(cuda.cudaMemcpy2DFromArray(
            self.buffers[idx], // dst - current write buffer
            @intCast(self.pitch),
            arr,
            0,
            0,
            @intCast(self.width * 4),
            @intCast(self.height),
            cuda.cudaMemcpyDeviceToDevice,
        ), "cudaMemcpy2DFromArray");

        checkCuda(cuda.cudaGraphicsUnmapResources(1, &self.graphics, null), "cuGraphicsUnmapResources");

        // Ensure copy is complete
        checkCuda(cuda.cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        // Mark this frame as queued
        self.queued_frame.store(current_frame, .release);
        self.frame = current_frame;
        return true;
    }

    /// Swap buffers and return pointer for processing. Called by SLAM thread.
    /// Returns the buffer containing the latest complete frame.
    pub fn acquireForProcessing(self: *Self) [*c]u8 {
        // Flip the write index - main thread will now write to the other buffer
        const old_idx = self.write_idx.fetchXor(1, .acq_rel);
        // Return the buffer that was being written to (now safe to read)
        return self.buffers[old_idx];
    }

    /// Get the current queued frame number
    pub fn getQueuedFrame(self: *Self) u64 {
        return self.queued_frame.load(.acquire);
    }

    /// Get the current write buffer pointer (for reading frame data without swapping)
    pub fn getWriteBuffer(self: *Self) [*c]u8 {
        const idx = self.write_idx.load(.acquire);
        return self.buffers[idx];
    }

    pub fn deinit(self: *CUDAGLTexture) void {
        std.debug.print("CUDAGLTexture.deinit called for texture {d}\n", .{self.gl_tex});
        _ = cuda.cudaFree(self.buffers[0]);
        _ = cuda.cudaFree(self.buffers[1]);
        _ = cuda.cudaGraphicsUnregisterResource(self.graphics);
    }
};

inline fn checkCuda(res: cuda.cudaError_t, msg: []const u8) void {
    if (res != cuda.cudaSuccess) {
        std.debug.print("CUDA ‼ {s} failed (code {d})\nName: {s} => Message: {s}\n", .{
            msg,
            res,
            cuda.cudaGetErrorName(res),
            cuda.cudaGetErrorString(res),
        });
        @panic("CUDA call failed");
    }
}

var ctx_initialised: bool = false;
fn ensureCudaContext() void {
    if (@atomicLoad(bool, &ctx_initialised, .acquire)) return;

    // Use runtime API to set up CUDA-GL interop
    var device_count: c_int = 0;
    const err = cuda.cudaGetDeviceCount(&device_count);
    if (err != cuda.cudaSuccess or device_count == 0) {
        std.debug.print("CUDA: No devices found or error: {s}\n", .{cuda.cudaGetErrorString(err)});
        return;
    }

    // Set device 0 - this initializes the runtime API context
    const set_err = cuda.cudaSetDevice(0);
    if (set_err != cuda.cudaSuccess) {
        std.debug.print("CUDA: Failed to set device: {s}\n", .{cuda.cudaGetErrorString(set_err)});
        return;
    }

    @atomicStore(bool, &ctx_initialised, true, .release);

    std.debug.print("CUDA Info => Device count: {d}, using device 0\n", .{device_count});
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

// ============================================================================
// Kernel tracking - wraps CUDA kernel calls with timing and error checking
// ============================================================================

pub fn KernelStats(comptime prefix: []const u8) type {
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

/// Track a CUDA kernel call with timing and error checking.
/// Panics if kernel fails.
pub fn track(
    comptime kernel_name: []const u8,
    comptime ResultType: type,
    kernel_fn: anytype,
    args: anytype,
) !ResultType {
    const tracked = trackKernelExecution(kernel_name, ResultType, kernel_fn, args) catch |err| {
        std.debug.print("CUDA[{s}] TRACKING FAILED: {any}\n", .{ kernel_name, err });
        @panic("CUDA kernel tracking failed");
    };
    std.debug.print("{}\n", .{tracked.stats});
    if (tracked.stats.err) |err| {
        std.debug.print("CUDA[{s}] KERNEL ERROR: {s}\n", .{ kernel_name, err });
        @panic("CUDA kernel execution failed");
    }
    return tracked.result;
}

/// Track kernel execution with timing, returning both result and stats.
pub fn trackKernelExecution(
    comptime kernel_name: []const u8,
    comptime ResultType: type,
    kernel_fn: anytype,
    args: anytype,
) !struct { result: ResultType, stats: KernelStats("CUDA") } {
    // Check for pre-existing CUDA errors before starting
    const pre_err = cuda.cudaGetLastError();
    if (pre_err != cuda.cudaSuccess) {
        std.debug.print("CUDA[{s}] pre-existing error: {s}\n", .{ kernel_name, cuda.cudaGetErrorString(pre_err) });
        return error.CudaPreExistingError;
    }

    var start: cuda.cudaEvent_t = undefined;
    var stop: cuda.cudaEvent_t = undefined;

    // Create timing events
    var err = cuda.cudaEventCreate(&start);
    if (err != cuda.cudaSuccess) {
        std.debug.print("CUDA[{s}] cudaEventCreate(start) failed: {s}\n", .{ kernel_name, cuda.cudaGetErrorString(err) });
        return error.CudaEventCreateFailed;
    }
    err = cuda.cudaEventCreate(&stop);
    if (err != cuda.cudaSuccess) {
        std.debug.print("CUDA[{s}] cudaEventCreate(stop) failed: {s}\n", .{ kernel_name, cuda.cudaGetErrorString(err) });
        _ = cuda.cudaEventDestroy(start);
        return error.CudaEventCreateFailed;
    }
    defer _ = cuda.cudaEventDestroy(start);
    defer _ = cuda.cudaEventDestroy(stop);

    // Record start time
    err = cuda.cudaEventRecord(start, null);
    if (err != cuda.cudaSuccess) {
        std.debug.print("CUDA[{s}] cudaEventRecord(start) failed: {s}\n", .{ kernel_name, cuda.cudaGetErrorString(err) });
        return error.CudaEventRecordFailed;
    }

    const result = @call(.auto, kernel_fn, args);

    // Record stop time
    err = cuda.cudaEventRecord(stop, null);
    if (err != cuda.cudaSuccess) {
        std.debug.print("CUDA[{s}] cudaEventRecord(stop) failed: {s}\n", .{ kernel_name, cuda.cudaGetErrorString(err) });
        return error.CudaEventRecordFailed;
    }

    err = cuda.cudaEventSynchronize(stop);
    if (err != cuda.cudaSuccess) {
        std.debug.print("CUDA[{s}] cudaEventSynchronize failed: {s}\n", .{ kernel_name, cuda.cudaGetErrorString(err) });
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
