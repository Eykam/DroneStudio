const std = @import("std");
const c = @cImport({
    @cInclude("libavfilter/avfilter.h");
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavformat/avformat.h");
    @cInclude("libavutil/frame.h");
    @cInclude("libavutil/hwcontext.h");
    @cInclude("libavutil/hwcontext_cuda.h");
    @cInclude("libavutil/imgutils.h");
    @cInclude("libavutil/pixdesc.h");
    @cInclude("libswscale/swscale.h");
    @cInclude("libavfilter/buffersink.h");
    @cInclude("libavfilter/buffersrc.h");
    @cInclude("libavutil/opt.h");
});
const Node = @import("Node.zig");

const mem = std.mem;
const fs = std.fs;

// Define custom errors
const errors = enum {
    CodecNotFound,
    ParserInitFailed,
    CodecContextAllocationFailed,
    CodecOpenFailed,
    FrameAllocationFailed,
    PacketAllocationFailed,
    PNGCodecNotFound,
    PNGCodecContextAllocationFailed,
    PNGCodecOpenFailed,
    PNGFrameAllocationFailed,
    PNGFrameBufferAllocationFailed,
    PNGPacketAllocationFailed,
    SwsContextAllocationFailed,
    ParsingFailed,
    DecodeFailed,
    FrameReceiveFailed,
    SwsScaleFailed,
    PNGFrameSendFailed,
    PNGFrameReceiveFailed,
    BufferResizeFailed,
    BufferOverflow,
};

const AVERROR_EAGAIN = -1 * @as(c_int, @intCast(c.EAGAIN));
const AVERROR_EOF = -1 * @as(c_int, @intCast(c.EOF));

const DecodedFrameCallback = *const fn (mem.Allocator, *Node, *c.struct_AVFrame) error{OutOfMemory}!void;

pub const VideoHandler = struct {
    const Self = @This();

    allocator: mem.Allocator,
    node: *Node,

    buffer: std.ArrayList(u8),
    current_timestamp: u128 = 0,
    nalu_start_codes: [4]u8 = [_]u8{ 0, 0, 0, 1 },
    image_dir: fs.Dir,
    stale_count: u8 = 0,
    last_valid_timestamp: u128 = 0,
    prev_frame_id: ?u32 = null,
    frame_delta: i128 = 0,
    packet: *c.AVPacket,

    // FFmpeg Decoder Fields
    parser_context: *c.AVCodecParserContext,
    codec_context: *c.AVCodecContext,
    hw_frame: *c.AVFrame,

    hw_device_ctx: ?*c.AVBufferRef = null,

    filter_graph: ?*c.AVFilterGraph = null,
    buffersrc: ?*c.AVFilterContext = null,
    buffersink: ?*c.AVFilterContext = null,

    // Callback for decoded RGB frames
    onDecodedFrame: DecodedFrameCallback,

    pub fn init(allocator: mem.Allocator, node: *Node, hw_type: ?c_uint, onDecodedFrame: DecodedFrameCallback) !Self {
        // Create images directory if it doesn't exist
        var cwd = fs.cwd();
        cwd.makeDir("images") catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        const image_dir = try cwd.openDir("images", .{});

        const selected_hw_type = hw_type orelse detectBestHardwareDevice();

        var hw_device_ctx: ?*c.AVBufferRef = null;
        const hw_device_type: c.AVHWDeviceType = selected_hw_type;

        if (hw_device_type == c.AV_HWDEVICE_TYPE_NONE) {
            std.debug.print("No hardware decoding device found. Falling back to software decoding.\n", .{});
            return error.NoHardwareDevice;
        }

        const hw_device_type_name = c.av_hwdevice_get_type_name(hw_device_type);
        std.debug.print("Hardware Device Name: {s}\n", .{hw_device_type_name});

        // Create hardware device context
        if (c.av_hwdevice_ctx_create(&hw_device_ctx, hw_device_type, null, null, 0) < 0) {
            std.debug.print("Failed to create hardware device context\n", .{});
            return error.HardwareDeviceContextCreationFailed;
        }

        // Find hardware-accelerated H264 decoder
        const codec = c.avcodec_find_decoder(c.AV_CODEC_ID_H264);
        if (codec == null) {
            std.debug.print("Hardware H264 decoder not found\n", .{});
            return error.HardwareDecoderNotFound;
        }

        const parser_context = c.av_parser_init(@intCast(codec.*.id)) orelse return error.ParserInitFailed;
        errdefer c.av_parser_close(parser_context); // Ensure parser_context is closed on error

        const codec_context = c.avcodec_alloc_context3(codec);
        if (codec_context == null) return error.CodecContextAllocationFailed;
        errdefer c.avcodec_free_context(@ptrCast(@constCast(&codec_context))); // Ensure codec_context is freed on error

        codec_context.*.hw_device_ctx = hw_device_ctx;
        codec_context.*.width = 1280;
        codec_context.*.height = 720;

        codec_context.*.pix_fmt = switch (hw_device_type) {
            c.AV_HWDEVICE_TYPE_CUDA => c.AV_PIX_FMT_CUDA,
            c.AV_HWDEVICE_TYPE_VAAPI => c.AV_PIX_FMT_VAAPI,
            c.AV_HWDEVICE_TYPE_D3D11VA => c.AV_PIX_FMT_D3D11,
            c.AV_HWDEVICE_TYPE_VIDEOTOOLBOX => c.AV_PIX_FMT_VIDEOTOOLBOX,
            else => c.AV_PIX_FMT_YUV420P,
        };
        codec_context.*.flags |= c.AV_CODEC_FLAG_LOW_DELAY; // Reduce latency
        codec_context.*.max_b_frames = 0; // Minimize B-frame processing overhead

        if (hw_device_type == c.AV_HWDEVICE_TYPE_CUDA) {
            codec_context.*.extra_hw_frames = 4; // Preallocate some frames
        }

        if (c.avcodec_open2(codec_context, codec, null) < 0) {
            return error.CodecOpenFailed;
        }

        const hw_frame = c.av_frame_alloc() orelse return error.FrameAllocationFailed;

        const packet = c.av_packet_alloc() orelse return error.PacketAllocationFailed;
        errdefer c.av_packet_free(@ptrCast(@constCast(&packet))); // Ensure packet is freed on error

        // Create a new instance without defers
        var self = Self{
            .allocator = allocator,
            .node = node,
            .buffer = std.ArrayList(u8).init(allocator),
            .image_dir = image_dir,
            .parser_context = parser_context,
            .codec_context = codec_context,
            .hw_frame = hw_frame,
            .packet = packet,
            .onDecodedFrame = onDecodedFrame,
        };
        try self.initHardwareFilterGraph();
        return self;
    }

    pub fn initHardwareFilterGraph(self: *Self) !void {
        const filter_graph = c.avfilter_graph_alloc();
        if (filter_graph == null) {
            std.debug.print("Failed to allocate filter graph.\n", .{});
            return error.FilterGraphAllocationFailed;
        }
        errdefer c.avfilter_graph_free(@ptrCast(@constCast(&filter_graph)));

        // Create a new hardware frames context if not already present
        if (self.codec_context.hw_device_ctx == null) {
            std.debug.print("No hardware device context available.\n", .{});
            return error.NoHardwareDeviceContext;
        }

        // Create a new hardware frames context if not already present
        const hw_frames_ctx = c.av_hwframe_ctx_alloc(self.codec_context.hw_device_ctx) orelse {
            std.debug.print("Failed to allocate hardware frames context.\n", .{});
            return error.HardwareFramesContextAllocationFailed;
        };

        // Configure the hardware frames context
        const frames_ctx = @as(*c.AVHWFramesContext, @ptrCast(@alignCast(hw_frames_ctx.*.data)));
        frames_ctx.format = c.AV_PIX_FMT_CUDA; // Adjust based on your hardware type
        frames_ctx.sw_format = c.AV_PIX_FMT_NV12;
        frames_ctx.width = self.codec_context.width;
        frames_ctx.height = self.codec_context.height;
        frames_ctx.initial_pool_size = 4; // Preallocate frames

        if (c.av_hwframe_ctx_init(hw_frames_ctx) < 0) {
            std.debug.print("Failed to initialize hardware frames context.\n", .{});
            c.av_buffer_unref(@constCast(&hw_frames_ctx));
            return error.HardwareFramesContextInitFailed;
        }

        if (hw_frames_ctx == null) {
            std.debug.print("Hardware frames context is null after initialization\n", .{});
            return error.HardwareFramesContextAllocationFailed;
        }

        // Create hwupload buffer source with explicit frames context
        const buffersrc = c.avfilter_graph_alloc_filter(filter_graph, c.avfilter_get_by_name("buffer"), "in");
        if (buffersrc == null) {
            std.debug.print("Failed to create buffer source filter.\n", .{});
            return error.FilterGraphAllocationFailed;
        }
        errdefer c.avfilter_free(@ptrCast(@constCast(buffersrc)));

        const params = c.av_buffersrc_parameters_alloc();
        params.*.hw_frames_ctx = hw_frames_ctx;
        params.*.format = self.codec_context.pix_fmt;
        params.*.height = self.codec_context.height;
        params.*.width = self.codec_context.width;
        params.*.time_base = .{
            .num = 1,
            .den = 1,
        };
        params.*.sample_aspect_ratio = .{
            .num = 1,
            .den = 1,
        };

        // Initialize buffer source with hardware context details

        if (c.av_buffersrc_parameters_set(buffersrc, params) < 0) {
            std.debug.print("Failed to set buffersrc params.\n", .{});
            return error.FilterGraphAllocationFailed;
        }
        if (c.avfilter_init_str(buffersrc, null) < 0) {
            std.debug.print("Failed to initialize buffer source filter.\n", .{});
            return error.FilterGraphAllocationFailed;
        }

        // Create hardware buffer sink
        const buffersink = c.avfilter_graph_alloc_filter(
            filter_graph,
            c.avfilter_get_by_name("buffersink"),
            "out",
        );
        if (buffersink == null) {
            std.debug.print("Failed to create buffer sink filter.\n", .{});
            return error.FilterGraphAllocationFailed;
        }
        errdefer c.avfilter_free(buffersink);

        if (c.avfilter_init_str(buffersink, null) < 0) {
            std.debug.print("Failed to initialize buffer sink filter.\n", .{});
            return error.FilterGraphAllocationFailed;
        }

        if (c.avfilter_link(buffersrc, 0, buffersink, 0) < 0) {
            std.debug.print("Failed to link buffer to hwupload filter.\n", .{});
            return error.FilterGraphAllocationFailed;
        }

        // Configure the filter graph
        if (c.avfilter_graph_config(filter_graph, null) < 0) {
            std.debug.print("Failed to configure filter graph.\n", .{});
            return error.FilterGraphConfigurationFailed;
        }

        // Store filter contexts
        self.filter_graph = filter_graph;
        self.buffersrc = buffersrc;
        self.buffersink = buffersink;

        if (self.buffersrc == null or self.buffersink == null) {
            std.debug.print("Failed to retrieve filter contexts.\n", .{});
            return error.FilterGraphAllocationFailed;
        }
    }

    pub fn deinit(self: *Self) void {
        self.buffer.deinit();
        self.image_dir.close();

        if (self.hw_device_ctx) |ctx| {
            c.av_buffer_unref(@ptrCast(@constCast(&ctx)));
        }

        if (self.filter_graph) |fg| {
            c.avfilter_graph_free(@ptrCast(@constCast(&fg)));
        }

        // Free FFmpeg Decoder Resources
        c.av_parser_close(self.parser_context);
        c.avcodec_free_context(@ptrCast(@constCast(&self.codec_context)));
        c.av_frame_free(@ptrCast(@constCast(&self.hw_frame)));
        c.av_packet_free(@ptrCast(@constCast(&self.packet)));
    }

    pub fn update(self: *Self, data: []const u8) !void {
        if (data.len <= 16) return;

        const frame_id: u32 = @bitCast(std.mem.readInt(u32, data[0..4], .big));
        const timestamp: u128 = @bitCast(std.mem.readInt(i128, data[4..20], .big));

        // Check for packet sequence and detect skipped or stale packets
        if (timestamp < self.last_valid_timestamp) {
            self.stale_count += 1;
            std.debug.print("Stale/Out-of-order Packet: {d}, Last Valid: {d}, Stale Count: {d}\n", .{ timestamp, self.last_valid_timestamp, self.stale_count });
            // Optional: Reset buffer on excessive stale packets
            if (self.stale_count > 10) {
                self.buffer.clearRetainingCapacity();
                self.stale_count = 0;
            }
            return;
        }

        if (self.prev_frame_id == null) {
            self.prev_frame_id = frame_id;
        } else {
            self.prev_frame_id.? += 1;
            if (frame_id != self.prev_frame_id.?) {
                // std.debug.print("Non-sequential Frames detected: Found => {}, Expected => {}\n", .{
                //     frame_id,
                //     self.prev_frame_id.?,
                // });
            }
        }

        self.last_valid_timestamp = timestamp;
        self.current_timestamp = timestamp;
        self.stale_count = 0;

        const video_data = data[20..];
        try self.buffer.appendSlice(video_data);

        try self.extractAndCaptureFrames();
    }

    fn extractAndCaptureFrames(self: *Self) !void {
        while (true) {
            const frame_result = try self.extractFrame();
            if (frame_result) |frame| {
                defer self.allocator.free(frame.data);
                try self.captureFrame(frame.data);
            } else {
                break;
            }
        }
    }

    const FrameResult = struct {
        data: []const u8,
        is_keyframe: bool,
    };

    fn extractFrame(self: *Self) !?FrameResult {
        if (self.buffer.items.len < 5) return null;

        var start_index: ?usize = null;
        var end_index: ?usize = null;
        var is_keyframe = false;

        // Find first NALU start code
        for (0..self.buffer.items.len - 3) |i| { // Corrected to -3 to prevent out-of-bounds
            if (mem.eql(u8, self.buffer.items[i .. i + 4], &self.nalu_start_codes)) {
                start_index = i;
                // Check NALU type (first byte after start code)
                const nalu_type = self.buffer.items[i + 4] & 0x1F;
                is_keyframe = (nalu_type == 5); // Typically, 5 indicates an IDR frame (keyframe)
                break;
            }
        }

        if (start_index == null) {
            // No start code found, clear buffer
            self.buffer.clearRetainingCapacity();
            return null;
        }

        // std.debug.print("Start Index: {?}\n", .{start_index});
        // std.debug.print("End Index: {?}\n", .{end_index});

        // Find next start code
        // std.debug.print("Buffer Length: {d} => {d}\n", .{ self.buffer.items.len, self.buffer.items.len - 3 });
        if (start_index.? + 4 < self.buffer.items.len - 3) {
            for (start_index.? + 4..self.buffer.items.len - 3) |i| { // Corrected to -3
                if (mem.eql(u8, self.buffer.items[i .. i + 4], &self.nalu_start_codes)) {
                    end_index = i;
                    break;
                }
            }
        }

        if (end_index == null) {
            // Implement maximum buffer size check
            const MAX_BUFFER_SIZE: usize = 10 * 1024 * 1024; // 10 MB
            if (self.buffer.items.len > MAX_BUFFER_SIZE) {
                std.debug.print("Buffer exceeded maximum size. Clearing buffer.\n", .{});
                self.buffer.clearRetainingCapacity();
            }
            return null;
        }

        // Ensure start_index < end_index
        if (start_index.? > end_index.?) {
            std.debug.print("Error: start_index ({}) > end_index ({})\n", .{ start_index.?, end_index.? });
            self.buffer.clearRetainingCapacity();
            return null;
        }

        const frame_size = end_index.? - start_index.?;
        const MAX_FRAME_SIZE: usize = 5 * 1024 * 1024; // 10 MB
        if (frame_size > MAX_FRAME_SIZE) {
            std.debug.print("Frame size ({}) exceeds maximum limit. Skipping frame.\n", .{frame_size});
            self.buffer.clearRetainingCapacity();
            return null;
        }

        // Extract frame data
        const frame_data = try self.allocator.dupe(u8, self.buffer.items[start_index.?..end_index.?]);

        // Remove processed data from buffer
        const remaining_start = end_index.?;
        const remaining_len = self.buffer.items.len - remaining_start;

        std.debug.print("Remaining Length: {any}\n", .{remaining_len});

        if (remaining_len > 0) {
            // Use mem.copyForward for overlapping regions
            std.mem.copyForwards(u8, self.buffer.items[0..remaining_len], self.buffer.items[remaining_start..]);
        }
        self.buffer.resize(remaining_len) catch return error.BufferResizeFailed;

        return FrameResult{
            .data = frame_data,
            .is_keyframe = is_keyframe,
        };
    }

    pub fn pipeline(self: *Self, input: []const u8) !void {
        // Reset packet
        c.av_packet_unref(self.packet);

        const start = try std.time.Instant.now();

        // Parse input data
        var parsed_data: *u8 = undefined;
        var parsed_size: c_int = 0;
        const consumed = c.av_parser_parse2(
            self.parser_context,
            self.codec_context,
            @ptrCast(&parsed_data),
            &parsed_size,
            input.ptr,
            @intCast(input.len),
            0,
            0,
            0,
        );

        if (consumed < 0) return error.ParsingFailed;
        if (parsed_size == 0) return;

        // Prepare packet
        self.packet.data = parsed_data;
        self.packet.size = parsed_size;

        // Send packet to decoder
        const send_result = c.avcodec_send_packet(self.codec_context, self.packet);
        if (send_result < 0) {
            std.debug.print("Failed to send packet to decoder\n", .{});
            return error.DecodeFailed;
        }

        // Receive decoded frame
        const receive_result = c.avcodec_receive_frame(self.codec_context, self.hw_frame);
        if (receive_result < 0) {
            if (receive_result == AVERROR_EAGAIN or receive_result == AVERROR_EOF) {
                return;
            }
            std.debug.print("Failed to receive frame\n", .{});
            return error.FrameReceiveFailed;
        }

        const transferred_frame = c.av_frame_alloc();
        if (transferred_frame == null) {
            std.debug.print("Failed to allocate transferred frame\n", .{});
            return error.FailedAllocation;
        }
        defer c.av_frame_free(@ptrCast(@constCast(&transferred_frame)));

        if (c.av_hwframe_transfer_data(transferred_frame, self.hw_frame, 0) < 0) {
            c.av_frame_free(@ptrCast(@constCast(transferred_frame)));
            std.debug.print("Failed to transfer data from HW frame\n", .{});
            return error.FailedHwTransfer;
        }

        // Invoke the callback with the RGB frame
        const end = try std.time.Instant.now();
        _ = start;
        _ = end;

        // std.debug.print("Decoded Frame Pixel Format: {d} ({s})\n", .{ filt_frame.*.format, pix_fmt_to_str(filt_frame.*.format) });
        // std.debug.print("Decoded Frame Dimensions: {}x{}\n", .{ filt_frame.*.width, filt_frame.*.height });
        // std.debug.print("Decoding Time: {d}\n", .{@as(f64, @floatFromInt(std.time.Instant.since(end, start))) / 1e9});

        if (self.frame_delta == -1) {
            self.frame_delta = std.time.nanoTimestamp();
        } else {
            const now = std.time.nanoTimestamp();
            const delta = now - self.frame_delta;

            std.debug.print("Delta: {d}\n", .{@as(f128, @floatFromInt(delta)) / 1e9});
            self.frame_delta = now;
        }

        try self.onDecodedFrame(self.allocator, self.node, transferred_frame);
    }

    pub fn captureFrame(self: *Self, frame: []const u8) !void {
        // Determine if this is a keyframe and convert to PNG
        try self.pipeline(frame);
    }
};

pub fn detectBestHardwareDevice() c_uint {
    std.debug.print("Detecting hardware devices...\n", .{});
    var device_type = c.av_hwdevice_iterate_types(c.AV_HWDEVICE_TYPE_NONE);

    while (device_type != c.AV_HWDEVICE_TYPE_NONE) {
        const device_name = c.av_hwdevice_get_type_name(device_type);
        std.debug.print("Found device type: {s}\n", .{device_name});

        switch (device_type) {
            c.AV_HWDEVICE_TYPE_CUDA => {
                std.debug.print("Selecting CUDA for hardware decoding\n", .{});
                return c.AV_HWDEVICE_TYPE_CUDA;
            },
            c.AV_HWDEVICE_TYPE_VAAPI => {
                std.debug.print("Selecting VAAPI for hardware decoding\n", .{});
                return c.AV_HWDEVICE_TYPE_VAAPI;
            },
            c.AV_HWDEVICE_TYPE_D3D11VA => {
                std.debug.print("Selecting D3D11VA for hardware decoding\n", .{});
                return c.AV_HWDEVICE_TYPE_D3D11VA;
            },
            c.AV_HWDEVICE_TYPE_VIDEOTOOLBOX => {
                std.debug.print("Selecting VideoToolbox for hardware decoding\n", .{});
                return c.AV_HWDEVICE_TYPE_VIDEOTOOLBOX;
            },
            else => device_type = c.av_hwdevice_iterate_types(device_type),
        }
    }

    std.debug.print("No suitable hardware decoding device found\n", .{});
    return c.AV_HWDEVICE_TYPE_NONE;
}

fn pix_fmt_to_str(pix_fmt: c.AVPixelFormat) []const u8 {
    const name_ptr = c.av_get_pix_fmt_name(pix_fmt);

    if (name_ptr == null) {
        return "yuv420p"; // Fallback to a default format
    }

    return std.mem.span(name_ptr);
}

pub fn frameCallback(allocator: std.mem.Allocator, node: *Node, frame: *c.AVFrame) error{OutOfMemory}!void {
    // std.debug.print("Pixel Format: {s}\n", .{c.av_get_pix_fmt_name(frame.format)});
    // std.debug.print("Width: {d}\n", .{frame.width});
    // std.debug.print("Height: {d}\n", .{frame.height});
    // std.debug.print("Linesize: {d}\n", .{frame.linesize});

    // // Debug data pointers and line sizes
    // std.debug.print("Data[0] ptr: {*}\n", .{frame.data[0]});
    // std.debug.print("Data[1] ptr: {*}\n", .{frame.data[1]});

    // std.debug.print("Linesize[0]: {}\n", .{frame.linesize[0]});
    // std.debug.print("Linesize[1]: {}\n", .{frame.linesize[1]});

    const frame_width = @as(usize, @intCast(frame.width));
    const frame_height = @as(usize, @intCast(frame.height));

    const y_plane = frame.data[0][0 .. @as(usize, @intCast(frame.linesize[0])) * frame_height];
    const uv_plane = frame.data[1][0 .. @as(usize, @intCast(frame.linesize[1])) * (frame_height / 2)];

    // Copy rows, skipping the padding
    for (0..frame_height) |i| {
        const src_row = y_plane[i * @as(usize, @intCast(frame.linesize[0])) ..][0..frame_width];
        @memcpy(node.y[i * frame_width ..][0..frame_width], src_row);
    }

    // // Similar for UV plane
    for (0..(frame_height / 2)) |i| {
        const src_row = uv_plane[i * @as(usize, @intCast(frame.linesize[1])) ..][0..frame_width];
        @memcpy(node.uv[i * frame_width ..][0..frame_width], src_row);
    }
    _ = allocator;

    node.width = @intCast(frame_width);
    node.height = @intCast(frame_height);
    node.texture_updated = true;
}
