const std = @import("std");
const c = @cImport({
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavformat/avformat.h");
    @cInclude("libswscale/swscale.h");
    @cInclude("libavutil/imgutils.h");
    @cInclude("libavutil/pixdesc.h");
});

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
    // Add other errors as needed
};

const AVERROR_EAGAIN = -1 * @as(c_int, @intCast(c.EAGAIN));
const AVERROR_EOF = -1 * @as(c_int, @intCast(c.EOF));

pub const VideoHandler = struct {
    const Self = @This();

    allocator: mem.Allocator,
    buffer: std.ArrayList(u8),
    current_timestamp: u128 = 0,
    nalu_start_codes: [4]u8 = [_]u8{ 0, 0, 0, 1 },
    image_dir: fs.Dir,
    stale_count: u8 = 0,
    last_valid_timestamp: u128 = 0,

    // FFmpeg Decoder Fields
    parser_context: *c.AVCodecParserContext,
    codec_context: *c.AVCodecContext,
    frame: *c.AVFrame,
    packet: *c.AVPacket,

    // FFmpeg PNG Encoder Fields
    png_codec: *c.AVCodec,
    png_codec_ctx: *c.AVCodecContext,
    png_frame: *c.AVFrame,
    png_packet: *c.AVPacket,
    sws_ctx: ?*c.SwsContext = null,

    pub fn init(allocator: mem.Allocator) !Self {
        // Create images directory if it doesn't exist
        var cwd = fs.cwd();
        cwd.makeDir("images") catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        const image_dir = try cwd.openDir("images", .{});

        // Initialize FFmpeg Decoder Components
        const codec = c.avcodec_find_decoder(c.AV_CODEC_ID_H264);
        if (codec == null) return error.CodecNotFound;

        const parser_context = c.av_parser_init(@intCast(codec.*.id)) orelse return error.ParserInitFailed;
        errdefer c.av_parser_close(parser_context); // Ensure parser_context is closed on error

        const codec_context = c.avcodec_alloc_context3(codec);
        if (codec_context == null) return error.CodecContextAllocationFailed;
        errdefer c.avcodec_free_context(@ptrCast(@constCast(&codec_context))); // Ensure codec_context is freed on error

        codec_context.*.width = 1280;
        codec_context.*.height = 720;

        if (c.avcodec_open2(codec_context, codec, null) < 0) {
            return error.CodecOpenFailed;
        }

        const frame = c.av_frame_alloc() orelse return error.FrameAllocationFailed;
        errdefer c.av_frame_free(@ptrCast(@constCast(&frame))); // Ensure frame is freed on error

        const packet = c.av_packet_alloc() orelse return error.PacketAllocationFailed;
        errdefer c.av_packet_free(@ptrCast(@constCast(&packet))); // Ensure packet is freed on error

        // Initialize PNG Encoder
        const png_codec = c.avcodec_find_encoder(c.AV_CODEC_ID_PNG);
        if (png_codec == null) {
            return error.PNGCodecNotFound;
        }

        const png_codec_ctx = c.avcodec_alloc_context3(png_codec);
        if (png_codec_ctx == null) return error.PNGCodecContextAllocationFailed;
        errdefer c.avcodec_free_context(@ptrCast(@constCast(&png_codec_ctx))); // Ensure png_codec_ctx is freed on error

        // Set required fields for PNG encoder
        png_codec_ctx.*.width = codec_context.*.width;
        png_codec_ctx.*.height = codec_context.*.height;
        png_codec_ctx.*.pix_fmt = c.AV_PIX_FMT_RGB24;
        png_codec_ctx.*.codec_type = c.AVMEDIA_TYPE_VIDEO; // Essential for encoders
        png_codec_ctx.*.time_base.num = 1; // Setting time_base to {1,1}
        png_codec_ctx.*.time_base.den = 1;

        if (c.avcodec_open2(png_codec_ctx, png_codec, null) < 0) {
            return error.PNGCodecOpenFailed;
        }

        // Allocate PNG Frame
        const png_frame = c.av_frame_alloc();
        if (png_frame == null) return error.PNGFrameAllocationFailed;
        errdefer c.av_frame_free(@ptrCast(@constCast(&png_frame))); // Ensure png_frame is freed on error

        png_frame.*.width = png_codec_ctx.*.width;
        png_frame.*.height = png_codec_ctx.*.height;
        png_frame.*.format = png_codec_ctx.*.pix_fmt;

        if (c.av_frame_get_buffer(png_frame, 32) < 0) {
            return error.PNGFrameBufferAllocationFailed;
        }

        // Allocate PNG Packet
        const png_packet = c.av_packet_alloc();
        if (png_packet == null) return error.PNGPacketAllocationFailed;
        errdefer c.av_packet_free(@ptrCast(@constCast(&png_packet))); // Ensure png_packet is freed on error

        // Create a new instance without defers
        return Self{
            .allocator = allocator,
            .buffer = std.ArrayList(u8).init(allocator),
            .image_dir = image_dir,
            .parser_context = parser_context,
            .codec_context = codec_context,
            .frame = frame,
            .packet = packet,
            .png_codec = @constCast(png_codec),
            .png_codec_ctx = png_codec_ctx,
            .png_frame = png_frame,
            .png_packet = png_packet,
        };
    }

    pub fn deinit(self: *Self) void {
        self.buffer.deinit();
        self.image_dir.close();

        // Free FFmpeg Decoder Resources
        c.av_parser_close(self.parser_context);
        c.avcodec_free_context(@ptrCast(@constCast(&self.codec_context)));
        c.av_frame_free(@ptrCast(@constCast(&self.frame)));
        c.av_packet_free(@ptrCast(@constCast(&self.packet)));

        // Free FFmpeg PNG Encoder Resources
        _ = c.avcodec_close(self.png_codec_ctx);
        c.avcodec_free_context(@ptrCast(@constCast(&self.png_codec_ctx)));
        c.av_frame_free(@ptrCast(@constCast(&self.png_frame)));
        c.av_packet_free(@ptrCast(@constCast(&self.png_packet)));
        c.sws_freeContext(self.sws_ctx);
    }

    pub fn update(self: *Self, data: []const u8) !void {
        if (data.len <= 16) return;

        const timestamp: u128 = @bitCast(std.mem.readInt(i128, data[0..16], .big));

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

        self.last_valid_timestamp = timestamp;
        self.current_timestamp = timestamp;
        self.stale_count = 0;

        const video_data = data[16..];
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

        // Find next start code
        for (start_index.? + 4..self.buffer.items.len - 3) |i| { // Corrected to -3
            if (mem.eql(u8, self.buffer.items[i .. i + 4], &self.nalu_start_codes)) {
                end_index = i;
                break;
            }
        }

        if (end_index == null) {
            // No end start code found yet; wait for more data
            // Optionally, set a limit to prevent buffer overflow
            return null;
        }

        // Ensure start_index < end_index
        if (start_index.? > end_index.?) {
            std.debug.print("Error: start_index ({}) > end_index ({})\n", .{ start_index.?, end_index.? });
            self.buffer.clearRetainingCapacity();
            return null;
        }

        // Extract frame data
        const frame_data = try self.allocator.dupe(u8, self.buffer.items[start_index.?..end_index.?]);

        // Remove processed data from buffer
        const remaining_start = end_index.?;
        const remaining_len = self.buffer.items.len - remaining_start;
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

    pub fn convertToPng(self: *Self, input: []const u8) !void {
        // Reset packet
        c.av_packet_unref(self.packet);

        // Parse input data
        var parsed_data: *u8 = undefined;
        var parsed_size: c_int = 0;
        const consumed = c.av_parser_parse2(self.parser_context, self.codec_context, @ptrCast(&parsed_data), &parsed_size, input.ptr, @intCast(input.len), 0, 0, 0);

        if (consumed < 0) return error.ParsingFailed;
        if (parsed_size == 0) return;

        // Prepare packet
        self.packet.data = parsed_data;
        self.packet.size = parsed_size;

        // Send packet to decoder
        const send_result = c.avcodec_send_packet(self.codec_context, self.packet);
        if (send_result < 0) return error.DecodeFailed;

        // Receive decoded frame
        const receive_result = c.avcodec_receive_frame(self.codec_context, self.frame);
        if (receive_result < 0) {
            if (receive_result == AVERROR_EAGAIN or receive_result == AVERROR_EOF) {
                return;
            }
            return error.FrameReceiveFailed;
        }

        if (self.frame.*.format == c.AV_PIX_FMT_NONE) {
            std.debug.print("Decoded frame has invalid pixel format (AV_PIX_FMT_NONE).\n", .{});
            return error.InvalidPixelFormat;
        }

        std.debug.print("Decoded Frame Pixel Format: {d} ({s})\n", .{ self.frame.*.format, pix_fmt_to_str(self.frame.*.format) });
        std.debug.print("Decoded Frame Dimensions: {}x{}\n", .{ self.frame.*.width, self.frame.*.height });

        // Initialize sws_context if not already initialized
        if (self.sws_ctx == null) {
            self.sws_ctx = c.sws_getContext(
                self.frame.*.width,
                self.frame.*.height,
                self.frame.*.format, // Use the actual decoder's pixel format
                1280,
                720,
                c.AV_PIX_FMT_RGB24,
                c.SWS_BILINEAR,
                null,
                null,
                null,
            );

            if (self.sws_ctx == null) {
                std.debug.print("Failed to initialize sws_context with actual pixel format.\n", .{});
                return error.SwsContextAllocationFailed;
            }
        }

        const sws_scale_result = c.sws_scale(self.sws_ctx, &self.frame.*.data[0], &self.frame.*.linesize[0], 0, self.frame.*.height, &self.png_frame.*.data[0], &self.png_frame.*.linesize[0]);

        if (sws_scale_result <= 0) {
            std.debug.print("sws_scale failed with result: {}\n", .{sws_scale_result});
            return error.SwsScaleFailed;
        }

        // **Send Converted Frame to PNG Encoder**
        const send_frame_result = c.avcodec_send_frame(self.png_codec_ctx, self.png_frame);
        if (send_frame_result < 0) {
            std.debug.print("Failed to send frame to PNG encoder: {}\n", .{send_frame_result});
            return error.PNGFrameSendFailed;
        }

        // **Receive Encoded PNG Packet**
        const receive_png_result = c.avcodec_receive_packet(self.png_codec_ctx, self.png_packet);
        if (receive_png_result < 0) {
            if (receive_png_result == AVERROR_EAGAIN or receive_png_result == AVERROR_EOF) {
                std.debug.print("No PNG packet available after sending frame.\n", .{});
                return;
            }
            std.debug.print("Failed to receive PNG packet: {}\n", .{receive_png_result});
            return error.PNGFrameReceiveFailed;
        }

        // Generate PNG filename
        var filename_buf: [100]u8 = undefined;
        const filename = try std.fmt.bufPrint(&filename_buf, "{d}.png", .{self.current_timestamp});

        // Write PNG Packet Data to File
        const file = self.image_dir.createFile(filename, .{}) catch |err| {
            std.debug.print("Error: {any} => Failed to create file: {s}\n", .{ err, filename });
            return;
        };
        defer file.close();

        try file.writeAll(self.png_packet.data[0..@intCast(self.png_packet.size)]);

        // Unreference the PNG Packet
        c.av_packet_unref(self.png_packet);

        std.debug.print("Frame converted to PNG: {s}\n", .{filename});
    }

    pub fn captureFrame(self: *Self, frame: []const u8) !void {
        // Determine if this is a keyframe and convert to PNG
        try self.convertToPng(frame);
    }
};

// Initialization helper
pub fn initVideoHandler(allocator: mem.Allocator) !VideoHandler {
    // Ensure FFmpeg is initialized (for older versions of FFmpeg)
    // Note: av_register_all() is deprecated in newer FFmpeg versions and can be omitted
    // c.av_register_all();
    return VideoHandler.init(allocator);
}

fn pix_fmt_to_str(pix_fmt: c.AVPixelFormat) []const u8 {
    return switch (pix_fmt) {
        c.AV_PIX_FMT_YUV420P => "YUV420P",
        c.AV_PIX_FMT_RGB24 => "RGB24",
        c.AV_PIX_FMT_YUV422P => "YUV422P",
        c.AV_PIX_FMT_YUV444P => "YUV444P",
        c.AV_PIX_FMT_NV12 => "NV12",
        c.AV_PIX_FMT_NV21 => "NV21",
        c.AV_PIX_FMT_GRAY8 => "GRAY8",
        // Add other pixel formats as needed
        else => "Unknown",
    };
}
