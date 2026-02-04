// src/Image.zig
const std = @import("std");
const c = @cImport({
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavformat/avformat.h");
    @cInclude("libavutil/imgutils.h");
    @cInclude("libswscale/swscale.h");
});

const Allocator = std.mem.Allocator;
const gl = @import("bindings/gl.zig");
const glad = gl.glad;

pub const ImageError = error{
    InvalidFile,
    UnsupportedFormat,
    DecoderNotFound,
    StreamNotFound,
    DecodingError,
    OutOfMemory,
    ScalingError,
};

pub const PixelFormat = enum {
    RGB,
    RGBA,
};

pub const Image = struct {
    width: u32,
    height: u32,
    channels: u32,
    format: PixelFormat,
    data: []u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, width: u32, height: u32, format: PixelFormat) !*Image {
        const channels: u32 = switch (format) {
            .RGB => 3,
            .RGBA => 4,
        };

        const data_size = width * height * channels;
        const data = try allocator.alloc(u8, data_size);

        // Initialize with zeros
        @memset(data, 0);

        const image = try allocator.create(Image);
        image.* = .{
            .width = width,
            .height = height,
            .channels = channels,
            .format = format,
            .data = data,
            .allocator = allocator,
        };

        return image;
    }

    pub fn deinit(self: *Image) void {
        self.allocator.free(self.data);
        self.allocator.destroy(self);
    }

    pub fn getChannels(self: *const Image) u8 {
        return switch (self.format) {
            .RGB => 3,
            .RGBA => 4,
        };
    }

    //TODO: Use sws to flip the image instead of manually doing it like this
    pub fn flipVertical(self: *Image) void {
        const channels = self.getChannels();
        const row_size = self.width * channels;
        const row_buffer = self.allocator.alloc(u8, row_size) catch return;
        defer self.allocator.free(row_buffer);

        const half_height = self.height / 2;

        for (0..half_height) |y| {
            const top_row = y;
            const bottom_row = self.height - 1 - y;

            const top_idx = top_row * row_size;
            const bottom_idx = bottom_row * row_size;

            // Copy top row to buffer
            @memcpy(row_buffer, self.data[top_idx .. top_idx + row_size]);
            // Copy bottom row to top
            @memcpy(self.data[top_idx .. top_idx + row_size], self.data[bottom_idx .. bottom_idx + row_size]);
            // Copy buffer to bottom row
            @memcpy(self.data[bottom_idx .. bottom_idx + row_size], row_buffer);
        }
    }

    // Public functions to load images
    pub fn loadFromFile(allocator: Allocator, filepath: []const u8) !*Image {
        // Initialize FFmpeg once (noop if already initialized)

        const c_filepath = try allocator.dupeZ(u8, filepath);
        defer allocator.free(c_filepath);

        // Create format context
        var format_ctx: ?*c.AVFormatContext = null;
        if (c.avformat_open_input(&format_ctx, c_filepath.ptr, null, null) < 0) {
            std.debug.print("Failed to open file: {s}\n", .{c_filepath});
            return ImageError.InvalidFile;
        }
        defer c.avformat_close_input(&format_ctx);

        // Retrieve stream information
        if (c.avformat_find_stream_info(format_ctx, null) < 0) {
            std.debug.print("Failed to find stream info in file: {s}\n", .{c_filepath});
            return ImageError.InvalidFile;
        }

        // Find the first video stream
        var stream_index: c_int = -1;
        for (0..format_ctx.?.nb_streams) |i| {
            if (format_ctx.?.streams[i].*.codecpar.*.codec_type == c.AVMEDIA_TYPE_VIDEO) {
                stream_index = @intCast(i);
                break;
            }
        }

        if (stream_index == -1) {
            return ImageError.StreamNotFound;
        }

        // Get the codec parameters
        const codec_params = format_ctx.?.streams[@intCast(stream_index)].*.codecpar;

        // Find decoder
        const decoder = c.avcodec_find_decoder(codec_params.*.codec_id);
        if (decoder == null) {
            return ImageError.DecoderNotFound;
        }

        // Create codec context
        var codec_ctx = c.avcodec_alloc_context3(decoder);
        if (codec_ctx == null) {
            return ImageError.OutOfMemory;
        }
        defer c.avcodec_free_context(&codec_ctx);

        // Copy parameters from stream to codec context
        if (c.avcodec_parameters_to_context(codec_ctx, codec_params) < 0) {
            return ImageError.DecodingError;
        }

        // Open the codec
        if (c.avcodec_open2(codec_ctx, decoder, null) < 0) {
            return ImageError.DecodingError;
        }

        // Allocate frame and packet
        var frame = c.av_frame_alloc();
        if (frame == null) {
            return ImageError.OutOfMemory;
        }
        defer c.av_frame_free(&frame);

        var packet = c.av_packet_alloc();
        if (packet == null) {
            return ImageError.OutOfMemory;
        }
        defer c.av_packet_free(&packet);

        // Read frames until we get a complete frame
        var frame_finished = false;
        while (c.av_read_frame(format_ctx, packet) >= 0) {
            if (packet.*.stream_index == stream_index) {
                // Send packet to decoder
                const send_result = c.avcodec_send_packet(codec_ctx, packet);
                if (send_result < 0) {
                    c.av_packet_unref(packet);
                    continue;
                }

                // Receive frame from decoder
                const receive_result = c.avcodec_receive_frame(codec_ctx, frame);
                if (receive_result == 0) {
                    frame_finished = true;
                    break;
                }
            }
            c.av_packet_unref(packet);
        }

        if (!frame_finished) {
            return ImageError.DecodingError;
        }

        // Determine target format
        const target_format = c.AV_PIX_FMT_RGB24; // We want RGB for OpenGL
        const pixel_format: PixelFormat = .RGB;

        // Create swscale context for format conversion
        const sws_ctx = c.sws_getContext(
            frame.*.width,
            frame.*.height,
            frame.*.format,
            frame.*.width,
            frame.*.height,
            target_format,
            c.SWS_BILINEAR,
            null,
            null,
            null,
        );

        if (sws_ctx == null) {
            return ImageError.ScalingError;
        }
        defer c.sws_freeContext(sws_ctx);

        if (sws_ctx != null) {
            // Set the correct color range (full range)
            _ = c.sws_setColorspaceDetails(
                sws_ctx,
                c.sws_getCoefficients(c.SWS_CS_DEFAULT),
                1,
                c.sws_getCoefficients(c.SWS_CS_DEFAULT),
                1,
                0,
                1 << 16,
                1 << 16,
            );
        }

        // Allocate destination frame
        var rgb_frame = c.av_frame_alloc();
        if (rgb_frame == null) {
            return ImageError.OutOfMemory;
        }
        defer c.av_frame_free(&rgb_frame);

        rgb_frame.*.format = target_format;
        rgb_frame.*.width = frame.*.width;
        rgb_frame.*.height = frame.*.height;

        // Allocate buffer for destination frame
        if (c.av_frame_get_buffer(rgb_frame, 32) < 0) {
            return ImageError.OutOfMemory;
        }

        // Convert frame to RGB
        _ = c.sws_scale(
            sws_ctx,
            &frame.*.data[0],
            &frame.*.linesize[0],
            0,
            frame.*.height,
            &rgb_frame.*.data[0],
            &rgb_frame.*.linesize[0],
        );

        // Create our image object
        const width: u32 = @intCast(rgb_frame.*.width);
        const height: u32 = @intCast(rgb_frame.*.height);
        const channels: u32 = 3; // RGB

        var image = try Image.init(allocator, width, height, pixel_format);

        // Copy data from FFmpeg frame to our image
        const row_size: u32 = @intCast(rgb_frame.*.linesize[0]);
        for (0..height) |y| {
            const src_offset = y * row_size;
            const dst_offset = y * width * channels;

            // Copy just the actual pixel data, not the padding
            @memcpy(image.data[dst_offset .. dst_offset + width * channels], rgb_frame.*.data[0][src_offset .. src_offset + width * channels]);
        }

        return image;
    }

    // Load image from memory buffer
    pub fn loadFromMemory(allocator: Allocator, buffer: []const u8) !*Image {
        var tmp_dir = std.fs.cwd().makeOpenPath("tmp", .{}) catch |err| {
            std.debug.print("Failed to create tmp directory: {}\n", .{err});
            return ImageError.InvalidFile;
        };
        defer tmp_dir.close();

        const tmp_filename = "tmp_image";
        const tmp_path = try std.fmt.allocPrint(allocator, "tmp/{s}", .{tmp_filename});
        defer allocator.free(tmp_path);

        var tmp_file = try std.fs.cwd().createFile(tmp_path, .{});
        defer tmp_file.close();

        try tmp_file.writeAll(buffer);

        const image = try loadFromFile(allocator, tmp_path);

        std.fs.cwd().deleteFile(tmp_path) catch |err| {
            std.debug.print("Warning: Failed to delete temp file: {}\n", .{err});
        };

        return image;
    }

};
