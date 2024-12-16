const std = @import("std");
const libav = @import("libav.zig");
const video = libav.video;
const Node = @import("Node.zig");

const mem = std.mem;
const fs = std.fs;

const AVERROR_EAGAIN = -1 * @as(c_int, @intCast(video.EAGAIN));
const AVERROR_EOF = -1 * @as(c_int, @intCast(video.EOF));

const DecodedFrameCallback = *const fn (mem.Allocator, *Node, *video.struct_AVFrame) error{OutOfMemory}!void;
pub const StreamPollingConfig = struct {
    max_retry_attempts: u32 = 100,
    retry_delay_ms: u64 = 2000, // 2 seconds between retries
    timeout_ms: u64 = 10000, // 1 minute total timeout
};

const InitializationError = error{
    RTPInitializationFailed,
    MaxRetriesExceeded,
    StreamInfoRetrievalTimeout,
    NoVideoStream,
    CodecNotFound,
    CodecContextAllocationFailed,
    CodecParametersCopyFailed,
    HardwareDeviceContextCreationFailed,
    HardwareFramesContextInitFailed,
    CodecOpenFailed,
    FrameAllocationFailed,
    FilterGraphAllocationFailed,
    NoHardwareDeviceContext,
    HardwareFramesContextAllocationFailed,
    FilterGraphConfigurationFailed,
    InvalidFrameDimensions,
};

pub const VideoHandler = struct {
    const Self = @This();

    allocator: mem.Allocator,
    node: *Node,

    format_context: *video.AVFormatContext,
    stream_index: c_int = -1,

    // FFmpeg Decoder Fields
    hw_device_ctx: ?*video.AVBufferRef = null,
    codec_context: *video.AVCodecContext,
    hw_frame: *video.AVFrame,

    filter_graph: ?*video.AVFilterGraph = null,
    buffersrc: ?*video.AVFilterContext = null,
    buffersink: ?*video.AVFilterContext = null,

    // Callback for decoded RGB frames
    onDecodedFrame: DecodedFrameCallback,

    polling_config: StreamPollingConfig,
    is_stream_ready: bool = false,

    const sdp_content =
        "v=0\r\n" ++
        "o=- 0 0 IN IP4 192.168.137.145\r\n" ++
        "s=No Name\r\n" ++
        "c=IN IP4 192.168.137.145\r\n" ++
        "t=0 0\r\n" ++
        "a=tool:libavformat LIBAVFORMAT_VERSION\r\n" ++
        "m=video 8888 RTP/AVP 96\r\n" ++
        "a=rtpmap:96 H264/90000\r\n" ++
        "a=fmtp:96 packetization-mode=1; sprop-parameter-sets=J2QAKqwrQCgC3QgAAAMACAAABLc2AAExLAADua973APEiag=,KO4CXLAA; profile-level-id=64002A\r\n";

    pub fn init(
        allocator: mem.Allocator,
        node: *Node,
        rtp_url: []const u8,
        hw_type: ?c_uint,
        onDecodedFrame: DecodedFrameCallback,
        polling_config: ?StreamPollingConfig,
    ) !Self {
        _ = rtp_url;

        const config = polling_config orelse StreamPollingConfig{};

        video.av_log_set_level(video.AV_LOG_DEBUG);

        // Initialize network
        if (video.avformat_network_init() < 0) {
            std.debug.print("Failed to initialize RTP Server", .{});
            return InitializationError.RTPInitializationFailed;
        }
        errdefer _ = video.avformat_network_deinit();

        const start_time = std.time.milliTimestamp();
        var attempts: u32 = 0;
        var format_context_ptr: *video.AVFormatContext = undefined;

        // Create a unique temporary directory
        const tmp_dir_name = try mkdtemp(allocator, "rtp_stream");
        std.debug.print("Dir successfully created!: {s}\n", .{tmp_dir_name});
        // Construct the SDP file path
        const sdp_file_path = try std.fs.path.join(allocator, &.{ tmp_dir_name, "rtp_stream.sdp" });
        std.debug.print("Temp File path: {s}\n", .{sdp_file_path});
        const tmp_file = try fs.cwd().createFile(sdp_file_path, .{});
        defer {
            tmp_file.close();
            allocator.free(sdp_file_path);
            allocator.free(tmp_dir_name);
        }
        errdefer {
            if (format_context_ptr != undefined) {
                video.avformat_close_input(@ptrCast(@constCast(&format_context_ptr)));
                video.avformat_free_context(@ptrCast(format_context_ptr));
            }
        }

        try tmp_file.writeAll(sdp_content);

        while (attempts < config.max_retry_attempts) : (attempts += 1) {
            std.debug.print("Attempt {d} to establish connection to Video Stream...\n", .{attempts + 1});
            format_context_ptr = initRTPStreamWithSDP(sdp_file_path) catch {
                // Wait before retrying
                std.time.sleep(config.retry_delay_ms * std.time.ns_per_ms);
                continue;
            };

            std.debug.print("Successfully established connection to Video Stream\n", .{});
            break;
        }

        // If we exhausted retry attempts
        if (attempts >= config.max_retry_attempts) {
            std.debug.print("Failed to open input stream after {} attempts\n", .{attempts});
            return InitializationError.MaxRetriesExceeded;
        }

        // Retrieve stream information with polling
        attempts = 0;
        while (attempts < config.max_retry_attempts) : (attempts += 1) {
            std.debug.print("Attempt {d} to extract stream info...\n", .{attempts + 1});
            const stream_info_result = video.avformat_find_stream_info(format_context_ptr, null);

            if (stream_info_result >= 0) {
                std.debug.print("Successfully extracted stream info\n", .{});
                break;
            }

            // Check if total timeout exceeded
            if (std.time.milliTimestamp() - start_time > config.timeout_ms) {
                std.debug.print("Stream info retrieval timed out after {} ms\n", .{config.timeout_ms});
                return InitializationError.StreamInfoRetrievalTimeout;
            }

            std.time.sleep(config.retry_delay_ms * std.time.ns_per_ms);
        }

        // Find video stream
        var stream_index: c_int = -1;
        std.debug.print("Number of streams found: {d}\n", .{format_context_ptr.*.nb_streams});
        for (0..format_context_ptr.*.nb_streams) |i| {
            const curr_codec = format_context_ptr.*.streams[i].*.codecpar.*.codec_type;
            if (curr_codec == video.AVMEDIA_TYPE_VIDEO) {
                std.debug.print("Current Codec: {d}\n", .{curr_codec});
                stream_index = @intCast(i);
                break;
            }
        }

        if (stream_index == -1) {
            std.debug.print("No video stream found\n", .{});
            return InitializationError.NoVideoStream;
        }

        // Find and open codec
        const stream = format_context_ptr.*.streams[@intCast(stream_index)];
        const codec = video.avcodec_find_decoder(stream.*.codecpar.*.codec_id);
        if (codec == null) {
            std.debug.print("Codec not found\n", .{});
            return InitializationError.CodecNotFound;
        }

        // Allocate codec context
        const codec_context = video.avcodec_alloc_context3(codec) orelse return InitializationError.CodecContextAllocationFailed;
        errdefer video.avcodec_free_context(@ptrCast(@constCast(&codec_context)));

        // Copy codec parameters
        if (video.avcodec_parameters_to_context(codec_context, stream.*.codecpar) < 0) {
            std.debug.print("Could not copy codec parameters\n", .{});
            return InitializationError.CodecParametersCopyFailed;
        }

        std.debug.print("Codec width: {d}, height: {d}\n", .{ codec_context.*.width, codec_context.*.height });
        if (codec_context.*.width == 0 or codec_context.*.height == 0) {
            return InitializationError.InvalidFrameDimensions;
        }

        const selected_hw_type = hw_type orelse detectBestHardwareDevice();
        var hw_device_ctx: ?*video.AVBufferRef = null;
        errdefer {
            if (hw_device_ctx) |ctx| {
                video.av_buffer_unref(@ptrCast(@constCast(&ctx)));
            }
        }

        if (selected_hw_type != video.AV_HWDEVICE_TYPE_NONE) {
            if (video.av_hwdevice_ctx_create(&hw_device_ctx, selected_hw_type, null, null, 0) < 0) {
                std.debug.print("Could not create hardware device context\n", .{});
                return InitializationError.HardwareDeviceContextCreationFailed;
            }
            codec_context.*.hw_device_ctx = hw_device_ctx;
        }

        // Open codec
        if (video.avcodec_open2(codec_context, codec, null) < 0) {
            std.debug.print("Could not open codec\n", .{});
            return InitializationError.CodecOpenFailed;
        }

        const hw_frame = video.av_frame_alloc() orelse return InitializationError.FrameAllocationFailed;
        errdefer video.av_frame_free(@ptrCast(@constCast(&hw_frame)));

        // Create VideoHandler instance
        var self = Self{
            .allocator = allocator,
            .node = node,
            .format_context = format_context_ptr,
            .stream_index = stream_index,
            .hw_device_ctx = hw_device_ctx,
            .codec_context = codec_context,
            .hw_frame = hw_frame,
            .onDecodedFrame = onDecodedFrame,
            .polling_config = config,
            .is_stream_ready = true,
        };

        try self.initHardwareFilterGraph();
        return self;
    }

    pub fn initHardwareFilterGraph(self: *Self) !void {
        const filter_graph = video.avfilter_graph_alloc();
        if (filter_graph == null) {
            std.debug.print("Failed to allocate filter graph.\n", .{});
            return InitializationError.FilterGraphAllocationFailed;
        }
        errdefer video.avfilter_graph_free(@ptrCast(@constCast(&filter_graph)));

        // Create a new hardware frames context if not already present
        if (self.codec_context.hw_device_ctx == null) {
            std.debug.print("No hardware device context available.\n", .{});
            return InitializationError.NoHardwareDeviceContext;
        }

        // Create a new hardware frames context if not already present
        const hw_frames_ctx = video.av_hwframe_ctx_alloc(self.codec_context.hw_device_ctx) orelse {
            std.debug.print("Failed to allocate hardware frames context.\n", .{});
            return InitializationError.HardwareFramesContextAllocationFailed;
        };

        // Configure the hardware frames context
        const frames_ctx = @as(*video.AVHWFramesContext, @ptrCast(@alignCast(hw_frames_ctx.*.data)));
        frames_ctx.format = video.AV_PIX_FMT_CUDA; // Adjust based on your hardware type
        frames_ctx.sw_format = video.AV_PIX_FMT_NV12;
        frames_ctx.width = self.codec_context.width;
        frames_ctx.height = self.codec_context.height;
        frames_ctx.initial_pool_size = 4; // Preallocate frames

        if (video.av_hwframe_ctx_init(hw_frames_ctx) < 0) {
            std.debug.print("Failed to initialize hardware frames context.\n", .{});
            video.av_buffer_unref(@ptrCast(@constCast(&hw_frames_ctx)));
            return InitializationError.HardwareFramesContextInitFailed;
        }

        if (hw_frames_ctx == null) {
            std.debug.print("Hardware frames context is null after initialization\n", .{});
            return InitializationError.HardwareFramesContextAllocationFailed;
        }

        // Create hwupload buffer source with explicit frames context
        const buffersrc = video.avfilter_graph_alloc_filter(filter_graph, video.avfilter_get_by_name("buffer"), "in");
        if (buffersrc == null) {
            std.debug.print("Failed to create buffer source filter.\n", .{});
            return InitializationError.FilterGraphAllocationFailed;
        }
        errdefer {
            std.debug.print("Error detected, freeing buffersrc", .{});
            video.avfilter_free(@ptrCast(@constCast(buffersrc)));
        }

        const params = video.av_buffersrc_parameters_alloc();
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

        if (video.av_buffersrc_parameters_set(buffersrc, params) < 0) {
            std.debug.print("Failed to set buffersrc params.\n", .{});
            return InitializationError.FilterGraphAllocationFailed;
        }
        if (video.avfilter_init_str(buffersrc, null) < 0) {
            std.debug.print("Failed to initialize buffer source filter.\n", .{});
            return InitializationError.FilterGraphAllocationFailed;
        }

        // Create hardware buffer sink
        const buffersink = video.avfilter_graph_alloc_filter(
            filter_graph,
            video.avfilter_get_by_name("buffersink"),
            "out",
        );
        if (buffersink == null) {
            std.debug.print("Failed to create buffer sink filter.\n", .{});
            return InitializationError.FilterGraphAllocationFailed;
        }
        errdefer video.avfilter_free(buffersink);

        if (video.avfilter_init_str(buffersink, null) < 0) {
            std.debug.print("Failed to initialize buffer sink filter.\n", .{});
            return InitializationError.FilterGraphAllocationFailed;
        }

        if (video.avfilter_link(buffersrc, 0, buffersink, 0) < 0) {
            std.debug.print("Failed to link buffer to hwupload filter.\n", .{});
            return InitializationError.FilterGraphAllocationFailed;
        }

        // Configure the filter graph
        if (video.avfilter_graph_config(filter_graph, null) < 0) {
            std.debug.print("Failed to configure filter graph.\n", .{});
            return InitializationError.FilterGraphConfigurationFailed;
        }

        // Store filter contexts
        self.filter_graph = filter_graph;
        self.buffersrc = buffersrc;
        self.buffersink = buffersink;

        if (self.buffersrc == null or self.buffersink == null) {
            std.debug.print("Failed to retrieve filter contexts.\n", .{});
            return InitializationError.FilterGraphAllocationFailed;
        }
    }

    pub fn deinit(self: *Self) void {
        std.debug.print("Destroying VideoHandler...\n", .{});

        // Cleanup filter graph and related contexts first
        std.debug.print("Cleaning up filter graph...\n", .{});
        if (self.filter_graph) |graph| {
            video.avfilter_graph_free(@ptrCast(@constCast(&graph)));
            self.filter_graph = null;
        }
        self.buffersrc = null;
        self.buffersink = null;

        // Close and free format context
        std.debug.print("Closing and freeing avformat...\n", .{});
        video.avformat_close_input(@ptrCast(@constCast(&self.format_context)));
        video.avformat_free_context(@ptrCast(self.format_context));

        // Free hardware device context
        std.debug.print("Freeing hw_device_ctx...\n", .{});
        if (self.hw_device_ctx) |ctx| {
            video.av_buffer_unref(@ptrCast(@constCast(&ctx)));
        }

        std.debug.print("Flushing buffers on codec context...\n", .{});
        video.avcodec_flush_buffers(self.codec_context);

        // Free hardware frame
        std.debug.print("Freeing hw_frame...\n", .{});
        video.av_frame_free(@ptrCast(@constCast(&self.hw_frame)));

        // Deinitialize network only once
        std.debug.print("Deinitializing avformat_network...\n", .{});
        _ = video.avformat_network_deinit();

        std.debug.print("Successfully Destroyed VideoHandler\n", .{});
    }

    pub fn start(
        allocator: mem.Allocator,
        node: *Node,
        rtp_url: []const u8,
        hw_type: ?c_uint,
        onDecodedFrame: DecodedFrameCallback,
        polling_config: ?StreamPollingConfig,
    ) !std.Thread {
        const spwn_config = std.Thread.SpawnConfig{
            .allocator = std.heap.page_allocator,
            .stack_size = 16 * 1024 * 1024, // Adjust the stack size as needed
        };

        return std.Thread.spawn(spwn_config, VideoHandler.consumer, .{
            allocator,
            node,
            rtp_url,
            hw_type,
            onDecodedFrame,
            polling_config,
        });
    }

    pub fn consumer(
        allocator: std.mem.Allocator,
        node: *Node,
        rtp_url: []const u8,
        hw_type: ?c_uint,
        onDecodedFrame: DecodedFrameCallback,
        polling_config: ?StreamPollingConfig,
    ) void {
        while (true) {
            var videoHandler: ?Self = null;
            while (videoHandler == null) {
                videoHandler = Self.init(
                    allocator,
                    node,
                    rtp_url,
                    hw_type,
                    onDecodedFrame,
                    polling_config,
                ) catch |err| {
                    std.debug.print("Error initializing Video Handler: {any}\n", .{err});
                    std.time.sleep(2 * std.time.ns_per_s); // Wait 2 seconds before retrying
                    continue;
                };
            }

            var handler = videoHandler.?;
            defer handler.deinit();

            if (!handler.is_stream_ready) {
                std.debug.print("Stream is not ready\n", .{});
                continue;
            }

            var retry_count: u32 = 0;
            const max_consecutive_errors = handler.polling_config.max_retry_attempts;

            while (retry_count < max_consecutive_errors) {
                const packet_result = handler.processNextPacket() catch |err| {
                    std.debug.print("Packet processing error: {any}\n", .{err});
                    retry_count += 1;
                    std.time.sleep(1 * std.time.ns_per_ms);
                    continue;
                };

                if (packet_result) {
                    // Successful packet processing resets retry count
                    retry_count = 0;
                } else {
                    retry_count += 1;

                    if (retry_count >= max_consecutive_errors) {
                        std.debug.print("Exceeded maximum consecutive packet processing errors\n", .{});
                        break;
                    }

                    std.time.sleep(1 * std.time.ns_per_ms);
                }
            }
        }
    }

    pub fn processNextPacket(self: *Self) !bool {
        const packet = video.av_packet_alloc() orelse return error.PacketAllocationFailed;
        defer video.av_packet_free(@ptrCast(@constCast(&packet)));

        // Read packet from stream
        const read_result = video.av_read_frame(self.format_context, packet);
        if (read_result < 0) {
            if (read_result == AVERROR_EOF) {
                std.debug.print("End of stream\n", .{});
                return false;
            }
            std.debug.print("Error reading frame\n", .{});
            return false;
        }

        // Ensure packet is from video stream
        if (packet.*.stream_index != self.stream_index) {
            return true;
        }

        // Send packet to decoder
        const send_result = video.avcodec_send_packet(self.codec_context, packet);
        if (send_result < 0) {
            std.debug.print("Error sending packet to decoder\n", .{});
            return false;
        }

        // Receive and process frames
        while (true) {
            const receive_result = video.avcodec_receive_frame(self.codec_context, self.hw_frame);
            if (receive_result < 0) {
                if (receive_result == AVERROR_EAGAIN or receive_result == AVERROR_EOF) {
                    break;
                }
                std.debug.print("Error receiving frame\n", .{});
                return false;
            }

            // Transfer frame if using hardware decoding
            const transferred_frame = video.av_frame_alloc() orelse {
                std.debug.print("Failed to allocate transferred frame\n", .{});
                return false;
            };
            defer video.av_frame_free(@ptrCast(@constCast(&transferred_frame)));

            if (video.av_hwframe_transfer_data(transferred_frame, self.hw_frame, 0) < 0) {
                std.debug.print("Failed to transfer frame data\n", .{});
                continue;
            }

            // Invoke callback
            try self.onDecodedFrame(self.allocator, self.node, transferred_frame);
        }

        return true;
    }
};

pub fn detectBestHardwareDevice() c_uint {
    std.debug.print("Detecting hardware devices...\n", .{});
    var device_type = video.av_hwdevice_iterate_types(video.AV_HWDEVICE_TYPE_NONE);

    while (device_type != video.AV_HWDEVICE_TYPE_NONE) {
        const device_name = video.av_hwdevice_get_type_name(device_type);
        std.debug.print("Found device type: {s}\n", .{device_name});

        switch (device_type) {
            video.AV_HWDEVICE_TYPE_CUDA => {
                std.debug.print("Selecting CUDA for hardware decoding\n", .{});
                return video.AV_HWDEVICE_TYPE_CUDA;
            },
            video.AV_HWDEVICE_TYPE_VAAPI => {
                std.debug.print("Selecting VAAPI for hardware decoding\n", .{});
                return video.AV_HWDEVICE_TYPE_VAAPI;
            },
            video.AV_HWDEVICE_TYPE_D3D11VA => {
                std.debug.print("Selecting D3D11VA for hardware decoding\n", .{});
                return video.AV_HWDEVICE_TYPE_D3D11VA;
            },
            video.AV_HWDEVICE_TYPE_VIDEOTOOLBOX => {
                std.debug.print("Selecting VideoToolbox for hardware decoding\n", .{});
                return video.AV_HWDEVICE_TYPE_VIDEOTOOLBOX;
            },
            else => device_type = video.av_hwdevice_iterate_types(device_type),
        }
    }

    std.debug.print("No suitable hardware decoding device found\n", .{});
    return video.AV_HWDEVICE_TYPE_NONE;
}

pub fn mkdtemp(allocator: std.mem.Allocator, path: []const u8) ![]u8 {

    // Attempt to create the directory
    const exe_dir = try fs.selfExeDirPathAlloc(allocator);
    defer allocator.free(exe_dir);

    const dir_name = try fs.path.join(allocator, &.{ exe_dir, path });

    _ = fs.cwd().makePath(dir_name) catch |err| {
        if (err == fs.Dir.MakeError.PathAlreadyExists) {
            std.debug.print("Temp folder already exists! => {s}", .{dir_name});
            return dir_name;
        }

        return err;
    };

    return dir_name;
}

pub fn initRTPStreamWithSDP(sdp_path: []const u8) !*video.AVFormatContext {
    // Initialize network

    // Allocate format context
    const format_context = video.avformat_alloc_context() orelse
        return error.FormatContextAllocationFailed;

    // Prepare options dictionary
    var options: ?*video.AVDictionary = null;
    defer video.av_dict_free(&options);

    _ = video.av_dict_set(&options, "protocol_whitelist", "file,crypto,data,rtp,udp,tcp,http,https", 0);
    const infmt = video.av_find_input_format("sdp");

    // Attempt to open input
    var format_context_ptr: *video.AVFormatContext = format_context;
    const open_result = video.avformat_open_input(
        @ptrCast(&format_context_ptr),
        sdp_path.ptr,
        infmt,
        @ptrCast(&options),
    );

    if (open_result < 0) {
        var error_buf: [256]u8 = undefined;
        _ = video.av_strerror(open_result, &error_buf, error_buf.len);
        std.debug.print("Failed to open RTP stream. Error: {s}\n", .{error_buf});
        video.avformat_close_input(@ptrCast(&format_context_ptr));
        video.avformat_free_context(@ptrCast(format_context_ptr));
        return error.InputOpenFailed;
    }

    return format_context_ptr;
}

pub fn frameCallback(allocator: std.mem.Allocator, node: *Node, frame: *video.AVFrame) error{OutOfMemory}!void {
    const frame_width = @as(usize, @intCast(frame.width));
    const frame_height = @as(usize, @intCast(frame.height));

    const y_plane = frame.data[0][0 .. @as(usize, @intCast(frame.linesize[0])) * frame_height];
    const uv_plane = frame.data[1][0 .. @as(usize, @intCast(frame.linesize[1])) * (frame_height / 2)];

    if (node.y == null) {
        node.y = try allocator.alloc(u8, frame_width * frame_height);
    }
    if (node.uv == null) {
        node.uv = try allocator.alloc(u8, frame_width * (frame_height / 2));
    }

    // Copy rows, skipping the padding
    for (0..frame_height) |i| {
        const src_row = y_plane[i * @as(usize, @intCast(frame.linesize[0])) ..][0..frame_width];
        @memcpy(node.y.?[i * frame_width ..][0..frame_width], src_row);
    }

    // // Similar for UV plane
    for (0..(frame_height / 2)) |i| {
        const src_row = uv_plane[i * @as(usize, @intCast(frame.linesize[1])) ..][0..frame_width];
        @memcpy(node.uv.?[i * frame_width ..][0..frame_width], src_row);
    }

    node.width = @intCast(frame_width);
    node.height = @intCast(frame_height);
    node.texture_updated = true;
}
