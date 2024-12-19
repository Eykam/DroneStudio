const Self = @This();

host_ip: []const u8 = "192.168.1.0",
host_port_imu: u16 = 8000,

// =====================
// Not used for now. Will be relevant when IMU is integrated back into the control flow
client_ip: []const u8 = "192.168.1.1",
client_port_imu: u16 = 8000,
// =====================

// Sample SDP file. Will be different based on your network setup
sdp_content_left: *const [307:0]u8 =
    "v=0\r\n" ++
        "o=- 0 0 IN IP4 192.168.1.1\r\n" ++
        "s=No Name\r\n" ++
        "c=IN IP4 192.168.1.1\r\n" ++
        "t=0 0\r\n" ++
        "a=tool:libavformat LIBAVFORMAT_VERSION\r\n" ++
        "m=video 8888 RTP/AVP 96\r\n" ++
        "a=rtpmap:96 H264/90000\r\n" ++
        "a=fmtp:96 packetization-mode=1\r\n",
sdp_content_right: *const [305:0]u8 =
    "v=0\r\n" ++
        "o=- 0 0 IN IP4 192.168.1.2\r\n" ++
        "s=No Name\r\n" ++
        "c=IN IP4 192.168.1.2\r\n" ++
        "t=0 0\r\n" ++
        "a=tool:libavformat LIBAVFORMAT_VERSION\r\n" ++
        "m=video 6666 RTP/AVP 96\r\n" ++
        "a=rtpmap:96 H264/90000\r\n" ++
        "a=fmtp:96 packetization-mode=1\r\n",
