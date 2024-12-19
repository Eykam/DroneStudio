#!/bin/bash
# Script to run UDP based RTP streaming server

WIDTH=480
HEIGHT=360
FORMAT=NV12
BITRATE=60000000
HOST_IP="192.168.1.100"  # Replace with your target host IP
PORT=8888

# Run GStreamer pipeline
gst-launch-1.0 \
    libcamerasrc ! \
    video/x-raw,width=${WIDTH},height=${HEIGHT},format=${FORMAT} ! \
    v4l2convert ! \
    queue max-size-buffers=1 leaky=downstream ! \
    v4l2h264enc extra-controls="controls,repeat_sequence_header=1,video_bitrate=${BITRATE},h264_i_frame_period=1" ! \
    video/x-h264,level=4.2 ! \
    h264parse ! \
    rtph264pay config-interval=1 pt=96 ! \
    queue max-size-buffers=1 leaky=downstream ! \
    udpsink host=${HOST_IP} port=${PORT} sync=false async=false