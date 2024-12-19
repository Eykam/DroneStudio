#!/bin/bash

# Script to configure a fresh Raspberry Pi with libraries needed for streaming video

# Update and upgrade the system
sudo apt update && sudo apt upgrade -y

# Install required GStreamer packages
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libcamera
