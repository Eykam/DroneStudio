# Drone Studio

Autonomous drone firmware, renderer and simulator for mapping environments.

## Overall Goal

Goal at the moment is to have the drone autonomously map a building in 3D using Gaussian Splatting. The renderer will then allow the user to walk around, view the scene like a video game. 

![alt text](https://github.com/Eykam/DroneStudio/blob/main/assets/hero.png)
(Not my image, but similar to what I'd like to produce)

Should be able to use pose from MPU (mpu-9250) along with distance data from a stereo camera to construct pose-graph of environment. With current sensors, would be able to obtain and transmit 1 khz of pose data, and 480p@100fps (still testing this), over UDP.

Later goals are to add object avoidance, image detection and synchronization with other drones within the same network / swarm.

## Setup

### Raspberry Pi's

To setup the raspberry pi's:
- Flash with [Raspberry Pi OS Lite (64 bit)](https://downloads.raspberrypi.com/raspios_lite_arm64/images/) (make sure to configure wifi and ssh)
- Replace config at `/boot/firmware/config.txt` with [this](https://github.com/Eykam/DroneStudio/blob/main/Firmware/piZero2W/config.txt)
- Then run `sudo reboot`
- You can now run the setup script [here](https://github.com/Eykam/DroneStudio/blob/main/Firmware/piZero2W/setup.sh)
- After the setup is done, you can start the streamer using this [script](https://github.com/Eykam/DroneStudio/blob/main/Firmware/piZero2W/run.sh)

You can ignore all the other code within [this library](https://github.com/Eykam/DroneStudio/blob/main/Firmware/piZero2W) for the moment. I'll likely build a binary to automate this process, along with other features I'm working on for the future.

### Video processing & Rendering Server

#### Configuring Secrets

Assuming you have the hardware setup (2 raspberry pi's with a camera module 3 on each), copy the [Secrets](https://github.com/Eykam/DroneStudio/blob/main/Studio/src/Secrets.example.zig) to Secrets.local.zig and modify the IP's to match the raspberry pi's. You may need to modify the sdp content to match your stream params.

#### Building from Source

Dependencies:
- libc
- Cuda
- openGL (GLAD and GLFW)
- FFmpeg

Currently working on creating statically compiled binaries to avoid needing to build from source. 
For now take a look at the [Build Script](https://github.com/Eykam/DroneStudio/blob/main/Studio/build.zig) to see how the source code is compiled and where the libraries are expected.

#### Running the Renderer

Once the binary has been created, you can run it and automatically capture the streams from the raspberry pi's

## Part list
Most of these parts can be found for cheaper from Aliexpress, etc.


### Stereo Camera Setup

- 2x [Raspberry Pi Zero 2W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/)  
- 2x [Raspberry Pi Camera Module 3](https://www.raspberrypi.com/products/camera-module-3/)  
- Motion Processing Unit (MPU): [Mpu-9250](https://www.amazon.com/dp/B01I1J0Z7Y?ref=ppx_yo2ov_dt_b_fed_asin_title)
- Rechargeable LiPo battery (sized appropriately to power both Raspberry Pi Zeros and cameras)
- 3D printed stereo camera housing (designed to hold both Pi boards and cameras exactly 6 cm apart)
- Small hardware (bolts/nuts) to secure cameras and Pi boards in housing


### Drone
- Electronic Speed Controller (ESC): [Jhemcu EM40a 4-in-1](https://www.aliexpress.us/item/2255799889128915.html?spm=a2g0o.productlist.main.1.3c37q24Bq24BjE&algo_pvid=f9d88402-ce9d-4e3c-a412-ddb82016e26e&algo_exp_id=f9d88402-ce9d-4e3c-a412-ddb82016e26e-2&pdp_npi=4%40dis%21USD%2123.99%2116.79%21%21%2123.99%2116.79%21%402103245417303536374263010edcc3%2112000024266524726%21sea%21US%216164672369%21X&curPageLogUid=e6vfdxoFeAjy&utparam-url=scene%3Asearch%7Cquery_from%3A)
- Frame: [5" Carbon Fiber (looking into making 3" 3D printed frame)](https://www.aliexpress.us/item/3256806814368043.html?spm=a2g0o.productlist.main.17.56a134f3a8JusT&algo_pvid=1ff9789c-b6ee-44c9-9c11-23c2d6d53f0c&algo_exp_id=1ff9789c-b6ee-44c9-9c11-23c2d6d53f0c-9&pdp_npi=4%40dis%21USD%2125.86%2115.26%21%21%2125.86%2115.26%21%402103209b17303538958174736ead90%2112000039007991042%21sea%21US%216164672369%21X&curPageLogUid=XdxX0TVz20sx&utparam-url=scene%3Asearch%7Cquery_from%3A)
- 4 Motors: [2205 2300kV 3-phase BLDC motor](https://www.aliexpress.us/item/3256806367616768.html?spm=a2g0o.productlist.main.3.3184ZtfdZtfd8D&algo_pvid=d8c1175d-6549-49c9-9d73-1175be48bbf3&algo_exp_id=d8c1175d-6549-49c9-9d73-1175be48bbf3-1&pdp_npi=4%40dis%21USD%2111.69%216.90%21%21%2111.69%216.90%21%402101c5a717303537115667019ee4af%2112000037653045149%21sea%21US%216164672369%21X&curPageLogUid=LovAIBWsRDjV&utparam-url=scene%3Asearch%7Cquery_from%3A)
- Propellers: 5" (too many options, most are fine)
- LiPo Battery: [3s 2200 mAh (35C)](https://www.amazon.com/dp/B0CS2YZCYD?ref=ppx_yo2ov_dt_b_fed_asin_title)
- XT90 Connectors: Female (Might be included with ESC)

### Tools Needed
- Linux / Windows Computer (might cross-compile for darwin later)
- 3D Printer
- Soldering Iron
- Flux
- Power Supply / battery chargers
- 22 awg wire
- Wire cutter / stripper
- Resistors (optional)
- Potentiometer (optional)
- Multimeter (optional)
- Oscilloscope (optional)


## Milestones

### 1. Basic Rendering

- :white_check_mark: Zig openGL bindings
- :white_check_mark: Create window
- :white_check_mark: Render 'Hello Triangle'
- :white_check_mark: Render grid
- :white_check_mark: Setup basic rendering pipeline
- :white_check_mark: Each vertex assigned a color using shaders
- :white_check_mark: Render box
- :white_check_mark: Color each face of the box differently
- :white_check_mark: Create hierarchy of nodes instead of flatmap of meshes
- :white_check_mark: Find way to position children as offset to parents position (rotation & scale too)
- :white_check_mark: Convert Euler rotations / transformations to Quaternions
- :white_square_button: Profile rendering time / mem allocations. Check to see if llvm IR uses SIMD automatically
- :white_square_button: Use SIMD in Vec and Matrix calculations
- :white_square_button: Change line rendering to quads
- :white_square_button: Kill threads / end processes when program closed

### 1.2 Camera

- :white_check_mark: Perspective Camera
- :white_check_mark: WASD movement
- :white_check_mark: Sprinting / speed up with Shift
- :white_check_mark: Free look with mouse
- :white_check_mark: Zooming in / out
- :white_square_button: Look into using openGL Frustum culling


### 2 Build system and ImGui Integration

- :white_check_mark: Embed Shaders inline instead of reading from fs
- :white_check_mark: Cross Compile for Windows
- :white_square_button: Find way to link dependencies without using @cimport ???
- :white_square_button: Create zig bindings for ImGui
- :white_square_button: Create drop down menus for basic configuration editing 
- :white_square_button: FPS counter
- :white_square_button: Debug / view meshes and vertices

### 3. Obtaining Poses

- :white_check_mark: Develop ESP firmware for MPU-9250 (no longer using esp32, convert to raspberry pi)
- :white_check_mark: Set up UDP server on ESP (no longer using esp32, convert to raspberry pi)
- :white_check_mark: Set up UDP server in renderer 
- :white_check_mark: Convert []u8 into little-endian signed f32
- :white_check_mark: Optimize Wifi UDP Tx speed (change to AP instead of relying on router)
- :white_check_mark: Try to get 1 kHz refresh rate on gyro
- :white_check_mark: Add timestamp / checksum to UDP packets to discard old poses

### 4. MPU Integration with Model

- :white_check_mark: Obtain Accelerometer data in renderer
- :white_check_mark: Obtain Gyro data in renderer
- :white_check_mark: Obtain Magnetometer in renderer
- :white_check_mark: Kalman filter on gyro & accelerometer for estimating Pitch and Roll
- :white_check_mark: Kalman filter on gyro and magnetometer for estimating Yaw
- :white_check_mark: Measure fps from MPU & incrementally update kalman filter delta time
- :white_check_mark: Find out why varying dt on kalman filter results in janky movement (Delta time calculation wasnt being calculated correctly)
- :white_check_mark: Convert Kalman filter to Madgwick filter
- :white_check_mark: Calibrate sensors
    - :white_check_mark: Larger Accelerometer range (+- 8g's)
    - :white_check_mark: 0 values when not moving
    - :white_check_mark: Store calibration settings to disk, to avoid having to run on every start-up (also option to recalibrate)
    - :white_square_button: Continuous magnetometer hard iron and soft iron calibration. (store max / min on renderer and update every x minutes)
- :white_square_button: Add compass overlay
- :white_square_button: Overlay avg of sensor data / Tx rate
- :white_square_button: Barometric sensor for altitude control??

### 5. Stereo Camera Integration

- :white_check_mark: Capture video frames from Raspberry Pi Zero 2W in desired resolution
- :white_check_mark: Find lowest latency mode of transport (currently testing raw UDP and RTP)
    - :white_check_mark: Create WiFi access point on server to minimize network latency
- :white_check_mark: Receive video stream on processing server
- :white_check_mark: Create pipeline to decode stream into frames using hardware acceleration
- :white_check_mark: Parse frames (YUV420) into FBO for visualization in OpenGL
- :white_check_mark: Build and 3D-print STL for stereo camera housing (2x RPi Zero 2W, 2x RPi Camera Module 3, 6 cm apart)
    - :white_square_button: Explore adding rechargeable LiPo battery to power both Raspberry Pis
- :white_square_button: Configure streams from each camera to be synced to server time using NTP
- :white_check_mark: Pair frames from each stream
- :white_check_mark: Derive distance from paired frames
    - Keypoint Detection
        - :white_check_mark: FAST Keypoint Detection
        - :white_check_mark: Gaussian Blurring
        - Visualizing Keypoints
            - :white_check_mark: Left canvas
            - :white_check_mark: Right canvas
            - :white_check_mark: Combined canvas
    - Keypoint Matching
        - :white_check_mark: Match based on argmin score, where score => weighted sum:
            - Hamming distance of Gaussian BRIEF Descriptor
            - Epipolar distance
            - Disparity
        - :white_square_button: Scale Invariance
        - :white_square_button: Rotation Invariance
        - Visualize Matching
            - :white_check_mark: Triangulation from Left => Combined <= Right
            - :white_square_button: Interactive matches to see matching details (disparity, coords, hamming distance, etc)
    - Finding nearest Matched keypoint
        - :white_check_mark: Argmin Euclidean distance from Matched Keypoints 
    - Assigning distances per pixel (look into quadtree etc)
        - :white_square_button: Visualize combined image as 3D plane
        - :white_square_button: Visualize combined image as depthmap
- :white_square_button: Derive new pose from previous frames
- :white_square_button: Fuse MPU pose data with visual odometry for more robust pose estimation
- :white_square_button: Project 3D texture into scene based on pose and depth information  
- :white_square_button: Visual odometry using pose


### 6. Drone & Motor Control
- :white_check_mark: ESP firmware for driving BLHELI_S based ESC (no longer using esp32, convert to RPi)
- :white_square_button: Create testing rig for running motor w/ propellers
- :white_square_button: Create 3D model for mounting ESC + RPi's + MPU-9250 + Cameras
- :white_check_mark: Add support for potentiometer based throttle
- :white_square_button: Lerp between current & min throttle when changing states
- :white_square_button: Arduino as bootloader to flash necessary config to ESC ??
- :white_square_button: PID loop
- :white_square_button: Look into using smaller chassis / motors

### 7. Model Controls 

- :white_square_button: Implement yaw / roll based on keyboard movement
- :white_square_button: Implement pitch using mouse inputs
- :white_square_button: Orbit Controls instead of Perspective
- :white_square_button: UDP Tx from renderer with Yaw / Pitch / Roll / Throttle
- :white_square_button: UDP Rx on RPi to receive & control drone 

### 8. Visualization

- :white_square_button: Guassian Splatting to visualize captured video.
- :white_square_button: Ability to walk around / tour room in real time
- :white_square_button: Colored / textured meshes
- :white_square_button: If segmented objects. highlight / click on those with details

### 9. SLAM (Simultaneous Localization and Mapping)

- :white_square_button: Combine MPU & LiDAR / Image data to associate FOV with location
- :white_square_button: Implement visual SLAM to stitch together FOV's / map the room in 3D


### 10. 3D Model Integration

- :white_square_button: Import GLTF models
- :white_square_button: Scan drone and import its 3D model

### 11. Drone Simulator

- :white_square_button: Import / Randomly generate scene
- :white_square_button: Scene invisible but vertex data available
- :white_square_button: Spawn drone in random position
- :white_square_button: Raycasting to simulate lidar
- :white_square_button: Map scene and route taken by drone w/ collisions
- :white_square_button: Make scene dynamic / changing overtime to see behavior

### 12. Synchronization

- :white_square_button: Implement synchronization for multiple devices or components
- :white_square_button: Swarm mode vs. Divide & conquer?
- :white_square_button: maybe some type of animations?


### 13. Create custom PCB's for ESC

- :white_square_button: Model components in LTSpice
- :white_square_button: Create PCB model gerber file
- :white_square_button: Print PCB's from manufacturer
- :white_square_button: Solder SMD's to printed pcb
- :white_square_button: Flash BLHELI_S or write custom firmware


### 14. Convert entire Repo to Zig

- :white_square_button: Generate glad bindings in zig
- :white_square_button: Create HAL for ESP32 in Zig