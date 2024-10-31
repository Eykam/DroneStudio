# Drone Studio

Autonomous drone firmware, renderer and simulator for mapping environments.

## Overall Goal

Goal at the moment is to have the drone autonomously map a building in 3D using LiDAR / photogrammetry. The renderer will then allow the user to walk around, view the scene like a video game. 

![alt text](https://github.com/Eykam/DroneStudio/blob/main/assets/hero.png)
(Not my image, but similar to what I'd like to produce)

Should be able to use pose from MPU (mpu-9250) along with distance data from lidar sensors to construct pose-graph of environment. With current sensors, would be able to obtain and transmit 1 khz of pose data, and 60 hz of distance data (looking into faster LiDAR sensors), over a UDP Tx server. The UDP Tx server will be hosted on a 2.4ghz access point from the esp32.

Later goals are to add object avoidance, image detection and synchronization with other drones within the same network / swarm.

## Part list
Most of these parts can be found for cheaper from Aliexpress, etc.

### Drone
- Flight Controller => [Esp32](https://www.amazon.com/dp/B09GK74F7N?ref=ppx_yo2ov_dt_b_fed_asin_title)
  - Motion Processing Unit (MPU) => [Mpu-9250](https://www.amazon.com/dp/B01I1J0Z7Y?ref=ppx_yo2ov_dt_b_fed_asin_title)
  - LiDAR => [VL53L1X](https://www.amazon.com/dp/B0CLDKMGZR?ref=ppx_yo2ov_dt_b_fed_asin_title)
- Electronic Speed Controller (ESC) => [Jhemcu EM40a 4-in-1](https://www.aliexpress.us/item/2255799889128915.html?spm=a2g0o.productlist.main.1.3c37q24Bq24BjE&algo_pvid=f9d88402-ce9d-4e3c-a412-ddb82016e26e&algo_exp_id=f9d88402-ce9d-4e3c-a412-ddb82016e26e-2&pdp_npi=4%40dis%21USD%2123.99%2116.79%21%21%2123.99%2116.79%21%402103245417303536374263010edcc3%2112000024266524726%21sea%21US%216164672369%21X&curPageLogUid=e6vfdxoFeAjy&utparam-url=scene%3Asearch%7Cquery_from%3A)
- Frame => [5" Carbon Fiber (looking into making 3" 3D printed frame)](https://www.aliexpress.us/item/3256806814368043.html?spm=a2g0o.productlist.main.17.56a134f3a8JusT&algo_pvid=1ff9789c-b6ee-44c9-9c11-23c2d6d53f0c&algo_exp_id=1ff9789c-b6ee-44c9-9c11-23c2d6d53f0c-9&pdp_npi=4%40dis%21USD%2125.86%2115.26%21%21%2125.86%2115.26%21%402103209b17303538958174736ead90%2112000039007991042%21sea%21US%216164672369%21X&curPageLogUid=XdxX0TVz20sx&utparam-url=scene%3Asearch%7Cquery_from%3A)
- 4 Motors => [2205 2300kV 3-phase BLDC motor](https://www.aliexpress.us/item/3256806367616768.html?spm=a2g0o.productlist.main.3.3184ZtfdZtfd8D&algo_pvid=d8c1175d-6549-49c9-9d73-1175be48bbf3&algo_exp_id=d8c1175d-6549-49c9-9d73-1175be48bbf3-1&pdp_npi=4%40dis%21USD%2111.69%216.90%21%21%2111.69%216.90%21%402101c5a717303537115667019ee4af%2112000037653045149%21sea%21US%216164672369%21X&curPageLogUid=LovAIBWsRDjV&utparam-url=scene%3Asearch%7Cquery_from%3A)
- Propellers => 5" (too many options, most are fine)
- LiPo Battery => [3s 2200 mAh (35C)](https://www.amazon.com/dp/B0CS2YZCYD?ref=ppx_yo2ov_dt_b_fed_asin_title)
- XT90 Connectors => Female (Might be included with ESC)

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
- :white_large_square: Create hierarchy of nodes instead of flatmap of meshes
- :white_large_square: Change line rendering to quads
- :white_large_square: Kill threads / end processes when program closed

### 1.2 Camera

- :white_check_mark: Perspective Camera
- :white_check_mark: WASD movement
- :white_check_mark: Sprinting / speed up with Shift
- :white_check_mark: Free look with mouse
- :white_check_mark: Zooming in / out
- :white_large_square: Look into using openGL Frustum instead

### 2. Model Controls 

- :white_large_square: Implement yaw / roll based on keyboard movement
- :white_large_square: Implement pitch using mouse inputs
- :white_large_square: Orbit Controls instead of Perspective
- :white_large_square: UDP Tx from renderer with Yaw / Pitch / Roll / Throttle
- :white_large_square: UDP Rx on ESP to receive & control drone

### 2.1 Build system and ImGui Integration

- :white_check_mark: Embed Shaders inline instead of reading from fs
- :white_check_mark: Cross Compile for Windows
- :white_large_square: Find way to link dependencies without using @cimport ???
- :white_large_square: Create zig bindings for ImGui
- :white_large_square: Create drop down menus for basic configuration editing 
- :white_large_square: FPS counter
- :white_large_square: Debug / view meshes and vertices

### 3. Obtaining Poses

- :white_check_mark: Develop ESP firmware for MPU-9250
- :white_check_mark: Set up UDP server on ESP
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
- :white_large_square: Kalman filter on gyro and magnetometer for estimating Yaw
- :white_check_mark: Measure fps from MPU & incrementally update kalman filter delta time
- :white_large_square: Find out why varying dt on kalman filter results in janky movement
- :white_large_square: Calibrate sensors
    - :white_check_mark: Larger Accelerometer range (+- 8g's)
    - :white_large_square: 0 values when not moving
- :white_large_square: Add compass overlay
- :white_large_square: Overlay avg of sensor data / Tx rate
- :white_large_square: Lidar / Barometric sensor for altitude control

### 5. LiDAR Integration

- :white_large_square: Develop ESP firmware for LiDAR integration
- :white_large_square: Relate SPAD matrix with current pose from MPU
- :white_large_square: Interpolate points between SPAD matrix obtained from sensor
- :white_large_square: Visualize FOV's captured by scanners in 3D space relative to drone

### 6. Motor Control
- :white_check_mark: ESP firmware for driving BLHELI_S based ESC
- :white_large_square: Create testing rig for running motor w/ propellers
- :white_check_mark: Add support for potentiometer based throttle
- :white_large_square: Lerp between current & min throttle when changing states
- :white_large_square: Arduino as bootloader to flash necessary config to ESC
- :white_large_square: PID loop
- :white_large_square: Look into using smaller chassis / motors

### 7. Visualization

- :white_large_square: Cube Marching to turn point clouds into mesh.
- :white_large_square: Visualize 3D models of scan in DroneStudio
- :white_large_square: Ability to walk around / tour room in real time
- :white_large_square: Colored / textured meshes
- :white_large_square: If segmented objects. highlight / click on those with details

### 8. SLAM (Simultaneous Localization and Mapping)

- :white_large_square: Combine MPU & LiDAR / Image data to associate FOV with location
- :white_large_square: Implement SLAM to stitch together FOV's / map the room in 3D

### 9. Camera and Depth Sensing

- :white_large_square: Use camera to stream image data to renderer
- :white_large_square: Use image data to color mesh
- :white_large_square: Use some image => depth map transformers
- :white_large_square: Maybe some segmentation? 

### 10. 3D Model Integration

- :white_large_square: Import GLTF models
- :white_large_square: Scan drone and import its 3D model

### 11. Drone Simulator

- :white_large_square: Import / Randomly generate scene
- :white_large_square: Scene invisible but vertex data available
- :white_large_square: Spawn drone in random position
- :white_large_square: Raycasting to simulate lidar
- :white_large_square: Map scene and route taken by drone w/ collisions
- :white_large_square: Make scene dynamic / changing overtime to see behavior

### 12. Synchronization

- :white_large_square: Implement synchronization for multiple devices or components
- :white_large_square: Swarm mode vs. Divide & conquer?
- :white_large_square: maybe some type of animations?


### 13. Create custom PCB's for ESC

- :white_large_square: Model components in LTSpice
- :white_large_square: Create PCB model gerber file
- :white_large_square: Print PCB's from manufacturer
- :white_large_square: Solder SMD's to printed pcb
- :white_large_square: Flash BLHELI_S or write custom firmware


### 14. Convert entire Repo to Zig

- :white_large_square: Generate glad bindings in zig
- :white_large_square: Create HAL for ESP32 in Zig
