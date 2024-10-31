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
- :white_large_square: Create window
- [X] Render 'Hello Triangle'
- [X] Render grid
- [X] Setup basic rendering pipeline
- [X] Each vertex assigned a color using shaders
- [X] Render box
- [X] Color each face of the box differently
- [ ] Create hierarchy of nodes instead of flatmap of meshes
- [ ] Change line rendering to quads
- [ ] Kill threads / end processes when program closed

### 1.2 Camera

- [X] Perspective Camera
- [X] WASD movement
- [X] Sprinting / speed up with Shift
- [X] Free look with mouse
- [X] Zooming in / out
- [ ] Look into using openGL Frustum instead

### 2. Model Controls 

- [ ] Implement yaw / roll based on keyboard movement
- [ ] Implement pitch using mouse inputs
- [ ] Orbit Controls instead of Perspective
- [ ] UDP Tx from renderer with Yaw / Pitch / Roll / Throttle
- [ ] UDP Rx on ESP to receive & control drone

### 2.1 Build system and ImGui Integration

- [X] Embed Shaders inline instead of reading from fs
- [X] Cross Compile for Windows
- [ ] Find way to link dependencies without using @cimport ???
- [ ] Create zig bindings for ImGui
- [ ] Create drop down menus for basic configuration editing 
- [ ] FPS counter
- [ ] Debug / view meshes and vertices

### 3. Obtaining Poses

- [X] Develop ESP firmware for MPU-9250
- [X] Set up UDP server on ESP
- [x] Set up UDP server in renderer 
- [x] Convert []u8 into little-endian signed f32
- [X] Optimize Wifi UDP Tx speed (change to AP instead of relying on router)
- [X] Try to get 1 kHz refresh rate on gyro
- [X] Add timestamp / checksum to UDP packets to discard old poses

### 4. MPU Integration with Model

- [X] Obtain Accelerometer data in renderer
- [X] Obtain Gyro data in renderer
- [X] Obtain Magnetometer in renderer
- [X] Kalman filter on gyro & accelerometer for estimating Pitch and Roll
- [ ] Kalman filter on gyro and magnetometer for estimating Yaw
- [X] Measure fps from MPU & incrementally update kalman filter delta time
- [ ] Find out why varying dt on kalman filter results in janky movement
- [ ] Calibrate sensors
    - [X] Larger Accelerometer range (+- 8g's)
    - [ ] 0 values when not moving
- [ ] Add compass overlay
- [ ] Overlay avg of sensor data / Tx rate
- [ ] Lidar / Barometric sensor for altitude control

### 5. LiDAR Integration

- [ ] Develop ESP firmware for LiDAR integration
- [ ] Relate SPAD matrix with current pose from MPU
- [ ] Interpolate points between SPAD matrix obtained from sensor
- [ ] Visualize FOV's captured by scanners in 3D space relative to drone

### 6. Motor Control
- [X] ESP firmware for driving BLHELI_S based ESC
- [ ] Create testing rig for running motor w/ propellers
- [X] Add support for potentiometer based throttle
- [ ] Lerp between current & min throttle when changing states
- [ ] Arduino as bootloader to flash necessary config to ESC
- [ ] PID loop
- [ ] Look into using smaller chassis / motors

### 7. Visualization

- [ ] Cube Marching to turn point clouds into mesh.
- [ ] Visualize 3D models of scan in DroneStudio
- [ ] Ability to walk around / tour room in real time
- [ ] Colored / textured meshes
- [ ] If segmented objects. highlight / click on those with details

### 8. SLAM (Simultaneous Localization and Mapping)

- [ ] Combine MPU & LiDAR / Image data to associate FOV with location
- [ ] Implement SLAM to stitch together FOV's / map the room in 3D

### 9. Camera and Depth Sensing

- [ ] Use camera to stream image data to renderer
- [ ] Use image data to color mesh
- [ ] Use some image => depth map transformers
- [ ] Maybe some segmentation? 

### 10. 3D Model Integration

- [ ] Import GLTF models
- [ ] Scan drone and import its 3D model

### 11. Drone Simulator

- [ ] Import / Randomly generate scene
- [ ] Scene invisible but vertex data available
- [ ] Spawn drone in random position
- [ ] Raycasting to simulate lidar
- [ ] Map scene and route taken by drone w/ collisions
- [ ] Make scene dynamic / changing overtime to see behavior

### 12. Synchronization

- [ ] Implement synchronization for multiple devices or components
- [ ] Swarm mode vs. Divide & conquer?
- [ ] maybe some type of animations?


### 13. Create custom PCB's for ESC

- [ ] Model components in LTSpice
- [ ] Create PCB model gerber file
- [ ] Print PCB's from manufacturer
- [ ] Solder SMD's to printed pcb
- [ ] Flash BLHELI_S or write custom firmware


### 14. Convert entire Repo to Zig

- [ ] Generate glad bindings in zig
- [ ] Create HAL for ESP32 in Zig
