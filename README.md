# Drone Studio

Autonomous drone firmware, renderer and simulator for mapping environments.

![alt text](https://github.com/Eykam/DroneStudio/blob/main/assets/hero.webp)

## Overall Goal

Goal at the moment is to have the drone autonomously map a building in 3D using LiDAR / photogrammetry. The renderer will then allow the user to walk around, view the scene like a video game. 

Should be able to use pose from MPU (mpu-9250) along with distance data from lidar sensors to construct pose-graph of environment. With current sensors, would be able to obtain and transmit 1 khz of pose data, and 60 hz of distance data (looking into faster LiDAR sensors), over a UDP Tx server. The UDP Tx server will be hosted on a 2.4ghz access point from the esp32.

Later goals are to add object avoidance, image detection and synchronization with other drones within the same network / swarm.

## Part list

### Drone
- Flight Controller => Esp32
  - Motion Processing Unit (MPU) => Mpu-9250
  - LiDAR => VL53L1X 
- Electronic Speed Controller (ESC) => Jhemcu EM-40 4-in-1
- Frame => 5" Carbon Fiber (looking into making 3" 3D printed frame)
- Motors => 2205 2300kV 3-phase BLDC motor
- Propellers => 5" (included with motors)
- LiPo Battery => 3s 2200 mAh (35C)

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

- [X] Zig openGL bindings
- [X] Create window
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
