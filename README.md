# Drone Studio

Autonomous drone firmware and viewer for data collected through various sensors.


## Overall Goal

Goal at the moment is to have the drone autonomously map a room in 3D using lidar / photogrammetry. Once the room is mapped, it should hover in the center. 

Later goals are to add object avoidance, image detection and synchronization with other drones within the same network / swarm.

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
- [ ] Create Hierarchy of nodes instead of flatmap of meshes
- [ ] Change line rendering to quads

### 1.2 Camera

- [X] Perspective Camera
- [X] WASD movement
- [X] Sprinting / speed up with Shift
- [X] Free look with mouse
- [X] Zooming in / out
- [ ] Look into using openGL Frustum instead

### 2. Box Interaction

- [ ] Implement yaw / roll based on keyboard movement
- [ ] Implement pitch using mouse inputs
- [ ] Orbit Controls instead of Perspective

### 2.1 Build system and ImGui Integration

- [X] Embed Shaders inline instead of reading from fs
- [X] Cross Compile for Windows
- [ ] Find way to link dependencies without using @cimport ???
- [ ] Create zig bindings for ImGui
- [ ] Create drop down menus for basic configuration editing 
- [ ] FPS counter
- [ ] Debug / view meshes and vertices

### 3. Communication

- [X] Develop ESP firmware for MPU-9250
- [X] Set up UDP server on ESP
- [x] Set up UDP server in renderer 
- [x] Convert []u8 into little-endian signed f32
- [ ] Optimize Wifi UDP Tx speed (currently around 500Kb/s ideally would need 1Mb/s)
- [ ] Add timestamp / checksum to UDP packets to discard old poses
- [ ] Kill threads / end processes when program closed

### 4. MPU Integration

- [X] Connect Accelerometer data to cube uniform
- [X] Connect Gyro data to cube uniform
- [ ] Connect Magnetometer data to cube uniform 
- [X] Kalman filter or some type of cleaning of MPU data
- [ ] Measure fps from MPU & incrementally update kalman filter delta time 
- [ ] Calibrate sensors
    - [X] Larger Accelerometer range (+- 8g's)
    - [ ] 0 values when not moving
- [ ] Add compass overlay
- [ ] Overlay avg of sensor data

### 5. LiDAR Integration

- [ ] Develop ESP firmware for LiDAR integration
- [ ] Relate SPAD matrix with current pose from MPU
- [ ] Interpolate points between SPAD matrix obtained from sensor
- [ ] Visualize FOV's captured by scanners in 3D space relative to drone


### 6. Visualization

- [ ] Cube Marching to turn point clouds into mesh.
- [ ] Visualize 3D models of scan in DroneStudio
- [ ] Ability to walk around / tour room in real time
- [ ] Colored / textured meshes
- [ ] If segmented objects. highlight / click on those with details


### 7. SLAM (Simultaneous Localization and Mapping)

- [ ] Combine MPU & LiDAR / Image data to associate FOV with location
- [ ] Implement SLAM to stitch together FOV's / map the room in 3D

### 8. Camera and Depth Sensing

- [ ] Use camera to stream image data to renderer
- [ ] Use image data to color mesh
- [ ] Use some image => depth map transformers
- [ ] Maybe some segmentation? 

### 9. 3D Model Integration

- [ ] Import GLTF models
- [ ] Scan drone and import its 3D model

### 10. Drone Simulator

- [ ] Import / Randomly generate scene
- [ ] Scene invisible but vertex data available
- [ ] Spawn drone in random position
- [ ] Raycasting to simulate lidar
- [ ] Map scene and route taken by drone w/ collisions
- [ ] Make scene dynamic / changing overtime to see behavior

### 11. Motor Control

- [ ] Calibrate motor speeds
- [ ] Implement motor speed control algorithms

### 12. Synchronization

- [ ] Implement synchronization for multiple devices or components


### 13. Convert entire Repo to Zig

- [ ] Generate glad bindings in zig
- [ ] Create HAL for ESP32 in Zig
