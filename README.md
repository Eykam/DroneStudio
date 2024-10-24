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

### 3. Communication

- [ ] Develop ESP firmware for MPU-9250
- [ ] Set up UDP server for communication
- [ ] Abstraction / protocol for sending data between esp & renderer

### 4. MPU Integration

- [ ] Add compass overlay
- [ ] Integrate gyro data
- [ ] Overlay acceleration data
- [ ] Connect MPU (Motion Processing Unit) data to the box

### 5. LiDAR Integration

- [ ] Develop ESP firmware for LiDAR integration
- [ ] Define LiDAR sensor field of view (FOV)
- [ ] Find way to calibrate sensors / get physical location of areas in FOV
- [ ] Visualize FOV's captured by scanners in 3D space relative to drone

### 6. Camera and Depth Sensing

- [ ] Use camera to stream image data to renderer
- [ ] Use image data to color mesh
- [ ] Use some image => depth map transformers
- [ ] Maybe some segmentation? 

### 7. SLAM (Simultaneous Localization and Mapping)

- [ ] Combine MPU & LiDAR / Image data to associate FOV with location
- [ ] Implement SLAM to stitch together FOV's / map the room in 3D

### 8. Visualization

- [ ] Cube Marching to turn point clouds into mesh.
- [ ] Visualize 3D models of scan in DroneStudio
- [ ] Ability to walk around / tour room in real time
- [ ] Colored / textured meshes
- [ ] If segmented objects. highlight / click on those with details

### 9. 3D Model Integration

- [ ] Import GLTF models
- [ ] Scan drone and import its 3D model

### 10. Motor Control

- [ ] Calibrate motor speeds
- [ ] Implement motor speed control algorithms

### 11. Synchronization

- [ ] Implement synchronization for multiple devices or components

### 12. Convert entire Repo to Zig

- [ ] Generate glad bindings in zig
- [ ] Create HAL for ESP32 in Zig
