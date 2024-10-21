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
- [ ] Render box
- [ ] Color each face of the box differently

### 2. Box Interaction

- [ ] Implement box rotation based on mouse movement
- [ ] Implement box rotation using keyboard inputs

### 2.1 Build system and ImGui Integration

- [] Embed Shaders inline instead of reading from fs
- [] Find way to link dependencies without using @cimport ???
- [] Create zig bindings for ImGui
- [] Create drop down menus for basic configuration editing 



### 3. Sensor Integration

- [ ] Add compass overlay
- [ ] Integrate gyro data
- [ ] Overlay acceleration data
- [ ] Connect MPU (Motion Processing Unit) data to the box

### 4. Communication

- [ ] Develop ESP firmware
- [ ] Set up UDP server for communication
- [ ] Ensure reliable data transmission between devices


## Future Milestones

### 5. 3D Model Integration

- [ ] Import GLTF models
- [ ] Scan drone and import its 3D model

### 6. Motor Control

- [ ] Calibrate motor speeds
- [ ] Implement motor speed control algorithms

### 7. LiDAR Integration

- [ ] Define LiDAR sensor field of view (FOV)
- [ ] Visualize FOV's captured by scanners in 3D space
- [ ] Develop ESP firmware for LiDAR integration

### 8. Camera and Depth Sensing

- [ ] Integrate camera functionality
- [ ] Utilize depth of field models for enhanced visualization / segmentation

### 9. SLAM (Simultaneous Localization and Mapping)

- [ ] Implement SLAM to map the room in 3D

### 10. Visualization

- [ ] Visualize 3D models within DroneStudio

### 11. Synchronization

- [ ] Implement synchronization for multiple devices or components

### 12. Convert entire Repo to Zig

- [ ] Generate glad bindings in zig
- [ ] Create HAL for ESP32 in Zig
