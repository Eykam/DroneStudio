TODO:
[] - Add input / controller capture & replay
    [] - Config to store / cache paths
    [] - UI to select paths
    [] - Timeline to show progress & step forward / backward
    [] - Automatic path generation ??
[] - Add visualization for path
    [] - Previous steps
    [] - Next steps
[] - IPC protocol using mmap to communicate between sim & python
[] - Generate colliders / convex hulls of meshes
[] - Add basic physics (gravity, collisions)
[] - Implement Drone flying physics
    [] - Controller Input => DSHOT => Thrust
    [] - Model Battery
        [] - type (3s,4s,6s,etc)
        [] - Voltage
        [] - Max Current
        [] - mAH
        [] - current %
        [] - internal resistance (voltage sag)
    [] - Model ESC
        [] - Throttle => Voltage
        [] - Voltage draw
        [] - Current draw
        [] - Voltage sag
    [] - Model Motor
        [] - Propellor size / pitch
        [] - Voltage & Current => RPM
        [] - Motor KV & Stator size
        [] - RPM & prop dims => Thrust
    [] - Wind Effect & possibly fluid dynamics
    [] - 4 Motor Thrusts => Accel, Velocity & Position
[] - Implement animations from GLTF
    [] - speed of animation depending on RPM of motors
[] - Make Skeleton / Show skeleton option of meshes
[] - Scene / Asset import. Ability to set positions & save.
[] - Framework for testing accuracy of VIO
    [] - Define cameras / lidar
    [] - Define OpenCV pipeline to use captured image / lidar data
    [] - Raycast in scene to get real distances / env mapping
    [] - Add realistic lighting & other env / weather effects
    [] - Generate random / select from presets of environments & paths to take
    [] - Compare captured mapping to real mapping
[] - Framework for testing RL
    [] - Similar to above, but using gymnasium
    [] - Expose scene data to user to use as input to agent
[] - Dynamic Scenes
