TODO:
[] - Add input / controller capture & replay
    [x] - Timeline to show progress & step forward / backward
        [] - Update timeline to look better
    [x] - Create Start / Stop recording button
        [] - On stop dialog to select path to save
    [x] - Create Play button to playback recording 
        [] - Store dt in PackedTransform
        [x] - Use seek to playback?
        [x] - Current Seek (00:00 / 10:00)
        [] - Work with new physics system
        [] - 1|2|4|8x speed
    [x] - Save to path / etc
    [x] - UI to select saved configs
    [] - Flag to stop transform from being captured
    [] - Automatic path generation ??
[] - Add visualization for path
    [] - Only shown in Free Camera
    [] - Previous steps
    [] - Next steps
[x] - IPC protocol using mmap to communicate between sim & python
    [x] - Create IPC shared mem using /dev/shm fd
        [] - Find way to do this cross-platform
    [x] - Create basic protocol to share viewports
    [x] - Register VP buffers with cuda (maybe opencl??)
        [] - Add fallback for non-cuda where write pixels to shm & read in python
    [x] - Map available camera / VP buffers
    [] - Python lib / package that users can plug into
        [x] - Define cv pipeline & use decorator to run them
        [] - Hot reloading?
    [] - Find way to access them typesafe???
    [] - Create demo depthmap using opencv
    [] - give VP back to engine to render??
[X] - Generate colliders / convex hulls of meshes
    [] - Use CoACD algo 
        [] - Possibly implement CoACD w/ hardware acceleration
[X] - Add basic physics (gravity, collisions)
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
[X] - Make Skeleton / Show skeleton option of meshes
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



FIX:
[x] - Flag to stop FBO from resizing with window
[X] - Maintain aspect ratio when resizing viewport
[] - Fix transformsystem to do a basic dfs w/ dirty flag
[X] - Figure out why asset caching isnt working properly
[] - Make new branch to store old studio code
[] - Write docs / landing page on Engine
[X] - Figure out why viewports initialize to black until toggleViewport 
[X] - Make main viewport larger (no max width)
[] - Frustum only visible in Free / Debug mode
[] - Move Active Flags from Camera => Viewport
    [] - Update Viewport Selector in UI
[] - Remove global component to global system
[] - Remove PBRTexture & material structs from Mesh
[] - Remove legacy Pipeline & code
[] - Remove old physics components / systems
    [] - Rename collisions.zig => PhysicsAdapter.zig
    [] - Rename PhysicsThread => PhysicsSimulator.zig
[] - Make ECS completely generic (no usage of specific components / systems)
[] - Make ResourceManager component / system agnostic (no imports or usages of components / systems)
[] - More modular testing
