Bugs to fix:

- [X]  All Textured planes use the same texture instead of being separately bound
    - Each node now has a unique generated texture Unit
    - y and uv texture IDs are now generated once during initialization to save memory
- []  Crash if video stream already running
- []  Indices are not used correctly on some shapes (Rectangle and Grid)
- []  Generic function to process movement vector in camera