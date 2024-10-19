const std = @import("std");
const Transformations = @import("Transformations.zig");
const Pipeline = @import("Pipeline.zig");
const Scene = Pipeline.Scene;
const Shape = @import("Shape.zig");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("glad/glad.h"); // Include GLAD
    @cInclude("GLFW/glfw3.h");
});

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const alloc = arena.allocator();

    // Initialize GLFW
    if (c.glfwInit() == 0) {
        std.debug.print("Failed to initialize GLFW\n", .{});
        return;
    }

    defer c.glfwTerminate();

    // Set OpenGL version (3.3 Core)
    c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MAJOR, 3);
    c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MINOR, 3);
    c.glfwWindowHint(c.GLFW_OPENGL_PROFILE, c.GLFW_OPENGL_CORE_PROFILE);

    // Create a windowed mode window and its OpenGL context
    const width = comptime 1920 * 0.75;
    const height = comptime 1080 * 0.75;
    const window = c.glfwCreateWindow(width, height, "Drone Studio", null, null);
    defer c.glfwDestroyWindow(window);

    //Initializing Scene
    var scene = try Scene.init(alloc, window);

    // Define transformation matrices
    const projection = Transformations.perspective(45.0, width / height, 0.1, 100.0);
    const eye = [3]f32{ 0.0, 0.0, 5.0 }; // Camera position
    const center = [3]f32{ 0.0, 0.0, 0.0 }; // Look at origin
    const up = [3]f32{ 0.0, 1.0, 0.0 }; // Up vector
    const view = Transformations.lookAt(eye, center, up);

    //Initializing Entities
    const triangle = try Shape.Triangle.init(alloc, null);

    //Adding Entities to Scene
    try scene.addObject("triangle", triangle);

    //Debugging Entities
    triangle.debug();

    //Render loop
    while (c.glfwWindowShouldClose(window) == 0) {
        scene.render(window, view, projection);
    }
}
