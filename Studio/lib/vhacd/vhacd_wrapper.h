#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void* VHACDHandle;

// Fill mode options
typedef enum {
    VHACD_FILL_FLOOD_FILL = 0,   // Default behavior, uses flood fill
    VHACD_FILL_SURFACE_ONLY = 1, // Only consider the surface, creates hollow centers
    VHACD_FILL_RAYCAST_FILL = 2  // Uses raycasting to determine inside from outside
} VHACDFillMode;

// Create/destroy V-HACD instance
VHACDHandle vhacd_create(void);
VHACDHandle vhacd_create_async(void);  // Asynchronous version
void vhacd_release(VHACDHandle handle);

// Parameter setters (all optional - uses defaults if not called)
void vhacd_set_max_convex_hulls(VHACDHandle handle, unsigned int max_convex_hulls);
void vhacd_set_resolution(VHACDHandle handle, unsigned int resolution);
void vhacd_set_minimum_volume_percent_error_allowed(VHACDHandle handle, double min_volume_percent);
void vhacd_set_max_recursion_depth(VHACDHandle handle, unsigned int max_depth);
void vhacd_set_shrink_wrap(VHACDHandle handle, int enable_shrink_wrap);
void vhacd_set_fill_mode(VHACDHandle handle, VHACDFillMode fill_mode);
void vhacd_set_max_num_vertices_per_ch(VHACDHandle handle, unsigned int max_vertices);
void vhacd_set_async_acd(VHACDHandle handle, int enable_async);
void vhacd_set_min_edge_length(VHACDHandle handle, unsigned int min_edge_length);
void vhacd_set_find_best_plane(VHACDHandle handle, int enable_best_plane);

// Process mesh (returns 1 on success, 0 on failure)
int vhacd_compute_float(VHACDHandle handle, 
                        const float* points, 
                        unsigned int count_points,
                        const unsigned int* triangles, 
                        unsigned int count_triangles);

int vhacd_compute_double(VHACDHandle handle, 
                         const double* points, 
                         unsigned int count_points,
                         const unsigned int* triangles, 
                         unsigned int count_triangles);

// Get results
unsigned int vhacd_get_n_convex_hulls(VHACDHandle handle);

// Get convex hull data (returns 1 on success, 0 on failure)
// Note: The returned pointers are valid until the next call to vhacd_clean() or vhacd_release()
int vhacd_get_convex_hull(VHACDHandle handle, 
                          unsigned int index,
                          double** out_points,           // Array of doubles: x1,y1,z1,x2,y2,z2,...
                          unsigned int* out_n_points,
                          unsigned int** out_triangles,  // Array of triangle indices: i1,i2,i3,i4,i5,i6,...
                          unsigned int* out_n_triangles);

// Get additional hull properties
int vhacd_get_convex_hull_volume(VHACDHandle handle, unsigned int index, double* volume);
int vhacd_get_convex_hull_center(VHACDHandle handle, unsigned int index, double center[3]);
int vhacd_get_convex_hull_bounds(VHACDHandle handle, unsigned int index, 
                                 double min_bounds[3], double max_bounds[3]);

// Optional but useful functions
int vhacd_is_ready(VHACDHandle handle);  // Returns 1 if ready, 0 if still processing (async mode)
void vhacd_cancel(VHACDHandle handle);   // Cancel processing early
int vhacd_compute_center_of_mass(VHACDHandle handle, double center_of_mass[3]); // Returns 1 on success

// Find which convex hull is closest to a given position
unsigned int vhacd_find_nearest_convex_hull(VHACDHandle handle, 
                                           const double pos[3], 
                                           double* distance_to_hull);

// Memory management
void vhacd_clean(VHACDHandle handle);  // Free internal memory but keep instance

#ifdef __cplusplus
}
#endif