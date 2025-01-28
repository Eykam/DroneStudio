#ifndef STEREO_MATCHING_KERNELS_H
#define STEREO_MATCHING_KERNELS_H

#include <sys/types.h>
#include <stdint.h>
#include <stdbool.h> 
#include <stddef.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#ifdef __cplusplus
extern "C" {
#endif


typedef struct BestMatch {
    int   bestIdx;   
    float bestCost;   
    //float secondBestCost; // for ratio tests
} BestMatch;

typedef struct Match {
    float3 position;   // Position in OpenGL world coordinates
    float2 image_coords;
    float disparity;    // Pixel disparity between left and right views
    float depth;
} Match;

// Structure to hold matched keypoint data
typedef struct MatchedKeypoint {
    float3 left_pos;
    float3 right_pos;
    Match world;
} MatchedKeypoint;


typedef struct BRIEFDescriptor {
    uint64_t descriptor[8];  // 256-bit descriptor (can be adjusted)
} BRIEFDescriptor;

typedef struct ImageParams {
    const uint8_t* y_plane;
    const uint8_t* uv_plane;
    int width;
    int height;
    int y_linesize;
    int uv_linesize;
    int* num_keypoints;
    float image_width;
    float image_height;
} ImageParams;


typedef struct StereoParams {
    int image_width;
    int image_height;
    float baseline_mm;
    float focal_length_mm;
    float focal_length_px;
    float sensor_width_mm;
    uint8_t intensity_threshold;
    uint32_t circle_radius;
    uint32_t arc_length;
    uint32_t max_keypoints;
    float sigma;
    float max_disparity;
    float epipolar_threshold;
    float max_hamming_dist;
    float lowes_ratio;
    float cost_threshold;
    float epipolar_weight;
    float disparity_weight;
    float hamming_dist_weight;
    bool show_connections;
    bool disable_matching;
} StereoParams;

// Initialize Gaussian kernel in constant memory
void init_gaussian_kernel(float sigma);

// Launch Gaussian blur kernels
void launch_gaussian_blur(
    const uint8_t* input,
    uint8_t* temp1,
    uint8_t* temp2,
    int width,
    int height,
    int pitch,
    dim3 grid,
    dim3 block
);

// Launch keypoint detection kernel
void launch_keypoint_detection(
    const uint8_t* input,
    int width,
    int height,
    int pitch,
    uint8_t threshold,
    uint32_t arc_length,
    float4* positions,
    float4* colors,
    BRIEFDescriptor* descriptors,
    uint* keypoint_count,
    int max_keypoints,
    dim3 grid,
    dim3 block
);

// Launch stereo matching kernels
void launch_stereo_matching(
    const float4* left_positions,
    const BRIEFDescriptor* left_descriptors,
    int left_count,
    const float4* right_positions,
    const BRIEFDescriptor* right_descriptors,
    int right_count,
    StereoParams params,
    BestMatch* matches_left,
    BestMatch* matches_right
);

// Launch cross-check matches kernel
void launch_cross_check_matches(
    const BestMatch* matches_left,
    const BestMatch* matches_right,
    const float4* left_positions,
    const float4* right_positions,
    int left_count,
    int right_count,
    const StereoParams* params,
    MatchedKeypoint* matched_pairs,
    int* out_count,
    dim3 grid,
    dim3 block
);

// Launch visualization kernel
void launch_visualization(
    const MatchedKeypoint* matches,
    int match_count,
    float4* keypoint_positions,
    float4* keypoint_colors,
    float4* left_line_positions,
    float4* left_line_colors,
    float4* right_line_positions,
    float4* right_line_colors,
    const float* left_transform,
    const float* right_transform,
    bool show_connections,
    dim3 grid,
    dim3 block
);

// Launch texture update kernels
void launch_texture_update(
    cudaSurfaceObject_t y_surface,
    cudaSurfaceObject_t uv_surface,
    cudaSurfaceObject_t depth_surface,
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    int width,
    int height,
    int y_linesize,
    int uv_linesize,
    dim3 grid,
    dim3 block
);

void launch_surface_copy(
    cudaSurfaceObject_t src_y,
    cudaSurfaceObject_t src_uv,
    cudaSurfaceObject_t dst_y,
    cudaSurfaceObject_t dst_uv,
    cudaSurfaceObject_t dst_depth,
    int width,
    int height,
    dim3 grid,
    dim3 block
);

void launch_depth_texture_update(
    cudaSurfaceObject_t depth_surface,
    MatchedKeypoint* matches,
    int num_matches,
    int width,
    int height,
    dim3 grid,
    dim3 block
);



typedef struct CameraPose {
    float rotation[9];     // 3x3 rotation matrix
    float translation[3];  // Translation vector
} CameraPose;

typedef struct TemporalMatch {
    float3 prev_pos;      // 3D position in previous frame
    float3 current_pos;   // 3D position in current frame
    float confidence;     // Match confidence score
} TemporalMatch;

typedef struct TemporalParams {
    float max_distance;           // Maximum distance for temporal matching
    float min_confidence;         // Minimum confidence threshold
    int min_matches;             // Minimum required matches
    float ransac_threshold;      // RANSAC inlier threshold
    int ransac_iterations;       // Number of RANSAC iterations
} TemporalParams;






#ifdef __cplusplus
}
#endif

#endif