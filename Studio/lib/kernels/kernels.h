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



typedef struct BRIEFDescriptor {
    uint64_t descriptor[8];  // 256-bit descriptor (can be adjusted)
} BRIEFDescriptor;

// Structure to hold matched keypoint data
typedef struct MatchedKeypoint {
    float3 left_pos;
    float3 right_pos;
    Match world;
    BRIEFDescriptor left_desc; 
} MatchedKeypoint;

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
    bool disable_depth;
    bool disable_spatial_tracking;
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
    uint left_count,
    const float4* right_positions,
    const BRIEFDescriptor* right_descriptors,
    uint right_count,
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
    uint left_count,
    uint right_count,
    const StereoParams params,
    MatchedKeypoint* matched_pairs,
    uint* out_count,
    dim3 grid,
    dim3 block
);

// Launch visualization kernel
void launch_visualization(
    const MatchedKeypoint* matches,
    uint match_count,
    float4* keypoint_positions,
    float4* keypoint_colors,
    float4* left_line_positions,
    float4* left_line_colors,
    float4* right_line_positions,
    float4* right_line_colors,
    const float* left_transform,
    const float* right_transform,
    bool show_connections,
    bool disable_depth,
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
    cudaSurfaceObject_t src_depth,
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
    uint num_matches,
    int width,
    int height,
    dim3 grid,
    dim3 block
);

// =================================================================== Visual Odometry =================================================================

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
    float max_pixel_distance;
    float min_confidence;         // Minimum confidence threshold
    uint min_matches;             // Minimum required matches
    float ransac_threshold;      // RANSAC inlier threshold
    uint ransac_iterations;       // Number of RANSAC iterations
    float spatial_weight;    // Weight for spatial distance term
    float hamming_weight;
    float img_weight;  
    float max_hamming_dist;
    float cost_threshold;
    float lowes_ratio;

} TemporalParams;

void launch_temporal_match_current_to_prev(
    const MatchedKeypoint* curr_matches,
    uint curr_match_count,
    const MatchedKeypoint* prev_matches, 
    uint prev_match_count,
    BestMatch* curr_to_prev_matches,
    TemporalParams params,
    dim3 grid,
    dim3 block
);
void launch_temporal_match_prev_to_current(
    const MatchedKeypoint* prev_matches,
    uint prev_match_count,
    const MatchedKeypoint* curr_matches,
    uint curr_match_count,
    BestMatch* prev_to_curr_matches,
    TemporalParams params,
    dim3 grid,
    dim3 block
);
void launch_temporal_cross_check(
    const BestMatch* curr_to_prev_matches,
    const BestMatch* prev_to_curr_matches,
    const MatchedKeypoint* curr_matches,
    const MatchedKeypoint* prev_matches,
    uint curr_match_count,
    uint prev_match_count,
    TemporalMatch* temporal_matches,
    uint* temporal_match_count,
    TemporalParams params,
    dim3 grid,
    dim3 block
);

void launch_motion_estimation(
    const TemporalMatch* d_matches,
    uint match_count,
    CameraPose* d_best_pose,
    uint* d_inlier_count,
    TemporalParams* params,
    dim3 grid,
    dim3 block
);

void launch_temporal_visualization(
    const TemporalMatch* matches,
    uint match_count,
    float4* keypoint_positions,
    float4* keypoint_colors,
    float4* connection_positions,
    float4* connection_colors,
    const float* prev_transform,
    const float* curr_transform,
    bool disable_depth,
    dim3 grid,
    dim3 block
);



#ifdef __cplusplus
}
#endif

#endif