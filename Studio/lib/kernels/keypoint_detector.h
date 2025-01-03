#ifndef KEYPOINT_DETECTOR_H
#define KEYPOINT_DETECTOR_H

#include <stdint.h>

struct CUDAGLResources {
    cudaGraphicsResource_t position_resource;
    cudaGraphicsResource_t color_resource;
    float4* d_positions;
    float4* d_colors;
    size_t buffer_size;
};

#ifdef __cplusplus
extern "C" {
#endif

// Match your Zig struct
typedef struct {
    float x;
    float y;
} KeyPoint;

typedef struct {
    int* d_keypoint_count;
    CUDAGLResources gl_resources;
    int id;
    bool initialized;
} DetectorInstance;


typedef struct {
    const uint8_t* y_plane;
    int width;
    int height;
    int y_linesize;
    int* num_keypoints;
    float image_width;
    float image_height;
} ImageParams;

// Initialize CUDA resources
int cuda_create_detector();
int cuda_register_gl_buffers(int detector_id, unsigned int position_buffer, unsigned int color_buffer, int max_keypoints);
int cuda_detect_keypoints(
    int detector_id,
    uint8_t threshold,
    ImageParams* image
);
void cuda_unregister_gl_buffers(int detector_id);
void cuda_cleanup_detector(int detector_id);


int cuda_match_keypoints(
    int detector_id_left,
    int detector_id_right,
    float baseline,
    float focal_length,
    int* num_matches,
    uint8_t threshold,
    
    ImageParams* left,
    ImageParams* right
);


#ifdef __cplusplus
}
#endif

#endif // KEYPOINT_DETECTOR_H