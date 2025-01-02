#ifndef CUDA_KEYPOINT_DETECTOR_H
#define CUDA_KEYPOINT_DETECTOR_H

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

// Initialize CUDA resources
int cuda_init_detector();
int cuda_register_gl_buffers(unsigned int position_buffer, unsigned int color_buffer, int max_keypoints);
int cuda_detect_keypoints_gl(
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    int width,
    int height,
    int y_linesize,
    int uv_linesize,
    uint8_t threshold,
    int* num_keypoints,
    float image_width,
    float image_height
);
void cuda_unregister_gl_buffers(void);
void cuda_cleanup_detector(void);


#ifdef __cplusplus
}
#endif

#endif // CUDA_KEYPOINT_DETECTOR_H