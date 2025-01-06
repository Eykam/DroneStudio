#ifndef KEYPOINT_DETECTOR_H
#define KEYPOINT_DETECTOR_H

#include <stdint.h>


struct BRIEFDescriptor {
    uint64_t descriptor[4];  // 256-bit descriptor (can be adjusted)
};


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

typedef struct {
    float x;
    float y;
} KeyPoint;

typedef struct {
    int* d_keypoint_count;
    CUDAGLResources gl_resources;
    BRIEFDescriptor* d_descriptors;
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
int cuda_create_detector(int max_keypoints);
int cuda_register_gl_buffers(int detector_id, unsigned int position_buffer, unsigned int color_buffer, int max_keypoints);
float cuda_detect_keypoints(
    int detector_id,
    uint8_t threshold,
    ImageParams* image
);
void cuda_unregister_gl_buffers(int detector_id);
void cuda_cleanup_detector(int detector_id);


int cuda_match_keypoints(
    int detector_id_left,
    int detector_id_right,
    int detector_id_combined,
    float baseline,
    float focal_length,
    int* num_matches,
    uint8_t threshold,
    
    ImageParams* left,
    ImageParams* right
);

int cuda_map_gl_resources(int detector_id);
void cuda_unmap_gl_resources(int detector_id);


#ifdef __cplusplus
}
#endif

#endif // KEYPOINT_DETECTOR_H