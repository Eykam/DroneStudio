#ifndef KEYPOINT_DETECTOR_H
#define KEYPOINT_DETECTOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KeyPoint {
    float x;
    float y;
} KeyPoint;

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

typedef struct InstanceBuffer {
    cudaGraphicsResource_t position_resource;
    cudaGraphicsResource_t color_resource;
    float4* d_positions;
    float4* d_colors;
    size_t buffer_size;
} InstanceBuffer;

typedef struct ConnectedKeypoint {
    InstanceBuffer left;
    InstanceBuffer right;
} ConnectedKeypoint;

typedef struct CudaGLResources {
    ConnectedKeypoint connections;
    InstanceBuffer keypoints;
} CudaGLResources;

typedef struct CudaGLTextureResource {
    cudaGraphicsResource_t tex_resource;
    cudaArray_t array;
    cudaSurfaceObject_t surface;
} CudaGLTextureResource;

typedef struct DetectorInstance {
    int* d_keypoint_count;
    int gl_ytexture;
    int gl_uvtexture;
    alignas(16) float world_transform[16];; 
    float* d_world_transform;
    CudaGLResources gl_resources;
    CudaGLTextureResource* y_texture;
    CudaGLTextureResource* uv_texture;
    BRIEFDescriptor* d_descriptors;
    int id;
    bool initialized;
} DetectorInstance;



// Initialize CUDA resources
int cuda_create_detector(int max_keypoints, int gl_ytexture, int gl_uvtexture, const float transform[16]);
void cuda_cleanup_detector(int detector_id);

int cuda_register_gl_texture(int detector_id);
void cuda_unregister_gl_texture(int detector_id);

int cuda_register_buffers(
    int detector_id,
    int keypoint_position_buffer,
    int keypoint_color_buffer,
    int* left_position_buffer,
    int* left_color_buffer,
    int* right_position_buffer,
    int* right_color_buffer,
    int buffer_size
);
void cuda_unregister_buffers(int detector_id);

int cuda_map_resources(int detector_id);
void cuda_unmap_resources(int detector_id);

int cuda_map_transformation(int detector_id,  const float transformation[16]);

float cuda_detect_keypoints(
    int detector_id,
    uint8_t threshold,
    ImageParams* image,
    float sigma
);

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


#ifdef __cplusplus
}
#endif

#endif // KEYPOINT_DETECTOR_H