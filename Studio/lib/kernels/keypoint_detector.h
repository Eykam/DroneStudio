#ifndef KEYPOINT_DETECTOR_H
#define KEYPOINT_DETECTOR_H

#include <stdint.h>
#include <stdbool.h> 

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include <cuda_gl_interop.h>
#else
    // Forward declare CUDA types when not compiling with CUDA
    typedef struct CUgraphicsResource_st *cudaGraphicsResource_t;
    typedef struct cudaArray *cudaArray_t;
    typedef struct CUsurfObject_st *cudaSurfaceObject_t;
#endif

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
    int gl_depthtexture;
    float world_transform[16];
    float* d_world_transform;
    CudaGLResources gl_resources;
    CudaGLTextureResource* y_texture;
    CudaGLTextureResource* uv_texture;
    CudaGLTextureResource* depth_texture;
    BRIEFDescriptor* d_descriptors;
    int id;
    bool initialized;
} DetectorInstance;


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


// Initialize CUDA resources
int cuda_create_detector(int max_keypoints, int gl_ytexture, int gl_uvtexture, int gl_depthtexture);
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
    StereoParams params,  
    ImageParams* image
);

int cuda_match_keypoints(
    int detector_id_left,
    int detector_id_right,
    int detector_id_combined,
    StereoParams params,  
    int* num_matches,

    ImageParams* left,
    ImageParams* right
);


#ifdef __cplusplus
}
#endif

#endif // KEYPOINT_DETECTOR_H