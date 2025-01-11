#include "keypoint_detector.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>

#define MAX_DETECTORS 16

#define GAUSSIAN_KERNEL_RADIUS 2
#define GAUSSIAN_KERNEL_SIZE (2 * GAUSSIAN_KERNEL_RADIUS + 1)

// ============================================================= Detection =================================================================
// Generated BRIEF pattern (512 pairs):
__device__ int2 brief_pattern[1024] = {
    { -7, -1}, {  0, -2},{ -6, -1}, { -5, -6},{ -6, -1}, { -7, -3},{ -6, -1}, { -1,  2},{ -5, -1}, { -5, -5},{ -5, -1}, { -7, -4},{ -5, -1}, { -3,  1},{ -5, -1}, { -7, -4},{ -5, -1}, { -7, -2},{ -4, -1}, { -7,  0},{ -4, -1}, { -7,  2},{ -4, -1}, { -5, -7},{ -7, -2}, { -4, -6},{ -7, -2}, { -7, -4},{ -3, -1}, { -4,  7},{ -3, -1}, { -3,  5},{ -6, -2}, { -1, -6},{ -6, -2}, { -7, -4},{ -3, -1}, { -3,  2},{ -3, -1}, { -5,  3},{ -3, -1}, { -4, -3},{ -5, -2}, { -7, -1},{ -2, -1}, { -2,  2},{ -6, -3}, { -7, -5},{ -2, -1}, {  1, -1},{ -2, -1}, { -5, -4},{ -2, -1}, { -4, -7},{ -4, -2}, { -5, -6},{ -2, -1}, { -1, -6},{ -2, -1}, {  0,  2},{ -2, -1}, { -7,  2},{ -2, -1}, {  0,  7},{ -4, -2}, { -4,  4},{ -3, -2}, { -7, -3},{ -3, -2}, { -5, -5},{ -3, -2}, { -3,  1},{ -6, -4}, { -4, -7},{ -4, -3}, {  0, -7},{ -4, -3}, { -7, -1},{ -5, -4}, { -3, -7},{ -5, -4}, { -2, -7},{ -5, -4}, { -6, -7},{ -5, -4}, { -4, -7},{ -5, -4}, { -7, -4},{ -1, -1}, { -1, -1},{ -2, -2}, {  3,  5},{ -2, -2}, {  0,  3},{ -1, -1}, { -1, -3},{ -4, -4}, { -4, -7},{ -3, -3}, { -6, -7},{ -1, -1}, {  2, -4},{ -2, -2}, { -2,  2},{ -4, -4}, { -3, -7},{ -5, -5}, { -4, -7},{ -3, -3}, {  0, -1},{ -1, -1}, { -3, -2},{ -3, -3}, { -5,  6},{ -1, -1}, { -3,  5},{ -2, -2}, { -2,  6},{ -3, -3}, {  1, -2},{ -1, -1}, {  3,  1},{ -1, -1}, {  0,  2},{ -5, -6}, { -7, -7},{ -4, -5}, { -6, -1},{ -4, -5}, {  0,  4},
    { -3, -4}, { -7, -4},{ -3, -5}, { -1, -5},{ -3, -5}, { -4, -4},{ -3, -5}, {  3, -7},{ -3, -5}, {  0, -7},{ -1, -2}, {  2,  1},{ -1, -2}, { -3, -3},{ -2, -4}, { -2, -2},{ -3, -6}, {  1, -7},{ -1, -2}, {  3, -3},{ -1, -2}, { -5, -7},{ -3, -6}, { -1, -3},{ -1, -2}, {  0, -3},{ -2, -4}, { -3, -1},{ -3, -6}, { -2, -7},{ -2, -5}, { -1, -6},{ -2, -5}, { -4, -7},{ -2, -6}, { -3, -4},{ -2, -6}, { -7, -7},{ -1, -3}, { -3, -3},{ -1, -3}, { -5,  3},{ -1, -3}, {  3, -1},{ -2, -7}, { -6, -4},{ -1, -4}, {  1, -1},{ -1, -4}, {  3, -7},{ -1, -4}, {  4, -5},{ -1, -7}, { -7, -7},{  0, -1}, {  2,  2},{  0, -1}, {  0, -3},{  0, -1}, {  0,  3},{  0, -5}, {  2, -6},{  0, -1}, {  3, -5},{  0, -2}, { -4,  5},{  0, -1}, { -6, -1},{  0, -4}, {  1, -2},{  0, -2}, { -1, -6},{  0, -1}, { -2, -2},{  0, -5}, {  1, -6},{  0, -5}, {  4, -6},{  0, -1}, { -7, -4},{  0, -3}, { -6, -6},{  0, -3}, {  1, -7},{  0, -5}, { -1, -7},{  0, -1}, {  2, -3},{  0, -1}, {  0, -1},{  0, -1}, { -6,  0},{  0, -5}, { -3, -2},{  0, -4}, { -2,  7},{  0, -1}, { -5,  2},{  0, -1}, {  1, -6},{  0, -2}, {  5, -2},{  0, -2}, {  1, -3},{  0, -3}, {  1, -7},{  1, -6}, {  1, -5},{  1, -6}, {  4, -4},{  1, -6}, {  2, -7},{  1, -6}, { -4, -7},{  1, -5}, {  1, -7},{  1, -5}, { -4, -2},{  1, -4}, {  4, -6},{  2, -7}, { -2, -2},{  1, -3}, {  2, -5},{  2, -6}, {  7, -3},{  1, -3}, {  3, -6},{  1, -3}, {  0, -6},
    {  1, -3}, {  1, -7},{  3, -7}, {  4, -7},{  3, -7}, {  7, -7},{  3, -6}, {  3, -5},{  1, -2}, { -2, -1},{  1, -2}, { -5,  0},{  3, -6}, {  7,  0},{  2, -4}, {  1, -7},{  1, -2}, {  2, -3},{  1, -2}, {  4, -2},{  1, -2}, {  2,  2},{  1, -2}, { -2, -4},{  2, -4}, {  3, -5},{  2, -4}, {  7, -1},{  2, -4}, { -4, -7},{  1, -2}, { -5, -3},{  1, -2}, { -3,  2},{  1, -2}, {  1, -5},{  4, -6}, {  0, -7},{  2, -3}, { -5,  0},{  4, -6}, {  2, -6},{  2, -3}, {  0,  3},{  3, -4}, {  0,  1},{  3, -4}, { -3, -7},{  3, -4}, {  2, -6},{  4, -5}, {  6, -7},{  4, -5}, {  5, -7},{  4, -5}, {  3,  2},{  4, -5}, {  2, -3},{  5, -6}, {  3, -7},{  5, -5}, {  4, -3},{  2, -2}, {  5, -3},{  3, -3}, {  6, -6},{  1, -1}, {  7, -1},{  4, -4}, {  2, -7},{  3, -3}, {  7, -7},{  4, -4}, {  6, -4},{  3, -3}, {  1, -6},{  5, -5}, {  5, -6},{  4, -4}, {  7, -4},{  1, -1}, {  5,  1},{  3, -3}, {  5,  3},{  5, -4}, {  7, -3},{  5, -4}, {  1,  0},{  4, -3}, {  5, -4},{  3, -2}, {  4, -3},{  6, -4}, {  4, -4},{  3, -2}, {  4, -5},{  3, -2}, { -2, -4},{  3, -2}, { -4, -3},{  3, -2}, {  5, -1},{  5, -3}, {  4,  1},{  5, -3}, { -4, -7},{  2, -1}, { -1, -5},{  2, -1}, { -5,  2},{  2, -1}, {  4, -5},{  4, -2}, { -1,  0},{  2, -1}, {  5,  2},{  4, -2}, {  4, -1},{  2, -1}, {  7, -5},{  4, -2}, {  3, -4},{  4, -2}, { -3,  3},{  4, -2}, {  5, -3},{  7, -3}, {  7, -5},{  5, -2}, {  6, -1},{  5, -2}, {  6,  0},
    {  5, -2}, {  6, -7},{  5, -2}, {  1,  0},{  5, -2}, {  1,  3},{  3, -1}, {  5, -3},{  6, -2}, {  7,  0},{  6, -2}, {  1,  1},{  3, -1}, { -4, -1},{  3, -1}, {  4,  3},{  7, -2}, {  7, -5},{  4, -1}, {  1,  2},{  5, -1}, {  4,  6},{  5, -1}, {  7,  3},{  6, -1}, {  7, -7},{  6, -1}, {  7,  0},{  6, -1}, {  7, -6},{  0,  0}, { -3, -4},{  1,  0}, {  5,  3},{  0,  0}, { -2, -4},{  0,  0}, {  7,  1},{  0,  0}, {  1,  7},{  3,  0}, {  2,  7},{  0,  0}, { -7,  0},{  1,  0}, {  3,  1},{  5,  0}, {  6, -3},{  3,  0}, {  7,  4},{  0,  0}, { -4,  7},{  0,  0}, { -6, -1},{  1,  0}, {  0,  0},{  5,  0}, { -3, -5},{  3,  0}, { -1,  5},{  0,  0}, {  3,  3},{  0,  0}, {  1,  3},{  0,  0}, {  4,  4},{  1,  0}, {  2, -3},{  1,  0}, { -2, -1},{  0,  0}, {  2,  1},{  0,  0}, { -3, -2},{  0,  0}, { -1,  3},{  1,  0}, {  4, -4},{  7,  0}, {  6, -3},{  0,  0}, { -5,  1},{  1,  0}, { -1, -2},{  0,  0}, {  7, -6},{  0,  0}, { -1,  4},{  0,  0}, { -7,  0},{  0,  0}, { -5,  2},{  1,  0}, {  0,  4},{  1,  0}, {  4,  3},{  0,  0}, { -3, -4},{  2,  0}, { -2, -1},{  3,  0}, {  6, -4},{  1,  0}, {  4,  4},{  0,  0}, {  0,  2},{  0,  0}, {  7, -4},{  1,  0}, { -2,  6},{  0,  0}, { -1,  3},{  0,  0}, { -2,  4},{  0,  0}, {  7,  3},{  1,  0}, {  7,  5},{  0,  0}, { -1, -3},{  0,  0}, {  5,  4},{  3,  0}, {  7,  1},{  2,  0}, {  1,  1},{  0,  0}, {  0,  2},{  2,  0}, {  3, -3},{  0,  0}, { -7, -1},
    {  0,  0}, { -4,  5},{  3,  0}, {  1,  7},{  0,  0}, { -5,  5},{  0,  0}, { -3, -4},{  1,  0}, { -3,  2},{  2,  0}, {  6, -2},{  1,  0}, {  1, -3},{  7,  0}, {  4,  3},{  0,  0}, {  7,  7},{  2,  0}, { -2, -1},{  0,  0}, { -5,  0},{  0,  0}, {  2, -6},{  0,  0}, {  7,  5},{  2,  0}, {  5,  0},{  1,  0}, { -4, -3},{  0,  0}, { -2,  2},{  0,  0}, {  6, -6},{  6,  1}, {  7,  4},{  5,  1}, {  5,  4},{  5,  1}, {  1,  4},{  5,  1}, {  7,  2},{  5,  1}, {  7, -2},{  5,  1}, {  1,  4},{  5,  1}, { -5,  3},{  4,  1}, {  7,  1},{  4,  1}, {  5, -2},{  7,  2}, {  7, -5},{  3,  1}, {  4,  5},{  3,  1}, {  0,  6},{  3,  1}, {  4, -1},{  6,  2}, {  6, -4},{  3,  1}, {  5, -2},{  5,  2}, {  5,  5},{  5,  2}, {  1,  5},{  5,  2}, {  3, -3},{  5,  2}, {  7,  4},{  6,  3}, {  6,  7},{  2,  1}, {  7,  2},{  2,  1}, {  3,  4},{  6,  3}, {  7,  5},{  2,  1}, { -4,  4},{  2,  1}, { -5, -1},{  6,  3}, {  7,  2},{  6,  3}, {  7,  3},{  5,  3}, {  3,  7},{  5,  3}, {  7,  7},{  5,  3}, { -1, -1},{  3,  2}, {  1,  5},{  3,  2}, { -2,  4},{  3,  2}, {  7, -1},{  4,  3}, {  2, -4},{  4,  3}, {  5,  3},{  4,  3}, {  7,  2},{  4,  3}, {  4,  3},{  5,  4}, {  7, -1},{  1,  1}, {  5, -1},{  1,  1}, { -2,  7},{  1,  1}, {  2, -7},{  5,  5}, {  7,  5},{  3,  3}, {  4,  7},{  1,  1}, {  2,  3},{  2,  2}, {  0, -3},{  1,  1}, {  5, -3},{  1,  1}, { -7, -1},
    {  2,  2}, {  4,  2},{  1,  1}, {  1,  3},{  2,  2}, {  3,  1},{  2,  2}, { -1,  6},{  2,  2}, {  1,  1},{  4,  4}, {  2,  6},{  1,  1}, {  1, -3},{  3,  3}, {  5,  0},{  1,  1}, { -2,  6},{  5,  5}, {  7,  5},{  1,  1}, { -1, -2},{  1,  1}, {  0,  4},{  2,  2}, {  5,  0},{  4,  4}, {  5,  0},{  3,  3}, {  3,  6},{  4,  5}, {  4,  4},{  3,  4}, {  4,  5},{  3,  4}, { -7,  6},{  2,  3}, {  7, -1},{  2,  3}, {  0,  3},{  2,  3}, { -1,  2},{  4,  6}, {  1,  1},{  2,  3}, { -2,  7},{  3,  5}, {  7,  7},{  3,  5}, {  2,  6},{  3,  5}, {  7, -1},{  3,  6}, { -2,  3},{  1,  2}, {  0, -1},{  2,  4}, { -2,  0},{  3,  6}, {  5,  4},{  2,  5}, {  0,  3},{  2,  5}, {  1,  7},{  1,  3}, {  5,  3},{  1,  3}, { -5,  2},{  1,  3}, { -7,  4},{  1,  3}, { -6,  7},{  2,  7}, {  1,  7},{  1,  4}, {  0,  5},{  1,  4}, { -1, -2},{  1,  4}, {  0,  4},{  1,  5}, {  1,  6},{  1,  5}, {  7,  7},{  1,  5}, { -2,  7},{  1,  5}, {  2,  7},{  1,  6}, {  3,  6},{  1,  6}, {  3,  7},{  0,  1}, {  2,  3},{  0,  2}, {  2,  7},{  0,  4}, { -7,  0},{  0,  7}, { -4,  4},{  0,  1}, { -1,  7},{  0,  5}, {  0,  0},{  0,  4}, {  0,  4},{  0,  4}, {  0,  7},{  0,  1}, {  0,  4},{  0,  7}, { -5,  7},{  0,  1}, { -1, -2},{  0,  1}, {  0, -4},{  0,  2}, {  7,  2},{  0,  4}, { -4,  2},{  0,  1}, {  7,  2},{  0,  2}, {  2,  0},{  0,  3}, { -2,  6},{  0,  3}, {  0,  1},{  0,  6}, { -4,  4},
    {  0,  1}, {  3, -4},{  0,  1}, {  2, -3},{  0,  4}, {  0,  2},{  0,  1}, {  4,  6},{  0,  2}, {  3,  4},{  0,  1}, { -3,  6},{  0,  3}, { -2,  3},{  0,  1}, { -3,  2},{  0,  2}, {  0, -7},{ -1,  7}, {  2,  7},{ -1,  7}, {  7,  6},{ -1,  6}, {  1,  5},{ -1,  5}, {  7,  6},{ -1,  5}, { -3,  1},{ -1,  4}, {  1,  4},{ -1,  4}, {  1,  0},{ -2,  7}, {  3,  6},{ -1,  3}, {  5,  3},{ -1,  3}, {  2,  6},{ -1,  3}, { -5,  7},{ -1,  3}, { -3, -1},{ -1,  3}, { -3,  6},{ -1,  3}, { -1, -1},{ -1,  3}, { -4,  6},{ -1,  3}, {  1,  4},{ -2,  5}, { -5,  7},{ -2,  5}, { -6,  2},{ -2,  5}, { -3,  7},{ -2,  5}, { -7,  7},{ -3,  7}, { -7,  3},{ -3,  7}, { -7,  7},{ -3,  6}, { -5,  3},{ -2,  4}, { -5,  7},{ -2,  4}, {  1, -2},{ -1,  2}, { -7, -5},{ -1,  2}, {  0,  3},{ -1,  2}, {  5,  2},{ -2,  4}, {  3,  7},{ -3,  6}, { -4,  7},{ -2,  4}, {  2,  7},{ -1,  2}, {  1,  7},{ -1,  2}, {  5, -1},{ -1,  2}, { -2, -1},{ -1,  2}, { -2,  7},{ -3,  5}, { -4,  2},{ -3,  5}, {  3,  3},{ -4,  6}, { -3,  7},{ -2,  3}, {  0,  1},{ -4,  6}, { -7,  1},{ -4,  6}, { -5, -1},{ -3,  4}, { -5,  6},{ -3,  4}, { -5,  3},{ -4,  5}, {  6,  1},{ -1,  1}, {  4,  1},{ -4,  4}, { -4, -3},{ -2,  2}, { -5,  2},{ -3,  3}, {  1,  1},{ -2,  2}, { -1,  4},{ -1,  1}, { -2, -3},{ -1,  1}, { -2,  5},{ -2,  2}, { -1,  4},{ -1,  1}, {  4, -3},{ -2,  2}, { -7,  0},{ -1,  1}, { -4, -1},{ -1,  1}, { -5,  2},
    { -2,  2}, { -1, -4},{ -1,  1}, { -3,  1},{ -5,  4}, { -7, -2},{ -5,  4}, { -1,  3},{ -4,  3}, {  1,  2},{ -3,  2}, {  7, -1},{ -3,  2}, { -2,  5},{ -3,  2}, { -2,  2},{ -3,  2}, { -5, -2},{ -3,  2}, { -4,  0},{ -5,  3}, { -2,  4},{ -2,  1}, { -6,  3},{ -4,  2}, {  0,  2},{ -4,  2}, { -7,  2},{ -5,  2}, { -1,  5},{ -3,  1}, { -3,  2},{ -6,  2}, { -7, -6},{ -3,  1}, {  4, -1},{ -6,  2}, { -7, -3},{ -3,  1}, { -7, -7},{ -6,  2}, { -1,  5},{ -6,  2}, { -7,  2},{ -7,  2}, { -7,  5},{ -4,  1}, { -7,  0},{ -4,  1}, { -7,  4},{ -4,  1}, { -5, -3},{ -4,  1}, { -3,  1},{ -6,  1}, { -4, -5},{ -7,  1}, { -5, -2},{ -7,  1}, { -7,  3},{ -7,  1}, { -4,  7},{ -3,  0}, { -1,  2},{ -1,  0}, { -5,  2},{ -1,  0}, { -4,  1},{ -1,  0}, {  7,  0},{ -1,  0}, { -2,  5},{ -2,  0}, { -1, -1},{ -4,  0}, { -7,  2},{ -7,  0}, { -4, -4},{ -7,  0}, { -7,  0},{ -3,  0}, {  2,  0},{ -1,  0}, { -3, -1},{ -6,  0}, { -7, -4},{ -7,  0}, { -3,  1},{ -1,  0}, { -7, -3},{ -1,  0}, {  5,  3},{ -6,  0}, { -3,  3},{ -1,  0}, {  2,  3},{ -6,  0}, { -6, -6},{ -4,  0}, {  1,  0},{ -2,  0}, { -4,  0},{ -6,  0}, { -7,  0},{ -1,  0}, { -2,  0},{ -5,  0}, { -7,  5},{ -3,  0}, {  2, -4},{ -1,  0}, { -7, -1}
};

// FAST circle offsets
__constant__ int2 fast_offsets[16] = {
    // Right side
    {3,  0}, {3,  1}, {3, -1},
    // Top right
    {2,  2}, {1,  3},
    // Top
    {0,  3}, {-1, 3},
    // Top left
    {-2, 2}, {-3, 1},
    // Left
    {-3, 0}, {-3,-1},
    // Bottom left
    {-2,-2}, {-1,-3},
    // Bottom
    {0, -3}, {1, -3},
    // Bottom right
    {2, -2}
};


// Global state
static DetectorInstance g_detectors[MAX_DETECTORS] = {0};
static int g_next_detector_id = 0;

static DetectorInstance* get_detector_instance(int id) {
    for (int i = 0; i < MAX_DETECTORS; i++) {
        if (g_detectors[i].initialized && g_detectors[i].id == id) {
            return &g_detectors[i];
        }
    }
    return NULL;
}

static int find_free_detector_slot(void) {
    for (int i = 0; i < MAX_DETECTORS; i++) {
        if (!g_detectors[i].initialized) {
            return i;
        }
    }
    return -1;
}


__constant__ float d_gaussian_kernel[GAUSSIAN_KERNEL_SIZE];


// Separable Gaussian blur kernels
__global__ void gaussianBlurHorizontal(
    const uint8_t* __restrict__ input,
    uint8_t* output,
    int width,
    int height,
    int pitch
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    
    // Horizontal pass
    for (int i = -GAUSSIAN_KERNEL_RADIUS; i <= GAUSSIAN_KERNEL_RADIUS; i++) {
        int cur_x = min(max(x + i, 0), width - 1);
        sum += input[y * pitch + cur_x] * d_gaussian_kernel[i + GAUSSIAN_KERNEL_RADIUS];
    }
    
    output[y * pitch + x] = (uint8_t)sum;
}

__global__ void gaussianBlurVertical(
    const uint8_t* __restrict__ input,
    uint8_t* output,
    int width,
    int height,
    int pitch
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    
    // Vertical pass
    for (int i = -GAUSSIAN_KERNEL_RADIUS; i <= GAUSSIAN_KERNEL_RADIUS; i++) {
        int cur_y = min(max(y + i, 0), height - 1);
        sum += input[cur_y * pitch + x] * d_gaussian_kernel[i + GAUSSIAN_KERNEL_RADIUS];
    }
    
    output[y * pitch + x] = (uint8_t)sum;
}

__device__ float3 convertImageToWorldCoords(float x, float y, float imageWidth, float imageHeight) {
    float normalizedX = (x / imageWidth) * 2.0f - 1.0f;
    float normalizedY = -((y / imageHeight) * 2.0f - 1.0f);
    
    float worldX = normalizedX * 6.4f;
    float worldY = normalizedY * 3.6f;
    
    return make_float3(worldX, -0.01f, worldY);
}

__global__ void detectFASTKeypoints(
    const uint8_t* __restrict__ y_plane,
    int width,
    int height,
    int linesize,
    uint8_t threshold,
    float4* positions,
    float4* colors,
    BRIEFDescriptor* descriptors,
    int* keypoint_count,
    int max_keypoints,
    float image_width,
    float image_height
) {
    __shared__ int block_counter;
    __shared__ float2 block_keypoints[256]; 
    __shared__ BRIEFDescriptor block_descriptors[256];
    
    // Initialize block counter
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        block_counter = 0;
    }
    __syncthreads();

    // Calculate global pixel coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Early exit for border pixels
    if (x < 3 || y < 3 || x >= width - 3 || y >= height - 3) {
        return;
    }

    const uint8_t center = y_plane[y * linesize + x];
    
    // Check for continuous segments
    int consecutive_brighter = 0;
    int consecutive_darker = 0;
    int max_consecutive_brighter = 0;
    int max_consecutive_darker = 0;
    
    // First, check pixels 1-16
    for (int i = 0; i < 16; i++) {
        const int2 offset = fast_offsets[i];
        const uint8_t pixel = y_plane[(y + offset.y) * linesize + (x + offset.x)];
        
        if (pixel > center + threshold) {
            consecutive_brighter++;
            consecutive_darker = 0;
            max_consecutive_brighter = max(max_consecutive_brighter, consecutive_brighter);
        }
        else if (pixel < center - threshold) {
            consecutive_darker++;
            consecutive_brighter = 0;
            max_consecutive_darker = max(max_consecutive_darker, consecutive_darker);
        }
        else {
            consecutive_brighter = 0;
            consecutive_darker = 0;
        }
    }
    
    // Then check pixels 1-3 again to handle wrapping
    for (int i = 0; i < 3; i++) {
        const int2 offset = fast_offsets[i];
        const uint8_t pixel = y_plane[(y + offset.y) * linesize + (x + offset.x)];
        
        if (pixel > center + threshold) {
            consecutive_brighter++;
            consecutive_darker = 0;
            max_consecutive_brighter = max(max_consecutive_brighter, consecutive_brighter);
        }
        else if (pixel < center - threshold) {
            consecutive_darker++;
            consecutive_brighter = 0;
            max_consecutive_darker = max(max_consecutive_darker, consecutive_darker);
        }
        else {
            consecutive_brighter = 0;
            consecutive_darker = 0;
        }
    }

    // Check if point is a corner
    bool is_keypoint = (max_consecutive_brighter >= 9 || max_consecutive_darker >= 9);

    // Non-maximum suppression
    if (is_keypoint) {
        // Simple score based on absolute difference from center
        BRIEFDescriptor desc = {0};  // Initialize all bits to 0
        
        #pragma unroll
        for (int i = 0; i < 512; i += 2) {
            int2 offset1 = brief_pattern[i];
            int2 offset2 = brief_pattern[i+1];
            
            uint8_t p1 = y_plane[(y + offset1.y) * linesize + (x + offset1.x)];
            uint8_t p2 = y_plane[(y + offset2.y) * linesize + (x + offset2.x)];
            
            // Set bit if first point is brighter than second
            if (p1 > p2) {
                int bitIdx = i / 2;
                desc.descriptor[bitIdx / 64] |= (1ULL << (bitIdx % 64));
            }
        }

        float score = 0.0f;
        for (int i = 0; i < 16; i++) {
            const int2 offset = fast_offsets[i];
            const uint8_t pixel = y_plane[(y + offset.y) * linesize + (x + offset.x)];
            score += abs(pixel - center);
        }
        
        // Check 3x3 neighborhood for non-maximum suppression
        bool is_local_maximum = true;
        for (int dy = -1; dy <= 1 && is_local_maximum; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                
                float neighbor_score = 0.0f;
                const int nx = x + dx;
                const int ny = y + dy;
                
                if (nx >= 3 && nx < width - 3 && ny >= 3 && ny < height - 3) {
                    const uint8_t neighbor_center = y_plane[ny * linesize + nx];
                    for (int i = 0; i < 16; i++) {
                        const int2 offset = fast_offsets[i];
                        const uint8_t pixel = y_plane[(ny + offset.y) * linesize + (nx + offset.x)];
                        neighbor_score += abs(pixel - neighbor_center);
                    }
                    if (neighbor_score >= score) {
                        is_local_maximum = false;
                        break;
                    }
                }
            }
        }
        
        if (is_local_maximum) {
            int local_idx = atomicAdd(&block_counter, 1);
            if (local_idx < 256) {
                block_keypoints[local_idx] = make_float2(x, y);
                block_descriptors[local_idx] = desc;
            }
        }
    }
    
    __syncthreads();
    
    // Store keypoints globally
    if (threadIdx.x == 0 && threadIdx.y == 0 && block_counter > 0) {
        int global_idx = atomicAdd(keypoint_count, block_counter);
        // printf("globalIdx: %d, block_counter: %d => max_keypoints: %d\n", global_idx, block_counter, max_keypoints);

        if (global_idx + block_counter <= max_keypoints) {
            for (int i = 0; i < block_counter; i++) {
                float2 kp = block_keypoints[i];
                float3 world_pos = convertImageToWorldCoords(
                    kp.x, kp.y,
                    image_width, image_height
                );
                
                positions[global_idx + i] = make_float4(world_pos.x, world_pos.y, world_pos.z, 1.0f);
                colors[global_idx + i] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
                descriptors[global_idx + i] = block_descriptors[i];
            }
        }
    }
}


// ============================================================= Matching =================================================================
// Matching parameters
struct MatchingParams {
    float baseline;
    float focal_length;
    float max_disparity;
    float epipolar_threshold;
    float sensor_width_mm;
    float sensor_width_pixels;
    float sensor_height_pixels;
};

// Structure to hold matched keypoint data
struct MatchedKeypoint {
    float3 left_pos;
    float3 right_pos;
    float3 world_pos;
    float disparity;
};

struct Keypoint {
    float3 position;    // Position in OpenGL world coordinates
    float disparity;    // Pixel disparity between left and right views
};

__device__ Keypoint triangulatePosition(
    float3 leftWorldPos,
    float3 rightWorldPos,
    float baseline,        
    float canvas_width
) {
    Keypoint result;
    
    // Calculate disparity with more stable threshold
    float worldDisparity = leftWorldPos.x - rightWorldPos.x;
    // if (fabsf(worldDisparity) < 0.01f) {  // Increased threshold
    //     result.position = make_float3(0.0f, 0.0f, 0.0f);
    //     result.disparity = worldDisparity;
    //     return result;
    // }

    float mm_per_world_unit = 6.45f / 6.4f;
    float disparity_mm = worldDisparity * mm_per_world_unit;
   

    // Calculate depth using similar triangles principle
    float depth_mm = (baseline * 2.612f) / disparity_mm;  // 3.04f is focal length in mm

    
    // Clamp depth to reasonable range (adjust these values based on your scene)
    // const float MIN_DEPTH_MM = 100.0f;   // 10cm
    // const float MAX_DEPTH_MM = 5000.0f;  // 5m
    // depth_mm = fmaxf(fminf(depth_mm, MAX_DEPTH_MM), MIN_DEPTH_MM);
 
    float depth_world = depth_mm / mm_per_world_unit;

    // Calculate final world position
    float world_scale = depth_world / depth_mm;
    float worldX = leftWorldPos.x;  // Use left camera x position
    float worldY = -depth_world / 1000;    // Negative because OpenGL Y goes up
    float worldZ = leftWorldPos.z * world_scale;
    
    // Bound check the final coordinates
    // worldX = fmaxf(fminf(worldX, canvas_width/2), -canvas_width/2);
    // worldZ = fmaxf(fminf(worldZ, canvas_width/2), -canvas_width/2);
    
    result.position = make_float3(rightWorldPos.x, -0.1, rightWorldPos.z);
    result.disparity = disparity_mm;  // Store disparity in mm for debugging
    
    return result;
}

__device__ inline int hammingDistance(const BRIEFDescriptor& desc1, const BRIEFDescriptor& desc2) {
    int distance = 0;
   
    for (int i = 0; i < 8; i++) {
        uint64_t xor_result = desc1.descriptor[i] ^ desc2.descriptor[i];
        distance += __popcll(xor_result);  // Count differing bits
    }
    return distance;
}


__global__ void matchKeypointsKernel(
    const float4* __restrict__ left_positions,
    const float4* __restrict__ right_positions,
    const BRIEFDescriptor* __restrict__ left_descriptors,
    const BRIEFDescriptor* __restrict__ right_descriptors,
    const int left_count,
    const int right_count,
    const MatchingParams params,
    MatchedKeypoint* matches,
    int* match_count,
    const int max_matches,
    cudaSurfaceObject_t source_tex,      // Y plane texture from left image
    cudaSurfaceObject_t combined_tex     // Target texture for visualization

) {
    extern __shared__ float shared_mem[];
    
    // Split shared memory into different arrays
    float* min_costs = shared_mem;
    float* second_min_costs = &min_costs[blockDim.x];
    int* best_matches = (int*)&second_min_costs[blockDim.x];
    int* used_right_points = &best_matches[blockDim.x];
    int* shared_count = (int*)&used_right_points[right_count];

    const int left_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (threadIdx.x < blockDim.x) {
        min_costs[threadIdx.x] = INFINITY;
        second_min_costs[threadIdx.x] = INFINITY;
        best_matches[threadIdx.x] = -1;
    }
    
    if (threadIdx.x < right_count) {
        used_right_points[threadIdx.x] = 0;
    }
    
    if (threadIdx.x == 0) {
        shared_count[0] = 0;
    }

    __syncthreads();

    if (left_idx >= left_count) return;
    
    float3 left_pos = make_float3(
        left_positions[left_idx].x,
        left_positions[left_idx].y,
        left_positions[left_idx].z
    );

    BRIEFDescriptor left_desc = left_descriptors[left_idx];

    float best_desc_distance = INFINITY;
    float best_y_diff = INFINITY;
    float best_disparity = INFINITY;


    const int MAX_DESC_DISTANCE = 64.0f; 
    
    // Find best and second-best matches for this left keypoint
    for (int right_idx = 0; right_idx < right_count; right_idx++) {
        float3 right_pos = make_float3(
            right_positions[right_idx].x,
            right_positions[right_idx].y,
            right_positions[right_idx].z
        );

        // Check epipolar constraint (y-difference)
        float y_diff = fabsf(left_pos.y - right_pos.y);
        float disparity = left_pos.x - right_pos.x;

        if (y_diff > params.epipolar_threshold || 
            disparity <= 0 || 
            disparity > params.max_disparity) {
            continue;
        }
      
        // Calculate descriptor distance
        int desc_distance = hammingDistance(left_desc, right_descriptors[right_idx]);
        if (desc_distance > MAX_DESC_DISTANCE) continue;
      
        // Calculate individual costs
        float desc_cost = desc_distance / 512.0f;
        float epipolar_cost = y_diff / params.epipolar_threshold;
        float disparity_cost = disparity / params.max_disparity;

        // Weighted combination of costs
        float total_cost = epipolar_cost * 0.3f +
                          desc_cost * 0.5f +
                          disparity_cost * 0.2f;

        // Maintain best and second-best matches
        if (total_cost < min_costs[threadIdx.x]) {
            best_desc_distance = desc_distance;
            best_y_diff = y_diff;
            best_disparity = disparity;
            
            second_min_costs[threadIdx.x] = min_costs[threadIdx.x];
            min_costs[threadIdx.x] = total_cost;
            best_matches[threadIdx.x] = right_idx;
        } else if (total_cost < second_min_costs[threadIdx.x]) {
            second_min_costs[threadIdx.x] = total_cost;
        }
    }

    __syncthreads();

    // Apply Lowe's ratio test and absolute threshold
    const float RATIO_THRESH = 0.7f; 
    const float ABS_COST_THRESH = 0.25f;
    
    bool is_good_match = false;
    int right_idx = best_matches[threadIdx.x];

    if (right_idx >= 0) {
        bool passes_thresholds = (min_costs[threadIdx.x] < ABS_COST_THRESH) &&
                                (min_costs[threadIdx.x] < second_min_costs[threadIdx.x] * RATIO_THRESH);
        
        bool passes_validation = (best_y_diff < params.epipolar_threshold * 0.5f) &&
                                (best_disparity > 1.0f) &&
                                (best_desc_distance < MAX_DESC_DISTANCE * 0.75f);
        
        is_good_match = passes_thresholds && passes_validation;
        
        // Try to claim the right point
        if (is_good_match && right_idx < right_count) {
            is_good_match = (atomicCAS(&used_right_points[right_idx], 0, 1) == 0);
            if (is_good_match) {
                atomicAdd(shared_count, 1);
            }
        }
    }

     __syncthreads();

    // Update global match count
    if (threadIdx.x == 0 && shared_count[0] > 0) {
        atomicAdd(match_count, min(shared_count[0], max_matches));
    }

    __syncthreads();

    // Store matches, respecting the max_matches limit
    if (is_good_match) {
        int match_idx = atomicSub(shared_count, 1) - 1;
        if (match_idx >= 0 && match_idx < max_matches) {
            float3 right_pos = make_float3(
                right_positions[right_idx].x,
                right_positions[right_idx].y,
                right_positions[right_idx].z
            );

            Keypoint matchedPoint = triangulatePosition(
                left_pos,
                right_pos,
                params.baseline,
                6.4f
            );

            matches[match_idx] = {
                left_pos,
                right_pos,
                matchedPoint.position,
                matchedPoint.disparity
            };
        }
    }
}

__global__ void setTextureKernel(
    cudaSurfaceObject_t y_surface,
    cudaSurfaceObject_t uv_surface,
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ uv_plane,
    int width,
    int height,
    int y_linesize,
    int uv_linesize
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    unsigned char pixel = y_plane[y * y_linesize + x];
    surf2Dwrite(pixel, y_surface, x * sizeof(u_char) , y);

    if (x < width/2 && y < height/2) {
        // UV planes are interleaved in memory as UVUV...
        const int uv_index = (y * uv_linesize) + (x * 2);
        uchar2 uv_pixels = make_uchar2(
            uv_plane[uv_index],     // U component
            uv_plane[uv_index + 1]  // V component
        );
        surf2Dwrite(uv_pixels, uv_surface, x * sizeof(uchar2), y);
    }
}

__global__ void copySurfaceKernel(
    cudaSurfaceObject_t srcSurf_y,
    cudaSurfaceObject_t srcSurf_uv,
    cudaSurfaceObject_t dstSurf_y,
    cudaSurfaceObject_t dstSurf_uv,
    int width,
    int height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
 
   
    unsigned char pixel;
    surf2Dread(&pixel, srcSurf_y, x, y);
    surf2Dwrite(pixel, dstSurf_y, x * sizeof(u_char), y);

    if (x < width / 2 && y < height / 2) { 
        uchar2 uv;
        surf2Dread(&uv, srcSurf_uv, x * sizeof(uchar2), y);
        surf2Dwrite(uv, dstSurf_uv, x * sizeof(uchar2), y);
    }
}


// Kernel to generate visualization data
__global__ void generateVisualizationKernel(
    const MatchedKeypoint* matches,
    const int match_count,
    float4* keypoint_positions,
    float4* keypoint_colors,
    
    float4* left_line_positions,
    float4* left_line_colors,

    float4* right_line_positions,
    float4* right_line_colors
) {
   const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= match_count) return;

    const MatchedKeypoint match = matches[idx];
    
    // Center point (world position)
    keypoint_positions[idx] = make_float4(match.world_pos.x, match.world_pos.y, match.world_pos.z, 1.0f);
    keypoint_colors[idx] = make_float4(1.0f, 0.0f, 1.0f, 1.0f); // Purple for center
    
    // For the left line connection:
    // Start position (2 vertices per line, so idx * 2)
    left_line_positions[idx * 2] = make_float4(match.left_pos.x - 2.5f, match.left_pos.y + 5.0f, match.left_pos.z, 1.0f);
    left_line_colors[idx * 2] = make_float4(1.0f, 0.0f, 0.0f, 1.0f); // Red start
    
    // End position
    left_line_positions[idx * 2 + 1] = make_float4(match.world_pos.x, match.world_pos.y, match.world_pos.z, 1.0f);
    left_line_colors[idx * 2 + 1] = make_float4(1.0f, 0.0f, 0.0f, 0.5f); // Red end (semi-transparent)
    
    // For the right line connection:
    // Start position
    right_line_positions[idx * 2] = make_float4(match.right_pos.x + 2.5f, match.right_pos.y + 5.0f, match.right_pos.z, 1.0f);
    right_line_colors[idx * 2] = make_float4(0.0f, 0.0f, 1.0f, 1.0f); // Blue start
    
    // End position
    right_line_positions[idx * 2 + 1] = make_float4(match.world_pos.x, match.world_pos.y , match.world_pos.z, 1.0f);
    right_line_colors[idx * 2 + 1] = make_float4(0.0f, 0.0f, 1.0f, 0.5f); // Blue end (semi-transparent)
}



// ============================================================= Bindings =================================================================

extern "C" {

//  int cudaDevice;
//     error = cudaGetDevice(&cudaDevice);
//     if (error != cudaSuccess) {
//         printf("Error getting CUDA device: %s\n", cudaGetErrorString(error));
//         return -1;
//     }

// cudaDeviceProp prop;
//     error = cudaGetDeviceProperties(&prop, cudaDevice);
//     if (error != cudaSuccess) {
//         printf("Error getting CUDA device properties: %s\n", cudaGetErrorString(error));
//         return -1;
//     }

//     printf("Using CUDA Device: %d - %s\n", cudaDevice, prop.name);

int cuda_create_detector(int max_keypoints, int gl_ytexture, int gl_uvtexture) {
    int slot = find_free_detector_slot();
    if (slot < 0) {
        return -1;
    }

    if (cudaMalloc(&g_detectors[slot].d_keypoint_count, sizeof(int)) != cudaSuccess) {
        return -1;
    }

    if (cudaMalloc(&g_detectors[slot].d_descriptors, max_keypoints * sizeof(BRIEFDescriptor) != cudaSuccess)) {
        return -1;
    } ;

    g_detectors[slot].initialized = true;
    g_detectors[slot].id = g_next_detector_id++;
    g_detectors[slot].gl_ytexture = gl_ytexture;
    g_detectors[slot].gl_uvtexture = gl_uvtexture;

    return g_detectors[slot].id;
}

void cuda_cleanup_detector(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return;

    cuda_unregister_buffers(detector_id);

    if (detector->d_keypoint_count) cudaFree(detector->d_keypoint_count);
    if (detector->d_descriptors) cudaFree(detector->d_descriptors);
    detector->d_keypoint_count = nullptr;
    detector->d_descriptors = nullptr;
}


int cuda_register_gl_texture(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return -1;

    printf("Registering GL textures for detector %d\n", detector_id);
    printf("GL texture IDs - Y: %d, UV: %d\n", detector->gl_ytexture, detector->gl_uvtexture);

    // Allocate texture resource
     detector->y_texture = (CudaGLTextureResource*)malloc(sizeof(CudaGLTextureResource));
    if (!detector->y_texture) {
        printf("Failed to allocate Y texture resource\n");
        return -1;
    }
    memset(detector->y_texture, 0, sizeof(CudaGLTextureResource));

    // Register Y texture
    cudaError_t err = cudaGraphicsGLRegisterImage(
        &detector->y_texture->tex_resource,
        detector->gl_ytexture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard
    );
    
    if (err != cudaSuccess) {
        printf("Failed to register Y texture: %s\n", cudaGetErrorString(err));
        free(detector->y_texture);
        detector->y_texture = NULL;
        return -1;
    }

    // Allocate UV texture resource
    detector->uv_texture = (CudaGLTextureResource*)malloc(sizeof(CudaGLTextureResource));
    if (!detector->uv_texture) {
        printf("Failed to allocate UV texture resource\n");
        cudaGraphicsUnregisterResource(detector->y_texture->tex_resource);
        free(detector->y_texture);
        detector->y_texture = NULL;
        return -1;
    }
    memset(detector->uv_texture, 0, sizeof(CudaGLTextureResource));

    // Register UV texture
    err = cudaGraphicsGLRegisterImage(
        &detector->uv_texture->tex_resource,
        detector->gl_uvtexture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard
    );
    
    if (err != cudaSuccess) {
        printf("Failed to register UV texture: %s\n", cudaGetErrorString(err));
        cudaGraphicsUnregisterResource(detector->y_texture->tex_resource);
        free(detector->y_texture);
        detector->y_texture = NULL;
        free(detector->uv_texture);
        detector->uv_texture = NULL;
        return -1;
    }


    return 0;
}

void cuda_unregister_gl_texture(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return;

    if (detector->y_texture){
        cudaGraphicsUnregisterResource(detector->y_texture->tex_resource);
        free(detector->y_texture);
        detector->y_texture = NULL;
    }

    if (detector->uv_texture){
        cudaGraphicsUnregisterResource(detector->uv_texture->tex_resource);
        free(detector->uv_texture);
        detector->uv_texture = NULL;
    }
}


int cuda_register_instance_buffer(
    InstanceBuffer* buffer,  
    unsigned int position_buffer,
    unsigned int color_buffer,
    unsigned int buffer_size
) {
    cudaError_t error = cudaGraphicsGLRegisterBuffer(
        &buffer->position_resource, 
        position_buffer,
        cudaGraphicsRegisterFlagsWriteDiscard
    );

    if (error != cudaSuccess) {
        printf("Error registering Line Position Buffer %d => %d\n", position_buffer, error);
        return -1;
    }

    error = cudaGraphicsGLRegisterBuffer(
        &buffer->color_resource,
        color_buffer,
        cudaGraphicsRegisterFlagsWriteDiscard
    );
    if (error != cudaSuccess) {
        printf("Error registering Line Color Buffer %d => %d\n", color_buffer, error);
        return -1;
    }

    buffer->buffer_size = buffer_size;

    return 0;  // Added missing return
}


int cuda_register_buffers(
    int detector_id,
    int keypoint_position_buffer,
    int keypoint_color_buffer,
    int* left_position_buffer,
    int* left_color_buffer,
    int* right_position_buffer,
    int* right_color_buffer,
    int buffer_size
) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return -1;

    int error;

    error = cuda_register_instance_buffer(
        &detector->gl_resources.keypoints,  // Pass address
        keypoint_position_buffer, 
        keypoint_color_buffer,
        buffer_size
    );
    if (error < 0) return error;

    if (left_position_buffer != NULL && left_color_buffer != NULL) {
        error = cuda_register_instance_buffer(
            &detector->gl_resources.connections.left,  // Pass address
            *left_position_buffer, 
            *left_color_buffer,
            buffer_size * 2
        );

        if (error < 0) return error;
    }

    if (right_position_buffer != NULL && right_color_buffer != NULL) {
        error = cuda_register_instance_buffer(
            &detector->gl_resources.connections.right,  // Pass address
            *right_position_buffer, 
            *right_color_buffer,
            buffer_size * 2
        );

        if (error < 0) return error;
    }

    
    return 0;
}


void cuda_unregister_buffers(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return;

    if (detector->gl_resources.keypoints.position_resource) {
        cudaGraphicsUnregisterResource(detector->gl_resources.keypoints.position_resource);
    }
    if (detector->gl_resources.keypoints.color_resource) {
        cudaGraphicsUnregisterResource(detector->gl_resources.keypoints.color_resource);
    }

    InstanceBuffer left = detector->gl_resources.connections.left;
    if (left.position_resource) {
            cudaGraphicsUnregisterResource(detector->gl_resources.connections.left.position_resource);
    }
    if (left.color_resource) {
            cudaGraphicsUnregisterResource(detector->gl_resources.connections.left.color_resource);
    }

    InstanceBuffer right = detector->gl_resources.connections.right;
    if (right.position_resource) {
            cudaGraphicsUnregisterResource(detector->gl_resources.connections.right.position_resource);
    }
    if (right.color_resource) {
            cudaGraphicsUnregisterResource(detector->gl_resources.connections.right.color_resource);
    }

    detector->gl_resources = {};
}



int cuda_map_buffer_resources(InstanceBuffer* buffer) {  
    cudaError_t error; 
    error = cudaGraphicsMapResources(1, &buffer->position_resource);
    if (error != cudaSuccess) {
        printf("Failed to Map Position Resources: %d\n", error);
        return -1;
    }

    error = cudaGraphicsMapResources(1, &buffer->color_resource);
    if (error != cudaSuccess) {
        printf("Failed to Map Color Resources: %d\n", error);
        return -1;
    }

    size_t bytes;
    error = cudaGraphicsResourceGetMappedPointer(
        (void**)&buffer->d_positions,
        &bytes,
        buffer->position_resource
    );

    if (error != cudaSuccess) {
        printf("Failed to get Positions Mapped Pointer: %d\n", error);
        return -1;
    }

    error = cudaGraphicsResourceGetMappedPointer(
        (void**)&buffer->d_colors,
        &bytes,
        buffer->color_resource
    );

    if (error != cudaSuccess) {
        printf("Failed to get Colors Mapped Pointer: %d\n", error);
        return -1;
    }

    return 0;
}


int cuda_map_resources(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return -1;

    printf("Mapping resources for detector %d\n", detector_id);
    printf("Keypoint buffers: pos=%p, color=%p\n", 
        detector->gl_resources.keypoints.position_resource,
        detector->gl_resources.keypoints.color_resource);
    printf("Y texture: %p, UV texture: %p\n", 
        detector->y_texture, 
        detector->uv_texture);

    int error;
      
    
    error = cuda_map_buffer_resources(&detector->gl_resources.keypoints);
    if (error < 0) printf("Failed to map keypoint buffers!\n");
    if (detector->gl_resources.connections.left.position_resource){
        error = cuda_map_buffer_resources(&detector->gl_resources.connections.left);
        if (error < 0)  printf("Failed to map left connection buffers!\n");
    }
    if (detector->gl_resources.connections.right.position_resource){
        error = cuda_map_buffer_resources(&detector->gl_resources.connections.right);
        if (error < 0) printf("Failed to map right connection buffers!\n");
    }

    if (error < 0) {
        printf("Failed to map resources!\n");
        cuda_unmap_resources(detector_id);
        return -1;
    }

    if (detector->y_texture) {
        cudaError_t err = cudaGraphicsMapResources(1, &detector->y_texture->tex_resource);
        if (err != cudaSuccess) {
            printf("Failed to map Y texture resource: %s\n", cudaGetErrorString(err));
            cuda_unmap_resources(detector_id);
            return -1;
        }

        err = cudaGraphicsSubResourceGetMappedArray(&detector->y_texture->array,
                                                  detector->y_texture->tex_resource,
                                                  0, 0);
        if (err != cudaSuccess) {
            printf("Failed to get mapped array for Y texture: %s\n", cudaGetErrorString(err));
            cuda_unmap_resources(detector_id);
            return -1;
        }

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = detector->y_texture->array;

        err = cudaCreateSurfaceObject(&detector->y_texture->surface, &res_desc);
        if (err != cudaSuccess) {
            printf("Failed to create Y surface object: %s\n", cudaGetErrorString(err));
            cuda_unmap_resources(detector_id);
            return -1;
        }
    }

    if (detector->uv_texture) {
        cudaError_t err = cudaGraphicsMapResources(1, &detector->uv_texture->tex_resource);
        if (err != cudaSuccess) {
            printf("Failed to map UV texture resource: %s\n", cudaGetErrorString(err));
            cuda_unmap_resources(detector_id);
            return -1;
        }

        err = cudaGraphicsSubResourceGetMappedArray(&detector->uv_texture->array,
                                                  detector->uv_texture->tex_resource,
                                                  0, 0);
        if (err != cudaSuccess) {
            printf("Failed to get mapped array for UV texture: %s\n", cudaGetErrorString(err));
            cuda_unmap_resources(detector_id);
            return -1;
        }

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = detector->uv_texture->array;

        err = cudaCreateSurfaceObject(&detector->uv_texture->surface, &res_desc);
        if (err != cudaSuccess) {
            printf("Failed to create UV surface object: %s\n", cudaGetErrorString(err));
            cuda_unmap_resources(detector_id);
            return -1;
        }
    }

    return error == cudaSuccess ? 0 : -1;
}

void cuda_unmap_resources(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return;

    if (detector->gl_resources.keypoints.position_resource) {
        cudaGraphicsUnmapResources(1, &detector->gl_resources.keypoints.position_resource);
    }
    if (detector->gl_resources.keypoints.color_resource) {
        cudaGraphicsUnmapResources(1, &detector->gl_resources.keypoints.color_resource);
    }

    // Check left connection resources
    if (detector->gl_resources.connections.left.position_resource) {
        cudaGraphicsUnmapResources(1, &detector->gl_resources.connections.left.position_resource);
    }
    if (detector->gl_resources.connections.left.color_resource) {
        cudaGraphicsUnmapResources(1, &detector->gl_resources.connections.left.color_resource);
    }

    // Check right connection resources
    if (detector->gl_resources.connections.right.position_resource) {
        cudaGraphicsUnmapResources(1, &detector->gl_resources.connections.right.position_resource);
    }
    if (detector->gl_resources.connections.right.color_resource) {
        cudaGraphicsUnmapResources(1, &detector->gl_resources.connections.right.color_resource);
    }

    if (detector->y_texture) {
        cudaDestroySurfaceObject(detector->y_texture->surface);
        cudaGraphicsUnmapResources(1, &detector->y_texture->tex_resource);
    }

    if (detector->uv_texture) {
        cudaDestroySurfaceObject(detector->uv_texture->surface);
        cudaGraphicsUnmapResources(1, &detector->uv_texture->tex_resource);
    }
}

static void cleanup_resources(
    int detector_id_left, 
    int detector_id_right, 
    int detector_id_combined,
    MatchedKeypoint* d_matches, 
    int* d_match_count,
    cudaEvent_t start, 
    cudaEvent_t stop
) {
    if (d_matches) cudaFree(d_matches);
    if (d_match_count) cudaFree(d_match_count);
    
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    
    cuda_unmap_resources(detector_id_left);
    cuda_unmap_resources(detector_id_right);
    cuda_unmap_resources(detector_id_combined);
    
    cuda_unregister_gl_texture(detector_id_left);
    cuda_unregister_gl_texture(detector_id_right);
    cuda_unregister_gl_texture(detector_id_combined);
}

// Helper function to set up resources
static int setup_resources(int detector_id_left, int detector_id_right, int detector_id_combined) {
    printf("Setting up resources for Detectors: {%d, %d, %d}\n", detector_id_left, detector_id_right, detector_id_combined);
    if (cuda_register_gl_texture(detector_id_left) < 0 ||
        cuda_register_gl_texture(detector_id_right) < 0 ||
        cuda_register_gl_texture(detector_id_combined) < 0) {
        printf("Failed to Register Textures!\n");
        return -1;
    }

    printf("Successfully registered textures. Moving onto mapping resources...\n");
    if (cuda_map_resources(detector_id_left) < 0 ||
        cuda_map_resources(detector_id_right) < 0 ||
        cuda_map_resources(detector_id_combined) < 0) {
        printf("Failed to Map Resources!\n");
        return -1;
    }

    return 0;
}

// Helper function to validate detector instances
static int validate_detectors(DetectorInstance* left, DetectorInstance* right, DetectorInstance* combined) {
    if (!left || !right || !combined) {
        printf("Invalid detector instances\n");
        return -1;
    }
    return 0;
}


void initGaussianKernel(float sigma) {
    float h_gaussian_kernel[GAUSSIAN_KERNEL_SIZE];
    float sum = 0.0f;
    
    // Calculate Gaussian kernel values
    for (int i = -GAUSSIAN_KERNEL_RADIUS; i <= GAUSSIAN_KERNEL_RADIUS; i++) {
        float x = i;
        h_gaussian_kernel[i + GAUSSIAN_KERNEL_RADIUS] = expf(-(x * x) / (2 * sigma * sigma));
        sum += h_gaussian_kernel[i + GAUSSIAN_KERNEL_RADIUS];
    }
    
    // Normalize kernel
    for (int i = 0; i < GAUSSIAN_KERNEL_SIZE; i++) {
        h_gaussian_kernel[i] /= sum;
    }
    
    // Copy kernel to constant memory
    cudaMemcpyToSymbol(d_gaussian_kernel, h_gaussian_kernel, GAUSSIAN_KERNEL_SIZE * sizeof(float));
}



float cuda_detect_keypoints(
    int detector_id,
    uint8_t threshold,
    ImageParams* image,
    float sigma = 1.0f
) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    
    if (!detector) return -1.0;
    
    cudaError_t error;
 
    uint8_t *d_temp1, *d_temp2;
    cudaMalloc(&d_temp1, image->y_linesize * image->height);
    cudaMalloc(&d_temp2, image->y_linesize * image->height);

    printf("Image Params: %d %d\n", image->y_linesize, image->height);
    
    // Initialize Gaussian kernel
    initGaussianKernel(sigma);
    
    // Setup kernel launch parameters
    dim3 block(16, 16);
    dim3 grid(
        (image->width + block.x - 1) / block.x,
        (image->height + block.y - 1) / block.y
    );
    
    // Apply horizontal Gaussian blur
    gaussianBlurHorizontal<<<grid, block>>>(
        image->y_plane,
        d_temp1,
        image->width,
        image->height,
        image->y_linesize
    );

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Gaussian Blur horizontal Kernel failed: %d\n", error);
        return -1.0;
    }
    
    // Apply vertical Gaussian blur
    gaussianBlurVertical<<<grid, block>>>(
        d_temp1,
        d_temp2,
        image->width,
        image->height,
        image->y_linesize
    );

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Gaussian Blur vertical Kernel failed: %d\n", error);
        return -1.0;
    }

    // Reset keypoint counter
    error = cudaMemset(detector->d_keypoint_count, 0, sizeof(int));
    if (error != cudaSuccess) {
        printf("Failed to reset keypoint count: %d\n", error);
        return -1.0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    cudaEventRecord(start);
    detectFASTKeypoints<<<grid, block>>>(
        d_temp2,
        image->width,
        image->height,
        image->y_linesize,
        threshold,
        detector->gl_resources.keypoints.d_positions,
        detector->gl_resources.keypoints.d_colors,
        detector->d_descriptors,
        detector->d_keypoint_count,
        detector->gl_resources.keypoints.buffer_size,
        image->image_width,
        image->image_height
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds; 
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Detection Kernel Timing (ms): %f\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Keypoint Detection Kernel failed: %d\n", error);
        return -1.0;
    }

    cudaFree(d_temp1);
    cudaFree(d_temp2);

    // Get keypoint count
    error = cudaMemcpy(image->num_keypoints, detector->d_keypoint_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Failed to copy keypoint count: %d\n", error);
        return -1.0;
    }
    
    return error == cudaSuccess ? milliseconds : -1.0;
}


static int allocate_matching_resources(MatchedKeypoint** d_matches, int** d_match_count,
                                     cudaEvent_t* start, cudaEvent_t* stop, int max_matches) {
    cudaError_t error;
    
    error = cudaMalloc(d_matches, max_matches * sizeof(MatchedKeypoint));
    if (error != cudaSuccess) return -1;
    
    error = cudaMalloc(d_match_count, sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(*d_matches);
        return -1;
    }
    
    error = cudaMemset(*d_match_count, 0, sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(*d_matches);
        cudaFree(*d_match_count);
        return -1;
    }
    
    error = cudaEventCreate(start);
    if (error != cudaSuccess) {
        cudaFree(*d_matches);
        cudaFree(*d_match_count);
        return -1;
    }
    
    error = cudaEventCreate(stop);
    if (error != cudaSuccess) {
        cudaEventDestroy(*start);
        cudaFree(*d_matches);
        cudaFree(*d_match_count);
        return -1;
    }
    
    return 0;
}

static float execute_matching(
    DetectorInstance* left_detector,
    DetectorInstance* right_detector, 
    DetectorInstance* combined_detector,
    MatchedKeypoint* d_matches,
    int* d_match_count,
    int max_matches,
    int* num_matches,
    float baseline,
    float focal_length,
    ImageParams* left,
    ImageParams* right,
    cudaEvent_t start,
    cudaEvent_t stop
) {
    // Set up matching parameters
    float sensor_width_mm = 6.45f;
    float baseline_world = (baseline / sensor_width_mm) * 6.4f;
    
    MatchingParams params = {
        .baseline = baseline_world,
        .focal_length = focal_length,
        .max_disparity = 200.0f,
        .epipolar_threshold = 150.0f,
        .sensor_width_mm = sensor_width_mm,
        .sensor_width_pixels = 4608.0f,
        .sensor_height_pixels = 2592.0f
    };


    // Configure kernel launch parameters
    dim3 blockMatching(512);
    dim3 gridMatching((max_matches + blockMatching.x - 1) / blockMatching.x);

    size_t shared_mem_size = 
        (2 * sizeof(float) + sizeof(int)) * blockMatching.x + // min_costs, second_min_costs, best_matches
        sizeof(int) * max_matches +                          // used_right_points
        sizeof(int);                                         // shared_count

    // Execute matching kernel
    cudaEventRecord(start);
    matchKeypointsKernel<<<gridMatching, blockMatching, shared_mem_size>>>(
        left_detector->gl_resources.keypoints.d_positions,
        right_detector->gl_resources.keypoints.d_positions,
        left_detector->d_descriptors,
        right_detector->d_descriptors,
        max_matches,
        max_matches,
        params,
        d_matches,
        d_match_count,
        max_matches,
        left_detector->y_texture->surface,
        combined_detector->y_texture->surface
    );
    cudaEventRecord(stop);

    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Matching kernel failed: %s\n", cudaGetErrorString(error));
        return -1.0;
    }
    cudaEventSynchronize(stop);

    float milliseconds_matching;
    cudaEventElapsedTime(&milliseconds_matching, start, stop);
    printf("Matching Kernel Time: %.2f ms\n", milliseconds_matching);

    // Get match count
    error = cudaMemcpy(num_matches, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Failed to copy match count: %s\n", cudaGetErrorString(error));
        return -1.0;
    }

    if (*num_matches == 0) {
        printf("No Matches detected!\n");
        return milliseconds_matching;
    }

    printf("num_matches: %d\n", *num_matches);

    // Generate visualization
    dim3 blockVis(1024);
    dim3 gridVis((*num_matches + blockVis.x - 1) / blockVis.x);

    cudaEventRecord(start);
    generateVisualizationKernel<<<gridVis, blockVis>>>(
        d_matches,
        *num_matches,
        combined_detector->gl_resources.keypoints.d_positions,
        combined_detector->gl_resources.keypoints.d_colors,
        combined_detector->gl_resources.connections.left.d_positions,
        combined_detector->gl_resources.connections.left.d_colors,
        combined_detector->gl_resources.connections.right.d_positions,
        combined_detector->gl_resources.connections.right.d_colors
    );
    cudaEventRecord(stop);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Visualization kernel failed: %s\n", cudaGetErrorString(error));
        return -1.0;
    }

    cudaEventSynchronize(stop);
    float milliseconds_vis;
    cudaEventElapsedTime(&milliseconds_vis, start, stop);
    printf("Visualization time: %.2f ms\n", milliseconds_vis);
    
    return milliseconds_vis + milliseconds_matching;
}

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
) {
    float sigma = 1.25f;

    DetectorInstance* left_detector = get_detector_instance(detector_id_left);
    DetectorInstance* right_detector = get_detector_instance(detector_id_right);
    DetectorInstance* combined_detector = get_detector_instance(detector_id_combined);

    if (validate_detectors(left_detector, right_detector, combined_detector) < 0) {
        return -1;
    }

    printf("Entering Setup\n");
    if (setup_resources(detector_id_left, detector_id_right, detector_id_combined) < 0) {
        return -1;
    }

    dim3 blockCopy(16, 16);
    dim3 gridCopy(
        (left->width + blockCopy.x - 1) / blockCopy.x,
        (left->height + blockCopy.y - 1) / blockCopy.y
    );

    // Set texture for left and right images
    setTextureKernel<<<gridCopy, blockCopy>>>(
        left_detector->y_texture->surface,
        left_detector->uv_texture->surface,
        left->y_plane,
        left->uv_plane,
        left->width,
        left->height,
        left->y_linesize,
        left->uv_linesize
    );

    setTextureKernel<<<gridCopy, blockCopy>>>(
        right_detector->y_texture->surface,
        right_detector->uv_texture->surface,
        right->y_plane,
        right->uv_plane,
        right->width,
        right->height,
        right->y_linesize,
        right->uv_linesize
    );

     // Copy surface from right to combined detector
    copySurfaceKernel<<<gridCopy, blockCopy>>>(
        right_detector->y_texture->surface,
        right_detector->uv_texture->surface,
        combined_detector->y_texture->surface,
        combined_detector->uv_texture->surface,
        right->width,
        right->height
    );

    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Surface copy kernel failed: %s\n", cudaGetErrorString(error));
        cleanup_resources(detector_id_left, detector_id_right, detector_id_combined, NULL, NULL, NULL, NULL);
    
        return -1;
    }

    // Detect keypoints
    float detection_time_left = cuda_detect_keypoints(detector_id_left, threshold, left, sigma);
    float detection_time_right = cuda_detect_keypoints(detector_id_right, threshold, right, sigma);
    
    cudaDeviceSynchronize();

    if (detection_time_left < 0 || detection_time_right < 0) {
        cleanup_resources(detector_id_left, detector_id_right, detector_id_combined, NULL, NULL, NULL, NULL);
        return -1;
    }

    // Allocate resources for matching
    MatchedKeypoint* d_matches = NULL;
    int* d_match_count = NULL;
    cudaEvent_t start = NULL, stop = NULL;
    
    const int max_matches = min(*right->num_keypoints, *left->num_keypoints);
    if (max_matches <= 0) {
         cleanup_resources(
            detector_id_left, 
            detector_id_right, 
            detector_id_combined,
            d_matches, 
            d_match_count, 
            start, 
            stop
        );
        printf("No matches possbile!\n");
        return 0;
    }

    if (allocate_matching_resources(&d_matches, &d_match_count, &start, &stop, max_matches) < 0) {
       cleanup_resources(
            detector_id_left, 
            detector_id_right, 
            detector_id_combined,
            d_matches, 
            d_match_count, 
            start, 
            stop
        );
        return -1;
    }

    float matching_time = execute_matching(
        left_detector, 
        right_detector, 
        combined_detector,
        d_matches, 
        d_match_count, 
        max_matches, 
        num_matches,
        baseline,
        focal_length, 
        left, 
        right, 
        start, 
        stop
    );

    // Perform matching
    if (matching_time < 0) {
        cleanup_resources(
            detector_id_left, 
            detector_id_right, 
            detector_id_combined,
            d_matches, 
            d_match_count, 
            start, 
            stop
        );
        cudaDeviceSynchronize();
        printf("Matching Failed!\n");
        return -1;
    }

    printf("Total Pipeline Execution Time: %f\n", detection_time_left + detection_time_right + matching_time);

    cleanup_resources(
        detector_id_left, 
        detector_id_right, 
        detector_id_combined,
        d_matches, 
        d_match_count, 
        start, 
        stop
    );
    cudaDeviceSynchronize();

    return 0;
}

}