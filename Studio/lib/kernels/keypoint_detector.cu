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
    float4* positions,
    float4* colors
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= match_count) return;

    const MatchedKeypoint match = matches[idx];
    // const int base_idx = idx; // 3 vertices per match (triangle strip)

    // Center point (world position)
    positions[idx] = make_float4(match.world_pos.x, match.world_pos.y, match.world_pos.z, 1.0f);
    colors[idx] = make_float4(0.5f, 0.0f, 0.5f, 1.0f); // Purple for center

    // Left keypoint
    // positions[base_idx + 1] = make_float4(match.left_pos.x, match.left_pos.y, match.left_pos.z, 1.0f);
    // colors[base_idx + 1] = make_float4(1.0f, 0.0f, 0.0f, 1.0f); // Red for left

    // // Right keypoint
    // positions[base_idx + 2] = make_float4(match.right_pos.x, match.right_pos.y, match.right_pos.z, 1.0f);
    // colors[base_idx + 2] = make_float4(0.0f, 0.0f, 1.0f, 1.0f); // Blue for right
}



// ============================================================= Bindings =================================================================

extern "C" {

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

    cuda_unregister_gl_buffers(detector_id);

    if (detector->d_keypoint_count) cudaFree(detector->d_keypoint_count);
    if (detector->d_descriptors) cudaFree(detector->d_descriptors);
    detector->d_keypoint_count = nullptr;
    detector->d_descriptors = nullptr;
}


int cuda_register_gl_texture(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return -1;

    // Allocate texture resource
    detector->y_texture = (CudaGLTextureResource*)malloc(sizeof(CudaGLTextureResource));
    if (!detector->y_texture) return -1;

    // Register the OpenGL texture with CUDA
    cudaError_t err = cudaGraphicsGLRegisterImage(
        &detector->y_texture->tex_resource,
        detector->gl_ytexture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard
    );
    
    if (err != cudaSuccess) {
        free(detector->y_texture);
        detector->y_texture = NULL;
        return -1;
    }

    detector->uv_texture = (CudaGLTextureResource*)malloc(sizeof(CudaGLTextureResource));
    if (!detector->uv_texture) return -1;

    // Register the OpenGL texture with CUDA
    err = cudaGraphicsGLRegisterImage(
        &detector->uv_texture->tex_resource,
        detector->gl_uvtexture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard
    );
    
    if (err != cudaSuccess) {
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


int cuda_register_gl_buffers(int detector_id, unsigned int position_buffer, unsigned int color_buffer, int max_keypoints) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return -1;

    cudaError_t error;

    // Check which CUDA device is currently in use
    int cudaDevice;
    error = cudaGetDevice(&cudaDevice);
    if (error != cudaSuccess) {
        printf("Error getting CUDA device: %s\n", cudaGetErrorString(error));
        return -1;
    }

    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, cudaDevice);
    if (error != cudaSuccess) {
        printf("Error getting CUDA device properties: %s\n", cudaGetErrorString(error));
        return -1;
    }

    printf("Using CUDA Device: %d - %s\n", cudaDevice, prop.name);

    // Register position buffer
    error = cudaGraphicsGLRegisterBuffer(
        &detector->gl_resources.position_resource,
        position_buffer,
        cudaGraphicsRegisterFlagsWriteDiscard
    );
    if (error != cudaSuccess) {
        printf("Error registering Position Buffer %d => %d\n", position_buffer, error);
        return -1;
    }

    // Register color buffer
    error = cudaGraphicsGLRegisterBuffer(
        &detector->gl_resources.color_resource,
        color_buffer,
        cudaGraphicsRegisterFlagsWriteDiscard
    );
    if (error != cudaSuccess) {
        cudaGraphicsUnregisterResource(detector->gl_resources.position_resource);
        printf("Error registering Color Buffer %d => %d\n", color_buffer, error);
        return -1;
    }

    detector->gl_resources.buffer_size = max_keypoints;
    return 0;
}


void cuda_unregister_gl_buffers(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return;

    if (detector->gl_resources.position_resource) {
        cudaGraphicsUnregisterResource(detector->gl_resources.position_resource);
    }
    if (detector->gl_resources.color_resource) {
        cudaGraphicsUnregisterResource(detector->gl_resources.color_resource);
    }
    detector->gl_resources = {};
}


int cuda_map_gl_resources(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return -1;

    // Map GL buffers for CUDA access
    cudaError_t error = cudaGraphicsMapResources(1, &detector->gl_resources.position_resource);
    if (error != cudaSuccess) {
        printf("Failed to Map Position Resources: %d\n", error);
        return -1;
    };

    error = cudaGraphicsMapResources(1, &detector->gl_resources.color_resource);
    if (error != cudaSuccess) {
        printf("Failed to Map Color Resources: %d\n", error);
        cudaGraphicsUnmapResources(1, &detector->gl_resources.position_resource);
        return -1;
    }
        
    size_t bytes;
    error = cudaGraphicsResourceGetMappedPointer(
        (void**)&detector->gl_resources.d_positions,
        &bytes,
        detector->gl_resources.position_resource
    );

    if (error != cudaSuccess) {
        printf("Failed to get Positions Mapped Pointer: %d\n", error);
        cuda_unmap_gl_resources(detector_id);
    }

    error = cudaGraphicsResourceGetMappedPointer(
        (void**)&detector->gl_resources.d_colors,
        &bytes,
        detector->gl_resources.color_resource
    );

    if (error != cudaSuccess) {
        printf("Failed to get Colors Mapped Pointer: %d\n", error);
        cuda_unmap_gl_resources(detector_id);
    }

    if (detector->y_texture) {
        cudaError_t err;
        
        err = cudaGraphicsMapResources(1, &detector->y_texture->tex_resource);
        if (err != cudaSuccess) return -1;
        
        err = cudaGraphicsSubResourceGetMappedArray(
            &detector->y_texture->array,
            detector->y_texture->tex_resource,
            0, 0
        );
        if (err != cudaSuccess) return -1;

        // Create surface object
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = detector->y_texture->array;

        // Ensure surface parameters match texture format (GL_R8)
        err = cudaCreateSurfaceObject(&detector->y_texture->surface, &res_desc);
        if (err != cudaSuccess) {
            printf("Failed to create surface object: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    if (detector->uv_texture) {
        cudaError_t err;
        
        err = cudaGraphicsMapResources(1, &detector->uv_texture->tex_resource);
        if (err != cudaSuccess) return -1;
        
        err = cudaGraphicsSubResourceGetMappedArray(
            &detector->uv_texture->array,
            detector->uv_texture->tex_resource,
            0, 0
        );
        if (err != cudaSuccess) return -1;

        // Create surface object
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = detector->uv_texture->array;

        // Ensure surface parameters match texture format (GL_R8)
        err = cudaCreateSurfaceObject(&detector->uv_texture->surface, &res_desc);
        if (err != cudaSuccess) {
            printf("Failed to create surface object: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    return error == cudaSuccess ? 0 : -1;
}

void cuda_unmap_gl_resources(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return;

    cudaGraphicsUnmapResources(1, &detector->gl_resources.position_resource);
    cudaGraphicsUnmapResources(1, &detector->gl_resources.color_resource);

    if (detector->y_texture) {
        cudaDestroySurfaceObject(detector->y_texture->surface);
        cudaGraphicsUnmapResources(1, &detector->y_texture->tex_resource);
    }

    if (detector->uv_texture) {
        cudaDestroySurfaceObject(detector->uv_texture->surface);
        cudaGraphicsUnmapResources(1, &detector->uv_texture->tex_resource);
    }
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
    
    // Apply vertical Gaussian blur
    gaussianBlurVertical<<<grid, block>>>(
        d_temp1,
        d_temp2,
        image->width,
        image->height,
        image->y_linesize
    );

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
        detector->gl_resources.d_positions,
        detector->gl_resources.d_colors,
        detector->d_descriptors,
        detector->d_keypoint_count,
        detector->gl_resources.buffer_size,
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
    DetectorInstance* left_detector = get_detector_instance(detector_id_left);
    DetectorInstance* right_detector = get_detector_instance(detector_id_right);
    DetectorInstance* combined_detector = get_detector_instance(detector_id_combined);

    if (!left_detector || !right_detector || !combined_detector) return -1;
    
    if (cuda_register_gl_texture(detector_id_left) < 0) {
        printf("Failed to register GL Texture for Left Detector!\n");
        return -1;
    }
    if (cuda_register_gl_texture(detector_id_right) < 0) {
        printf("Failed to register GL Texture for Combined Detector!\n");
        cuda_unregister_gl_texture(detector_id_left);
        return -1;
    }
    if (cuda_register_gl_texture(detector_id_combined) < 0) {
        printf("Failed to register GL Texture for Combined Detector!\n");
        cuda_unregister_gl_texture(detector_id_right);
        cuda_unregister_gl_texture(detector_id_left);
        return -1;
    }

    if (cuda_map_gl_resources(detector_id_left) < 0){
        printf("Failed to map GL Resources for Left Detector!\n");
        cuda_unregister_gl_texture(detector_id_left);
        cuda_unregister_gl_texture(detector_id_right);
        cuda_unregister_gl_texture(detector_id_combined);
        return -1;
    }  

    float sigma = 1.25f;

    printf("Getting keypoints from left...\n");
    float detection_time_left = cuda_detect_keypoints(
        detector_id_left,
        threshold,
        left,
        sigma
    );

    if (detection_time_left < 0){
        printf("Failed to detect keypoints from left image\n");
        cuda_unmap_gl_resources(detector_id_left);
        cuda_unregister_gl_texture(detector_id_left);
        cuda_unregister_gl_texture(detector_id_right);
        cuda_unregister_gl_texture(detector_id_combined);
        
        return -1;
    };

    if (cuda_map_gl_resources(detector_id_right) < 0){
        printf("Failed to map GL Resources for Right Detector!\n");
        cuda_unmap_gl_resources(detector_id_left);
        
        cuda_unregister_gl_texture(detector_id_left);
        cuda_unregister_gl_texture(detector_id_right);
        cuda_unregister_gl_texture(detector_id_combined);
        return -1;
    }

    printf("Getting keypoints from right...\n");
    float detection_time_right = cuda_detect_keypoints(
        detector_id_right,
        threshold,
        right,
        sigma
    );
   

    if (detection_time_right < 0){
        printf("Failed to detect keypoints from right image\n");
        cuda_unmap_gl_resources(detector_id_left);
        cuda_unmap_gl_resources(detector_id_right);
        
        cuda_unregister_gl_texture(detector_id_left);
        cuda_unregister_gl_texture(detector_id_right);
        cuda_unregister_gl_texture(detector_id_combined);
        return -1;
    };
    

    cudaDeviceSynchronize();   
    
    //Use left_detector as the basis for matching
    const int max_matches = min(*right->num_keypoints, *left->num_keypoints);

    printf("Left keypoints: %d\n", *left->num_keypoints);
    printf("Right Keypoints: %d\n", *right->num_keypoints);
    printf("Max matches allowed: %d\n", max_matches);

    dim3 blockCopyLeft(16,16);
    dim3 gridCopyLeft(
        (left->width + blockCopyLeft.x - 1) / blockCopyLeft.x, 
        (left->height + blockCopyLeft.y - 1) / blockCopyLeft.y
    );

    dim3 blockCopyRight(16,16);
    dim3 gridCopyRight(
        (left->width + blockCopyRight.x - 1) / blockCopyRight.x, 
        (left->height + blockCopyRight.y - 1) / blockCopyRight.y
    );


    if (max_matches <= 0) {
        printf("No matches possible! Returning...\n");
        cuda_unmap_gl_resources(detector_id_left);
        cuda_unmap_gl_resources(detector_id_right);

        setTextureKernel<<<gridCopyLeft, blockCopyLeft>>>(
            left_detector->y_texture->surface,
            left_detector->uv_texture->surface,
            left->y_plane,
            left->uv_plane,
            left->width,
            left->height,
            left->y_linesize,
            left->uv_linesize
        );

        setTextureKernel<<<gridCopyRight, blockCopyRight>>>(
            right_detector->y_texture->surface,
            right_detector->uv_texture->surface,
            right->y_plane,
            right->uv_plane,
            right->width,
            right->height,
            right->y_linesize,
            right->uv_linesize
        );

        cudaDeviceSynchronize();

        cuda_unregister_gl_texture(detector_id_left);
        cuda_unregister_gl_texture(detector_id_right);
        cuda_unregister_gl_texture(detector_id_combined);
        return 0;
    };
   
    // Allocate device memory for matches
    MatchedKeypoint* d_matches;
    int* d_match_count;
    
    cudaMalloc(&d_matches, max_matches * sizeof(MatchedKeypoint));
    cudaMalloc(&d_match_count, sizeof(int));
    cudaMemset(d_match_count, 0, sizeof(int));

    // Set up matching parameter
    float sensor_width_mm = 6.45f; // width of actual camera sensor in mm
    float baseline_world = (baseline / sensor_width_mm) * 6.4f;

    MatchingParams params = {
        .baseline = baseline_world,  // mm
        .focal_length = focal_length, // mm
        .max_disparity = 200.0f,     // pixels
        .epipolar_threshold = 75.0f,  // pixels
        .sensor_width_mm = sensor_width_mm,
        .sensor_width_pixels = 4608.0f,  // pixels
        .sensor_height_pixels = 2592.0f  // pixels
    };

    // Launch matching kernel
    dim3 blockMatching(512);
    dim3 gridMatching((max_matches + blockMatching.x - 1) / blockMatching.x); 

    printf("Matching Dims: Thread per Block: %d : Blocks: %d => Total Threads: %d\n", blockMatching.x, gridMatching.x, blockMatching.x * gridMatching.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t shared_mem_size = 
        (2 * sizeof(float) + sizeof(int)) * blockMatching.x + 
        sizeof(int) * *right->num_keypoints +                         
        sizeof(int);                                        
    
    cudaEventRecord(start);
    matchKeypointsKernel<<<gridMatching, blockMatching, shared_mem_size>>>(
        left_detector->gl_resources.d_positions,
        right_detector->gl_resources.d_positions,
        left_detector->d_descriptors,
        right_detector->d_descriptors,
        *left->num_keypoints,
        *right->num_keypoints,
        params,
        d_matches,
        d_match_count,
        max_matches,
        left_detector->y_texture->surface,
        combined_detector->y_texture->surface
    );
    cudaEventRecord(stop);

    cudaDeviceSynchronize();   
    cudaEventSynchronize(stop);
    float milliseconds_matching;
    cudaEventElapsedTime(&milliseconds_matching, start, stop);
    printf("Matching Kernel Execution Time (ms): %f\n", milliseconds_matching);


    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Matching Kernel failed: %s\n", cudaGetErrorString(error));
        cuda_unmap_gl_resources(detector_id_left);
        cuda_unmap_gl_resources(detector_id_right);
        
        cuda_unregister_gl_texture(detector_id_left);
        cuda_unregister_gl_texture(detector_id_right);
        cuda_unregister_gl_texture(detector_id_combined);
        
        cudaFree(d_matches);
        cudaFree(d_match_count);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }

    // Get match count
    cudaMemcpy(num_matches, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Detected Matches: %d\n", *num_matches);
    
    if (cuda_map_gl_resources(detector_id_combined) < 0){
        printf("Failed to map GL Resources for Combined Detector!\n");
        cuda_unmap_gl_resources(detector_id_left);
        cuda_unmap_gl_resources(detector_id_right);

        cuda_unregister_gl_texture(detector_id_left);
        cuda_unregister_gl_texture(detector_id_right);
        cuda_unregister_gl_texture(detector_id_combined);

        cudaFree(d_matches);
        cudaFree(d_match_count);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }

    setTextureKernel<<<gridCopyLeft, blockCopyLeft>>>(
        left_detector->y_texture->surface,
        left_detector->uv_texture->surface,
        left->y_plane,
        left->uv_plane,
        left->width,
        left->height,
        left->y_linesize,
        left->uv_linesize
    );

    setTextureKernel<<<gridCopyRight, blockCopyRight>>>(
        right_detector->y_texture->surface,
        right_detector->uv_texture->surface,
        right->y_plane,
        right->uv_plane,
        right->width,
        right->height,
        right->y_linesize,
        right->uv_linesize
    );

    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Texture set kernel failed: %s\n", cudaGetErrorString(err));
        cuda_unmap_gl_resources(detector_id_left);
        cuda_unmap_gl_resources(detector_id_right);
        cuda_unmap_gl_resources(detector_id_combined);
            
        cuda_unregister_gl_texture(detector_id_left);
        cuda_unregister_gl_texture(detector_id_right);
        cuda_unregister_gl_texture(detector_id_combined);

        cudaFree(d_matches);
        cudaFree(d_match_count);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    if (left_detector->y_texture && left_detector->uv_texture && combined_detector->y_texture && combined_detector->y_texture) {
        dim3 blockCopyCombined(16, 16);
        dim3 gridCopyCombined(
            (left->width + blockCopyCombined.x - 1) / blockCopyCombined.x,
            (left->height + blockCopyCombined.y - 1) / blockCopyCombined.y
        );
        
        printf("Copying surface from left to combined detector...\n");
        cudaEventRecord(start);
        copySurfaceKernel<<<gridCopyCombined, blockCopyCombined>>>(
            right_detector->y_texture->surface,     // source surface
            right_detector->uv_texture->surface,     // source surface
            combined_detector->y_texture->surface, // destination surface
            combined_detector->uv_texture->surface, // destination surface
            right->width,
            right->height
        );
        cudaEventRecord(stop);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Surface copy kernel failed: %s\n", cudaGetErrorString(err));
            cuda_unmap_gl_resources(detector_id_left);
            cuda_unmap_gl_resources(detector_id_right);
            cuda_unmap_gl_resources(detector_id_combined);
            
            cuda_unregister_gl_texture(detector_id_left);
            cuda_unregister_gl_texture(detector_id_right);
            cuda_unregister_gl_texture(detector_id_combined);

            cudaFree(d_matches);
            cudaFree(d_match_count);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return -1;
        }
        
        cudaEventSynchronize(stop);
    }
    
    float milliseconds_surface_cp = 0;
    cudaEventElapsedTime(&milliseconds_surface_cp, start, stop);
    printf("Surface Copy Kernel Execution Time (ms): %f\n", milliseconds_surface_cp);

    dim3 blockVis(1024);
    dim3 gridVis(((*num_matches) + blockVis.x - 1) / blockVis.x); 
    
    // Generate visualization
    cudaEventRecord(start);
    generateVisualizationKernel<<<gridVis, blockVis>>>(
        d_matches,
        *num_matches,
        combined_detector->gl_resources.d_positions,
        combined_detector->gl_resources.d_colors
    );
    cudaEventRecord(stop);

    cudaDeviceSynchronize();   
    cudaEventSynchronize(stop);
    float milliseconds_vis;
    cudaEventElapsedTime(&milliseconds_vis, start, stop);
    printf("Visualization Kernel Execution Time (ms): %f\n", milliseconds_vis);
    printf("Total Pipeline Execution (ms): %f\n", milliseconds_matching + milliseconds_vis + detection_time_left + detection_time_right + milliseconds_surface_cp);


    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Visualization Kernel failed: %s\n", cudaGetErrorString(error));
    }


    cudaFree(d_matches);
    cudaFree(d_match_count);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    
    cuda_unmap_gl_resources(detector_id_left);
    cuda_unmap_gl_resources(detector_id_right);
    cuda_unmap_gl_resources(detector_id_combined);

    cuda_unregister_gl_texture(detector_id_left);
    cuda_unregister_gl_texture(detector_id_right);
    cuda_unregister_gl_texture(detector_id_combined);

    glFinish();

    return 0;
}

}