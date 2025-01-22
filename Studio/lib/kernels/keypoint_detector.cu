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

__device__ float3 convertPxToCanvasCoords(float x, float y, float imageWidth, float imageHeight) {
    float normalizedX = (x / imageWidth) * 2.0f - 1.0f;
    float normalizedY = -((y / imageHeight) * 2.0f - 1.0f);
    
    float worldX = normalizedX * 6.4f;
    float worldZ = normalizedY * 3.6f;
    
    return make_float3(worldX, 0.01f, worldZ);
}

__device__ float2 convertCanvasToPxCoords(float x, float z, float imageWidth, float imageHeight) {
    float normX = x / 6.4f;
    float normZ = z / 3.6f;
    float pixelX = (normX + 1.0f) * 0.5f * imageWidth;
    float pixelY = (-normZ + 1.0f) * 0.5f * imageHeight; 
    return make_float2(pixelX, pixelY);
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
                float3 world_pos = convertPxToCanvasCoords(
                    kp.x, kp.y,
                    image_width, image_height
                );
                
                positions[global_idx + i] = make_float4(world_pos.x, world_pos.y, world_pos.z, 0.0f);
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
    float focal_length_px;
    float max_disparity;
    float epipolar_threshold;
    int image_width;
    int image_height;
};

struct BestMatch {
    int   bestIdx;   
    float bestCost;   
    //float secondBestCost; // for ratio tests
};

struct Keypoint {
    float3 position;   // Position in OpenGL world coordinates
    float2 image_coords;
    float disparity;    // Pixel disparity between left and right views
    float depth;
};

// Structure to hold matched keypoint data
struct MatchedKeypoint {
    float3 left_pos;
    float3 right_pos;
    Keypoint world;
};


__device__ Keypoint triangulatePosition(
    float3 leftWorldPos,
    float3 rightWorldPos,
    float baseline,
    float focal_length_px,  
    float image_width,
    float image_height
) {
    Keypoint result;
    
    float2 left_px_coords = convertCanvasToPxCoords(leftWorldPos.x, leftWorldPos.z, image_width, image_height);
    float2 right_px_coords = convertCanvasToPxCoords(rightWorldPos.x, rightWorldPos.z, image_width, image_height);

    float disparity_px = left_px_coords.x - right_px_coords.x;
   
    // Calculate depth using similar triangles principle
    float depth_mm = baseline * focal_length_px / disparity_px;  // 3.04f is focal length in mm

    float world_unit_per_mm = 100; // 1.0 world => 1.0m
    float depth_world = depth_mm / world_unit_per_mm;

    // Calculate final world position
    float worldX = leftWorldPos.x; 
    float worldY = depth_world;
    float worldZ = leftWorldPos.z;
        
    result.position = make_float3(worldX, worldY, worldZ);
    result.disparity = disparity_px;  // Store disparity in mm for debugging
    result.image_coords = make_float2(left_px_coords.x, left_px_coords.y);
    result.depth = depth_world;
    
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


__device__ float computeCost(
    const float4& leftPos,            
    const BRIEFDescriptor& leftDesc,  
    const float4& rightPos,           
    const BRIEFDescriptor& rightDesc, 
    const MatchingParams& params     
) {
    // 1. Epipolar difference
    float yDiff = fabsf(leftPos.y - rightPos.y);
    // Scale it by the epipolar threshold so that epipolarCost is roughly in [0..1]
    float epipolarCost = yDiff / params.epipolar_threshold;

    // 2. Disparity difference
    float disparity = leftPos.x - rightPos.x; 
    // If disparity <= 0 or beyond max, we give a large cost
    if (disparity <= 0.0f || disparity > params.max_disparity) {
        return 1e10f;  // A huge cost => effectively invalid match
    }
    float disparityCost = disparity / params.max_disparity; // Normalized [0..1]

    // 3. Descriptor distance
    int descDistance = hammingDistance(leftDesc, rightDesc);
    float descCost = (float)descDistance / 512.0f;


    float totalCost = 0.5f * epipolarCost 
                    + 0.3f * descCost
                    + 0.2f * disparityCost;

    return totalCost;
}

__global__ void matchLeftToRight(
    const float4* __restrict__ leftPositions,
    const BRIEFDescriptor* __restrict__ leftDescriptors,
    int leftCount,

    const float4* __restrict__ rightPositions,
    const BRIEFDescriptor* __restrict__ rightDescriptors,
    int rightCount,

    MatchingParams params,

    BestMatch* __restrict__ bestMatchesLeft  // [leftCount]
) {
    int leftIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leftIdx >= leftCount) return;

    float4 leftPos = leftPositions[leftIdx];
    BRIEFDescriptor leftDesc = leftDescriptors[leftIdx];

    float bestCost      = 1e9f;  // large sentinel
    float secondBestCost= 1e9f;  // for ratio test
    int   bestIdx       = -1;

    // Search all right keypoints (naive LxR approach)
    for (int rightIdx = 0; rightIdx < rightCount; rightIdx++) {
        float cost = computeCost(
            leftPos, leftDesc,
            rightPositions[rightIdx], rightDescriptors[rightIdx],
            params
        );
        if (cost < bestCost) {
            secondBestCost = bestCost;
            bestCost = cost;
            bestIdx  = rightIdx;
        }
        else if (cost < secondBestCost) {
            secondBestCost = cost;
        }
    }

    // Optionally apply ratio test or absolute threshold
    // e.g., if bestCost > SOME_THRESH or bestCost > 0.7f * secondBestCost => discard
    // but for cross-check, you can also do it after the cross-check step.
    
    bestMatchesLeft[leftIdx].bestIdx  = bestIdx;
    bestMatchesLeft[leftIdx].bestCost = bestCost;
    // bestMatchesLeft[leftIdx].secondBestCost = secondBestCost; // optional
}

__global__ void matchRightToLeft(
    const float4* __restrict__ rightPositions,
    const BRIEFDescriptor* __restrict__ rightDescriptors,
    int rightCount,

    const float4* __restrict__ leftPositions,
    const BRIEFDescriptor* __restrict__ leftDescriptors,
    int leftCount,

    MatchingParams params,

    BestMatch* __restrict__ bestMatchesRight  // [rightCount]
) {
    int rightIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rightIdx >= rightCount) return;

    float4 rightPos = rightPositions[rightIdx];
    BRIEFDescriptor rightDesc = rightDescriptors[rightIdx];

    float bestCost = 1e9f;
    float secondBestCost = 1e9f;
    int   bestIdx = -1;

    for (int leftIdx = 0; leftIdx < leftCount; leftIdx++) {
        float cost = computeCost(
            leftPositions[leftIdx], leftDescriptors[leftIdx],
            rightPos, rightDesc,
            params
        );
        if (cost < bestCost) {
            secondBestCost = bestCost;
            bestCost = cost;
            bestIdx  = leftIdx;
        }
        else if (cost < secondBestCost) {
            secondBestCost = cost;
        }
    }

    // optional ratio test or threshold
    bestMatchesRight[rightIdx].bestIdx  = bestIdx;
    bestMatchesRight[rightIdx].bestCost = bestCost;
}


__global__ void crossCheckMatches(
    const BestMatch* __restrict__ bestMatchesLeft,
    const BestMatch* __restrict__ bestMatchesRight,

    const float4* __restrict__ leftPositions,
    const float4* __restrict__ rightPositions,
    
    int leftCount,
    int rightCount,

    const MatchingParams params,

    // Output arrays:
    MatchedKeypoint* __restrict__ matchedPairs, 
    int* outCount
) {
    int leftIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leftIdx >= leftCount) return;

    // If left side found a best match
    int rightIdx = bestMatchesLeft[leftIdx].bestIdx;
    if (rightIdx < 0 || rightIdx >= rightCount) {
        return; // no valid match
    }

    // Check the reciprocal
    if (bestMatchesRight[rightIdx].bestIdx == leftIdx) {
        // We have a consistent match L_i <--> R_j
        int idx = atomicAdd(outCount, 1);
        
        float3 left_pos = make_float3(
            leftPositions[leftIdx].x,
            leftPositions[leftIdx].y,
            leftPositions[leftIdx].z
        );

        float3 right_pos = make_float3(
                rightPositions[rightIdx].x,
                rightPositions[rightIdx].y,
                rightPositions[rightIdx].z
        );

        Keypoint matchedPoint = triangulatePosition(
                left_pos,
                right_pos,
                params.baseline,
                params.focal_length_px,
                params.image_width,
                params.image_height
        );

        matchedPairs[idx] = {
            left_pos,
            right_pos,
            matchedPoint
        };
    }
}


// __global__ void matchKeypointsKernel(
//     const float4* __restrict__ left_positions,
//     const float4* __restrict__ right_positions,
//     const BRIEFDescriptor* __restrict__ left_descriptors,
//     const BRIEFDescriptor* __restrict__ right_descriptors,
//     const int left_count,
//     const int right_count,
//     const MatchingParams params,
//     MatchedKeypoint* matches,
//     int* match_count,
//     const int max_matches,
//     cudaSurfaceObject_t source_tex,      // Y plane texture from left image
//     cudaSurfaceObject_t combined_tex     // Target texture for visualization

// ) {
//     extern __shared__ float shared_mem[];
    
//     // Split shared memory into different arrays
//     float* min_costs = shared_mem;
//     float* second_min_costs = &min_costs[blockDim.x];
//     int* best_matches = (int*)&second_min_costs[blockDim.x];
//     int* used_right_points = &best_matches[blockDim.x];
//     int* shared_count = (int*)&used_right_points[right_count];

//     const int left_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
//     // Initialize shared memory
//     if (threadIdx.x < blockDim.x) {
//         min_costs[threadIdx.x] = INFINITY;
//         second_min_costs[threadIdx.x] = INFINITY;
//         best_matches[threadIdx.x] = -1;
//     }
    
//     if (threadIdx.x < right_count) {
//         used_right_points[threadIdx.x] = 0;
//     }
    
//     if (threadIdx.x == 0) {
//         shared_count[0] = 0;
//     }

//     __syncthreads();

//     if (left_idx >= left_count) return;
    
//     float3 left_pos = make_float3(
//         left_positions[left_idx].x,
//         left_positions[left_idx].y,
//         left_positions[left_idx].z
//     );

//     BRIEFDescriptor left_desc = left_descriptors[left_idx];

//     float best_desc_distance = INFINITY;
//     float best_y_diff = INFINITY;
//     float best_disparity = INFINITY;


//     const int MAX_DESC_DISTANCE = 300.0f; 
    
//     // Find best and second-best matches for this left keypoint
//     for (int right_idx = 0; right_idx < right_count; right_idx++) {
//         float3 right_pos = make_float3(
//             right_positions[right_idx].x,
//             right_positions[right_idx].y,
//             right_positions[right_idx].z
//         );

//         // Check epipolar constraint (y-difference)
//         float y_diff = fabsf(left_pos.y - right_pos.y);
//         float disparity = left_pos.x - right_pos.x;

//         if (y_diff > params.epipolar_threshold || 
//             disparity <= 0 || 
//             disparity > params.max_disparity) {
//             continue;
//         }
      
//         // Calculate descriptor distance
//         int desc_distance = hammingDistance(left_desc, right_descriptors[right_idx]);
//         if (desc_distance > MAX_DESC_DISTANCE) continue;
      
//         // Calculate individual costs
//         float desc_cost = desc_distance / 512.0f;
//         float epipolar_cost = y_diff / params.epipolar_threshold;
//         float disparity_cost = disparity / params.max_disparity;

//         // Weighted combination of costs
//         float total_cost = epipolar_cost * 0.3f +
//                           desc_cost * 0.5f +
//                           disparity_cost * 0.2f;

//         // Maintain best and second-best matches
//         if (total_cost < min_costs[threadIdx.x]) {
//             best_desc_distance = desc_distance;
//             best_y_diff = y_diff;
//             best_disparity = disparity;
            
//             second_min_costs[threadIdx.x] = min_costs[threadIdx.x];
//             min_costs[threadIdx.x] = total_cost;
//             best_matches[threadIdx.x] = right_idx;
//         } else if (total_cost < second_min_costs[threadIdx.x]) {
//             second_min_costs[threadIdx.x] = total_cost;
//         }
//     }

//     __syncthreads();

//     // Apply Lowe's ratio test and absolute threshold
//     const float RATIO_THRESH = 0.9f; 
//     const float ABS_COST_THRESH = 0.5f;
    
//     bool is_good_match = false;
//     int right_idx = best_matches[threadIdx.x];

//     if (right_idx >= 0) {
//         bool passes_thresholds = (min_costs[threadIdx.x] < ABS_COST_THRESH) &&
//                                 (min_costs[threadIdx.x] < second_min_costs[threadIdx.x] * RATIO_THRESH);
        
//         bool passes_validation = (best_y_diff < params.epipolar_threshold * 0.5f) &&
//                                 (best_disparity > 1.0f) &&
//                                 (best_desc_distance < MAX_DESC_DISTANCE * 0.75f);
        
//         is_good_match = passes_thresholds && passes_validation;
        
//         // Try to claim the right point
//         if (is_good_match && right_idx < right_count) {
//             is_good_match = (atomicCAS(&used_right_points[right_idx], 0, 1) == 0);
//             if (is_good_match) {
//                 atomicAdd(shared_count, 1);
//             }
//         }
//     }

//      __syncthreads();

//     // Update global match count
//     if (threadIdx.x == 0 && shared_count[0] > 0) {
//         atomicAdd(match_count, min(shared_count[0], max_matches));
//     }

//     __syncthreads();

//     // Store matches, respecting the max_matches limit
//     if (is_good_match) {
//         int match_idx = atomicSub(shared_count, 1) - 1;
//         if (match_idx >= 0 && match_idx < max_matches) {
//             float3 right_pos = make_float3(
//                 right_positions[right_idx].x,
//                 right_positions[right_idx].y,
//                 right_positions[right_idx].z
//             );

//             Keypoint matchedPoint = triangulatePosition(
//                 left_pos,
//                 right_pos,
//                 params.baseline,
//                 params.focal_length_px,
//                 params.image_width,
//                 params.image_height
//             );

//             matches[match_idx] = {
//                 left_pos,
//                 right_pos,
//                 matchedPoint
//             };
//         }
//     }
// }


__global__ void setTextureKernel(
    cudaSurfaceObject_t y_surface,
    cudaSurfaceObject_t uv_surface,
    cudaSurfaceObject_t depth_surface,
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

    surf2Dwrite((float)0, depth_surface, x * sizeof(float), y);
}

__global__ void copySurfaceKernel(
    cudaSurfaceObject_t srcSurf_y,
    cudaSurfaceObject_t srcSurf_uv,
    cudaSurfaceObject_t dstSurf_y,
    cudaSurfaceObject_t dstSurf_uv,
    cudaSurfaceObject_t dstSurf_depth,
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
    
    surf2Dwrite((float)0, dstSurf_depth, x * sizeof(float), y);
}


__global__ void set_distance_texture(
    cudaSurfaceObject_t depth_texture,
    MatchedKeypoint* matches,
    int num_matches,
    int width,
    int height
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    MatchedKeypoint nearest;
    float current_min;

    for (int i = 0; i < num_matches; i++){
        MatchedKeypoint current_match = matches[i]; 
        float2 current_keypoint = current_match.world.image_coords;
        float dist = pow(current_keypoint.x - x, 2) + pow(current_keypoint.y - y, 2);
        
        if (i == 0) {
            current_min = dist;
            nearest  = current_match;
        }
        else if (dist < current_min) { 
            current_min = dist;
            nearest = current_match;
        }
    }

    surf2Dwrite(nearest.world.position.y, depth_texture, x * sizeof(float), y);
}


__device__ float4 generate_unique_color(int idx, float alpha) {
    // Use a simple hash function to generate pseudo-random but consistent colors
    const float golden_ratio = 0.618033988749895f;
    float hue = fmodf(idx * golden_ratio, 1.0f);
    
    // Convert HSV to RGB (simplified, assuming S=V=1)
    float h = hue * 6.0f;
    float x = 1.0f - fabsf(fmodf(h, 2.0f) - 1.0f);
    
    float r = 0.0f, g = 0.0f, b = 0.0f;
    
    if      (h < 1.0f) { r = 1.0f; g = x; }
    else if (h < 2.0f) { r = x; g = 1.0f; }
    else if (h < 3.0f) { g = 1.0f; b = x; }
    else if (h < 4.0f) { g = x; b = 1.0f; }
    else if (h < 5.0f) { r = x; b = 1.0f; }
    else               { r = 1.0f; b = x; }
    
    return make_float4(r, g, b, alpha);
}



__device__ float4 transform_point(const float* transform, float4 point) {
    // Ensure w component is 1.0 for proper transformation
    point.w = 1.0f;
    
    float4 result;
    // Column-major matrix multiplication 
    result.x = transform[0] * point.x + transform[4] * point.y + 
               transform[8] * point.z + transform[12] * point.w;
    result.y = transform[1] * point.x + transform[5] * point.y + 
               transform[9] * point.z + transform[13] * point.w;
    result.z = transform[2] * point.x + transform[6] * point.y + 
               transform[10] * point.z + transform[14] * point.w;
    result.w = transform[3] * point.x + transform[7] * point.y + 
               transform[11] * point.z + transform[15] * point.w;
    
    if (result.w != 0.0f) {
        result.x /= result.w;
        result.y /= result.w;
        result.z /= result.w;
        result.w = 1.0f;
    }

    return result;
}



// Kernel to generate visualization data
__global__ void visualizeTriangulation(
    const MatchedKeypoint* matches,
    const int match_count,
    
    float4*  keypoint_positions,
    float4*  keypoint_colors,
    
    float4*  left_line_positions,
    float4* left_line_colors,
    
    float4* __restrict__ right_line_positions,
    float4* __restrict__ right_line_colors,

    const float* left_transform,
    const float* right_transform
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= match_count) return;

    const MatchedKeypoint match = matches[idx];
    
    float4 match_color = generate_unique_color(idx, 0.5f);
    
    // Transform the world position
    keypoint_positions[idx] = make_float4(match.world.position.x, match.world.position.y, match.world.position.z, 0.0f);
    keypoint_colors[idx] = make_float4(1.0f, 0.0f, 1.0f, 1.0f);

    // Transform left keypoint from canvas space to world space
    float4 left_world_pos = transform_point(left_transform, 
        make_float4(match.left_pos.x, match.left_pos.y, match.left_pos.z, 0.0f));
    // Transform right keypoint from canvas space to world space
    float4 right_world_pos = transform_point(right_transform, 
        make_float4(match.right_pos.x, match.right_pos.y, match.right_pos.z, 0.0f));


    // Set line positions
    left_line_positions[idx * 2] = make_float4(left_world_pos.x, left_world_pos.y,left_world_pos.z, 0.0f);
    left_line_positions[idx * 2 + 1] = keypoint_positions[idx];
    left_line_colors[idx] = match_color;
    
    right_line_positions[idx * 2] = make_float4(right_world_pos.x, right_world_pos.y,right_world_pos.z, 0.0f);;
    right_line_positions[idx * 2 + 1] = keypoint_positions[idx];
    right_line_colors[idx] = match_color;

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

int cuda_create_detector(int max_keypoints, int gl_ytexture, int gl_uvtexture, int gl_depthtexture, const float transform[16]) {
    int slot = find_free_detector_slot();
    if (slot < 0) {
        return -1;
    }    

    memset(&g_detectors[slot], 0, sizeof(DetectorInstance));

    if (cudaMalloc(&g_detectors[slot].d_keypoint_count, sizeof(int)) != cudaSuccess) {
        return -1;
    }

    if (cudaMalloc(&g_detectors[slot].d_descriptors, max_keypoints * sizeof(BRIEFDescriptor)) != cudaSuccess) {
        return -1;
    } ;

    cudaError_t error = cudaMalloc((void**)&g_detectors[slot].d_world_transform, 16 * sizeof(float));
    if (error != cudaSuccess) {
        printf("Transform allocation failed: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    // Initialize transform to identity matrix
    float identity[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    error = cudaMemcpy(g_detectors[slot].d_world_transform, identity, 16 * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Initial transform copy failed: %s\n", cudaGetErrorString(error));
        return -1;
    }

    g_detectors[slot].initialized = true;
    g_detectors[slot].id = g_next_detector_id++;
    g_detectors[slot].gl_ytexture = gl_ytexture;
    g_detectors[slot].gl_uvtexture = gl_uvtexture;
    g_detectors[slot].gl_depthtexture = gl_depthtexture;


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

    if (detector->d_world_transform) {
        cudaFree(detector->d_world_transform);
        detector->d_world_transform = nullptr;
    }
}

int cuda_map_transformation(int detector_id,  const float transformation[16]) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return -1;


    memcpy(detector->world_transform, transformation, 16 * sizeof(float));
    cudaError_t error = cudaMemcpy(detector->d_world_transform, detector->world_transform, 
                                  16 * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Transform copy to device failed: %s\n", cudaGetErrorString(error));
        return -1;
    }

    return 0;
}


int cuda_register_gl_texture(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return -1;

   
   
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

    
    // Allocate Depth texture resource
    detector->depth_texture = (CudaGLTextureResource*)malloc(sizeof(CudaGLTextureResource));
    if (!detector->depth_texture) {
        printf("Failed to allocate Depth texture resource\n");
        cudaGraphicsUnregisterResource(detector->y_texture->tex_resource);
        free(detector->y_texture);
        detector->y_texture = NULL;
        cudaGraphicsUnregisterResource(detector->uv_texture->tex_resource);
        free(detector->uv_texture);
        detector->uv_texture = NULL;
        return -1;
    }
    memset(detector->depth_texture, 0, sizeof(CudaGLTextureResource));

    // Register Depth texture
    err = cudaGraphicsGLRegisterImage(
        &detector->depth_texture->tex_resource,
        detector->gl_depthtexture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard
    );
    
    if (err != cudaSuccess) {
        printf("Failed to register UV texture: %s\n", cudaGetErrorString(err));
        cudaGraphicsUnregisterResource(detector->y_texture->tex_resource);
        free(detector->y_texture);
        detector->y_texture = NULL;
        cudaGraphicsUnregisterResource(detector->uv_texture->tex_resource);
        free(detector->uv_texture);
        detector->uv_texture = NULL;
        free(detector->depth_texture);
        detector->depth_texture = NULL;
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

    if (detector->depth_texture){
        cudaGraphicsUnregisterResource(detector->depth_texture->tex_resource);
        free(detector->depth_texture);
        detector->depth_texture = NULL;
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
        &detector->gl_resources.keypoints,
        keypoint_position_buffer, 
        keypoint_color_buffer,
        buffer_size
    );
    if (error < 0) return error;

    if (left_position_buffer != NULL && left_color_buffer != NULL) {
        error = cuda_register_instance_buffer(
            &detector->gl_resources.connections.left,
            *left_position_buffer, 
            *left_color_buffer,
            buffer_size * 2
        );

        if (error < 0) return error;
    }

    if (right_position_buffer != NULL && right_color_buffer != NULL) {
        error = cuda_register_instance_buffer(
            &detector->gl_resources.connections.right,
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

// int cuda_map_texture_resources(CudaGLTextureResource* texture){
//     if (texture) {
//         cudaError_t err = cudaGraphicsMapResources(1, &texture->tex_resource);
//         if (err != cudaSuccess) {
//             printf("Failed to map texture resource: %s\n", cudaGetErrorString(err));
//             return -1;
//         }

//         err = cudaGraphicsSubResourceGetMappedArray(&texture->array,
//                                                     texture->tex_resource,
//                                                     0, 0);
//         if (err != cudaSuccess) {
//             printf("Failed to get mapped array for texture: %s\n", cudaGetErrorString(err));
//             return -1;
//         }

//         cudaResourceDesc res_desc = {};
//         res_desc.resType = cudaResourceTypeArray;
//         res_desc.res.array.array = texture->array;

//         err = cudaCreateSurfaceObject(&texture->surface, &res_desc);
//         if (err != cudaSuccess) {
//             printf("Failed to create surface object: %s\n", cudaGetErrorString(err));
//             return -1;
//         }
//     }
// }


int cuda_map_resources(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return -1;   
    
    int error = cuda_map_buffer_resources(&detector->gl_resources.keypoints);
    if (error < 0) printf("Failed to map keypoint buffers!\n");

    if (detector->gl_resources.connections.left.position_resource){
        error |= cuda_map_buffer_resources(&detector->gl_resources.connections.left);
        if (error < 0)  printf("Failed to map left connection buffers!\n");
    }
    if (detector->gl_resources.connections.right.position_resource){
        error |= cuda_map_buffer_resources(&detector->gl_resources.connections.right);
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

    if (detector->depth_texture) {
        cudaError_t err = cudaGraphicsMapResources(1, &detector->depth_texture->tex_resource);
        if (err != cudaSuccess) {
            printf("Failed to map Depth texture resource: %s\n", cudaGetErrorString(err));
            cuda_unmap_resources(detector_id);
            return -1;
        }

        err = cudaGraphicsSubResourceGetMappedArray(&detector->depth_texture->array,
                                                  detector->depth_texture->tex_resource,
                                                  0, 0);
        if (err != cudaSuccess) {
            printf("Failed to get mapped array for Depth texture: %s\n", cudaGetErrorString(err));
            cuda_unmap_resources(detector_id);
            return -1;
        }

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = detector->depth_texture->array;

        err = cudaCreateSurfaceObject(&detector->depth_texture->surface, &res_desc);
        if (err != cudaSuccess) {
            printf("Failed to create Depth surface object: %s\n", cudaGetErrorString(err));
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

    if (detector->depth_texture) {
        cudaDestroySurfaceObject(detector->depth_texture->surface);
        cudaGraphicsUnmapResources(1, &detector->depth_texture->tex_resource);
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



// TODO: Need to set matches to position and colors of combined keypoints
// Not sure how combined is currently being visualized
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

    MatchingParams params = {
        .baseline = baseline,
        .focal_length_px = focal_length * (left->width / sensor_width_mm),
        .max_disparity = 100.0f,
        .epipolar_threshold = 10.0f,
        .image_width = left->width,
        .image_height = left->height
    };


    // Configure kernel launch parameters
    // dim3 blockMatching(512);
    // dim3 gridMatching((max_matches + blockMatching.x - 1) / blockMatching.x);

    // size_t shared_mem_size = 
    //     (2 * sizeof(float) + sizeof(int)) * blockMatching.x + // min_costs, second_min_costs, best_matches
    //     sizeof(int) * max_matches +                          // used_right_points
    //     sizeof(int);                                         // shared_count

    // Execute matching kernel
    // cudaEventRecord(start);
    // matchKeypointsKernel<<<gridMatching, blockMatching, shared_mem_size>>>(
    //     left_detector->gl_resources.keypoints.d_positions,
    //     right_detector->gl_resources.keypoints.d_positions,
    //     left_detector->d_descriptors,
    //     right_detector->d_descriptors,
    //     max_matches,
    //     max_matches,
    //     params,
    //     d_matches,
    //     d_match_count,
    //     max_matches,
    //     left_detector->y_texture->surface,
    //     combined_detector->y_texture->surface
    // );
    // cudaEventRecord(stop);

    // error = cudaGetLastError();
    // if (error != cudaSuccess) {
    //     printf("Matching kernel failed: %s\n", cudaGetErrorString(error));
    //     return -1.0;
    // }
    // cudaEventSynchronize(stop);

    // float milliseconds_matching;
    // cudaEventElapsedTime(&milliseconds_matching, start, stop);
    // printf("Matching Kernel Time: %.2f ms\n", milliseconds_matching);

    // if (*num_matches == 0) {
    //     printf("No Matches detected!\n");
    //     return milliseconds_matching;
    // }

    // // Get match count
    // cudaError_t error;
    // error = cudaMemcpy(num_matches, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);
    // if (error != cudaSuccess) {
    //     printf("Failed to copy match count: %s\n", cudaGetErrorString(error));
    //     return -1.0;
    // }

    // printf("num_matches: %d\n", *num_matches);

    BestMatch* d_bestMatchesLeft  = nullptr;
    BestMatch* d_bestMatchesRight = nullptr;

    cudaMalloc(&d_bestMatchesLeft,  *left->num_keypoints  * sizeof(BestMatch));
    cudaMalloc(&d_bestMatchesRight, *right->num_keypoints * sizeof(BestMatch));
    
    float milliseconds_left_right;
    float milliseconds_right_left;
    float milliseconds_cross_check;
    
    {
        dim3 block(256);
        dim3 grid( (*left->num_keypoints + block.x - 1)/block.x );
        cudaEventRecord(start);
        matchLeftToRight<<<grid, block>>>(
            left_detector->gl_resources.keypoints.d_positions, left_detector->d_descriptors, *left->num_keypoints,
            right_detector->gl_resources.keypoints.d_positions, right_detector->d_descriptors, *right->num_keypoints,
            params,
            d_bestMatchesLeft
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds_left_right, start, stop);
        printf("Matching Time Left => Right: %.2f ms\n", milliseconds_left_right);
    }

    {
        dim3 block(256);
        dim3 grid( (*right->num_keypoints + block.x - 1)/block.x );
        cudaEventRecord(start);
        matchRightToLeft<<<grid, block>>>(
            right_detector->gl_resources.keypoints.d_positions, right_detector->d_descriptors, *right->num_keypoints,
            left_detector->gl_resources.keypoints.d_positions, left_detector->d_descriptors, *left->num_keypoints,
            params,
            d_bestMatchesRight
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds_right_left, start, stop);
        printf("Matching Time Right => Left: %.2f ms\n", milliseconds_right_left);
    }


    {
        dim3 block(256);
        dim3 grid( (*left->num_keypoints + block.x - 1)/block.x );
        cudaEventRecord(start);
        crossCheckMatches<<<grid, block>>>(
            d_bestMatchesLeft,
            d_bestMatchesRight,
            left_detector->gl_resources.keypoints.d_positions,
            right_detector->gl_resources.keypoints.d_positions,
            *left->num_keypoints,
            *right->num_keypoints,
            params,
            d_matches,
            d_match_count
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds_cross_check, start, stop);
        printf("Cross Checking Matches Time: %.2f ms\n", milliseconds_cross_check);
    }

    cudaMemcpy(num_matches, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("num_matches: %d\n", *num_matches);

    if (*num_matches == 0) {
        printf("No Matches detected!\n");
        return milliseconds_left_right + milliseconds_right_left + milliseconds_cross_check;
    }

    // Generate visualization
    dim3 blockVis(512);
    dim3 gridVis((*num_matches + blockVis.x - 1) / blockVis.x);

    cudaEventRecord(start);
    visualizeTriangulation<<<gridVis, blockVis>>>(
        d_matches,
        *num_matches,
        combined_detector->gl_resources.keypoints.d_positions,
        combined_detector->gl_resources.keypoints.d_colors,
        combined_detector->gl_resources.connections.left.d_positions,
        combined_detector->gl_resources.connections.left.d_colors,
        combined_detector->gl_resources.connections.right.d_positions,
        combined_detector->gl_resources.connections.right.d_colors,
        left_detector->d_world_transform,
        right_detector->d_world_transform
    );
    cudaEventRecord(stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Visualization kernel failed: %s\n", cudaGetErrorString(error));
        return -1.0;
    }

    cudaEventSynchronize(stop);
    float milliseconds_vis;
    cudaEventElapsedTime(&milliseconds_vis, start, stop);
    printf("Visualization time: %.2f ms\n", milliseconds_vis);

    dim3 blockCopy(16, 16);
    dim3 gridCopy(
        (left->width + blockCopy.x - 1) / blockCopy.x,
        (left->height + blockCopy.y - 1) / blockCopy.y
    );

    cudaEventRecord(start);
    set_distance_texture<<<gridCopy, blockCopy>>>(
        combined_detector->depth_texture->surface,
        d_matches,
        *num_matches,
        left->image_width,
        left->image_height
    );
    cudaEventRecord(stop);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Visualization kernel failed: %s\n", cudaGetErrorString(error));
        return -1.0;
    }

    cudaEventSynchronize(stop);
    float milliseconds_segmentation;
    cudaEventElapsedTime(&milliseconds_segmentation, start, stop);
    printf("Assigning Nearest keypoint Distance time: %.2f ms\n", milliseconds_segmentation);

    cudaFree(d_bestMatchesLeft);
    cudaFree(d_bestMatchesRight);
 
    return milliseconds_left_right + milliseconds_right_left + milliseconds_cross_check + milliseconds_vis + milliseconds_segmentation;
    // return milliseconds_vis + milliseconds_matching + milliseconds_segmentation;
    
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
    float sigma = 1.0f;

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

    // Allocate resources for matching
    MatchedKeypoint* d_matches = NULL;
    int* d_match_count = NULL;
    cudaEvent_t start = NULL, stop = NULL;


    dim3 blockCopy(16, 16);
    dim3 gridCopy(
        (left->width + blockCopy.x - 1) / blockCopy.x,
        (left->height + blockCopy.y - 1) / blockCopy.y
    );

    // Set texture for left and right images
    setTextureKernel<<<gridCopy, blockCopy>>>(
        left_detector->y_texture->surface,
        left_detector->uv_texture->surface,
        left_detector->depth_texture->surface,
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
        right_detector->depth_texture->surface,
        right->y_plane,
        right->uv_plane,
        right->width,
        right->height,
        right->y_linesize,
        right->uv_linesize
    );

     // Copy surface from left to combined detector
    copySurfaceKernel<<<gridCopy, blockCopy>>>(
        left_detector->y_texture->surface,
        left_detector->uv_texture->surface,
        combined_detector->y_texture->surface,
        combined_detector->uv_texture->surface,
        combined_detector->depth_texture->surface,
        left->width,
        left->height
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

    printf("Detected Left: %d <====> Detected Right: %d\n", *left->num_keypoints, *right->num_keypoints);

    if (detection_time_left < 0 || detection_time_right < 0) {
        cleanup_resources(detector_id_left, detector_id_right, detector_id_combined, NULL, NULL, NULL, NULL);
        return -1;
    }

    
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