#include "kernels.h"
#include <float.h>
#include <curand_kernel.h>

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


__device__ float computeCornerScore(
    const uint8_t* __restrict__ y_plane,
    int x, int y, 
    int linesize,
    uint8_t center,
    uint8_t threshold
) {
    float score = 0.0f;
    float darker_score = 0.0f;
    float brighter_score = 0.0f;
    
    // Calculate weighted score for both brighter and darker pixels
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        const int2 offset = fast_offsets[i];
        const uint8_t pixel = y_plane[(y + offset.y) * linesize + (x + offset.x)];
        
        // Weight based on distance from center
        float dist = sqrtf(float(offset.x * offset.x + offset.y * offset.y));
        float weight = expf(-dist / 3.0f);  // Exponential falloff
        
        if (pixel > center + threshold) {
            brighter_score += weight * float(pixel - center);
        } 
        else if (pixel < center - threshold) {
            darker_score += weight * float(center - pixel);
        }
    }
    
    // Use maximum of brighter or darker score
    score = fmaxf(brighter_score, darker_score);
    
    return score;
}

__device__ bool isLocalMaximum(
    const uint8_t* __restrict__ y_plane,
    int x, int y,
    int linesize,
    float score,
    uint8_t threshold,
    int width,
    int height
) {
    const int window_size = 3;  // Increased window size for more stable maxima
    
    // Check larger neighborhood for more stable maxima
    for (int dy = -window_size; dy <= window_size; dy++) {
        for (int dx = -window_size; dx <= window_size; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            const int nx = x + dx;
            const int ny = y + dy;
            
            if (nx >= 3 && nx < width - 3 && ny >= 3 && ny < height - 3) {
                const uint8_t neighbor_center = y_plane[ny * linesize + nx];
                float neighbor_score = computeCornerScore(
                    y_plane, nx, ny, linesize, 
                    neighbor_center, threshold
                );
                
                if (neighbor_score >= score) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Modified FAST detection kernel incorporating the new scoring:
__global__ void detectFASTKeypoints(
    const uint8_t* __restrict__ y_plane,
    int width,
    int height,
    int linesize,
    uint8_t threshold,
    uint32_t arc_length,
    float4* positions,
    float4* colors,
    BRIEFDescriptor* descriptors,
    uint* keypoint_count,
    int max_keypoints
) {
    __shared__ int block_counter;
    __shared__ float2 block_keypoints[256]; 
    __shared__ BRIEFDescriptor block_descriptors[256];
    __shared__ float block_scores[256];
    
    // Initialize block counter
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        block_counter = 0;
    }
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

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
    bool is_keypoint = (max_consecutive_brighter >= arc_length || max_consecutive_darker >= arc_length);

    if (is_keypoint) {
        // Compute corner score
        float score = computeCornerScore(y_plane, x, y, linesize, center, threshold);
        
        // Non-maximum suppression with new scoring
        if (isLocalMaximum(y_plane, x, y, linesize, score, threshold, width, height)) {
            BRIEFDescriptor desc = {0};  // Initialize all bits to 0
            
            #pragma unroll
            for (int i = 0; i < 512; i += 2) {
                int2 offset1 = brief_pattern[i];
                int2 offset2 = brief_pattern[i+1];
                
                uint8_t p1 = y_plane[(y + offset1.y) * linesize + (x + offset1.x)];
                uint8_t p2 = y_plane[(y + offset2.y) * linesize + (x + offset2.x)];
                
                if (p1 > p2) {
                    int bitIdx = i / 2;
                    desc.descriptor[bitIdx / 64] |= (1ULL << (bitIdx % 64));
                }
            }

            int local_idx = atomicAdd(&block_counter, 1);
            if (local_idx < 256) {
                block_keypoints[local_idx] = make_float2(x, y);
                block_descriptors[local_idx] = desc;
                block_scores[local_idx] = score;
            }
        }
    }
    
    __syncthreads();
    

    // Store keypoints globally with score-based filtering
    if (threadIdx.x == 0 && threadIdx.y == 0 && block_counter > 0) {
        int global_idx = atomicAdd(keypoint_count, block_counter);
        
        if (global_idx + block_counter <= max_keypoints) {
            for (int i = 0; i < block_counter; i++) {
                float2 kp = block_keypoints[i];
                float3 world_pos = convertPxToCanvasCoords(
                    kp.x, kp.y,
                    width, height
                );
                
                positions[global_idx + i] = make_float4(world_pos.x, world_pos.y, world_pos.z, 0.0f);
                colors[global_idx + i] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
                descriptors[global_idx + i] = block_descriptors[i];
            }
        }
    }
}


// ============================================================= Matching =================================================================

__device__ Match triangulatePosition(
    float3 leftWorldPos,
    float3 rightWorldPos,
    float baseline,
    float focal_length_px,  
    float image_width,
    float image_height
) {
    Match result;
    
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
    const float4 &leftPos,            
    const BRIEFDescriptor &leftDesc,  
    const float4 &rightPos,           
    const BRIEFDescriptor &rightDesc, 
    const StereoParams &params     
) {
    // 1. Epipolar difference
    float yDiff = fabsf(leftPos.z - rightPos.z);
    
    if (yDiff > params.epipolar_threshold) {
        return 1e10f;
    }

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

    if (descCost > params.max_hamming_dist) {
        return 1e10f;
    }

    float totalCost = params.epipolar_weight * epipolarCost 
                    + params.hamming_dist_weight * descCost
                    + params.disparity_weight * disparityCost;

    return totalCost;
}

__global__ void matchLeftToRight(
    const float4* __restrict__ leftPositions,
    const BRIEFDescriptor* __restrict__ leftDescriptors,
    int leftCount,

    const float4* __restrict__ rightPositions,
    const BRIEFDescriptor* __restrict__ rightDescriptors,
    int rightCount,

    StereoParams params,

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
    
    const float COST_THRESHOLD = params.cost_threshold;  // Reject matches with high absolute cost
    const float RATIO_THRESHOLD = params.lowes_ratio; // Lowe's ratio test threshold

    if (bestCost > COST_THRESHOLD || 
        (secondBestCost < 1e9f && bestCost > RATIO_THRESHOLD * secondBestCost)) {
        bestIdx = -1;  // Mark as invalid match
        bestCost = 1e9f;
    }
    
    bestMatchesLeft[leftIdx].bestIdx  = bestIdx;
    bestMatchesLeft[leftIdx].bestCost = bestCost;
}

__global__ void matchRightToLeft(
    const float4* __restrict__ rightPositions,
    const BRIEFDescriptor* __restrict__ rightDescriptors,
    uint rightCount,

    const float4* __restrict__ leftPositions,
    const BRIEFDescriptor* __restrict__ leftDescriptors,
    uint leftCount,

    StereoParams params,

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
    
    uint leftCount,
    uint rightCount,

    const StereoParams params,

    // Output arrays:
    MatchedKeypoint* __restrict__ matchedPairs, 
    uint* outCount
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

        Match matchedPoint = triangulatePosition(
                left_pos,
                right_pos,
                params.baseline_mm,
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
    cudaSurfaceObject_t srcSurf_depth, 
    cudaSurfaceObject_t dstSurf_y,
    cudaSurfaceObject_t dstSurf_uv,
    cudaSurfaceObject_t dstSurf_depth,
    int width,
    int height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    // Copy Y plane
    unsigned char pixel;
    surf2Dread(&pixel, srcSurf_y, x, y);
    surf2Dwrite(pixel, dstSurf_y, x * sizeof(u_char), y);

    // Copy UV plane
    if (x < width / 2 && y < height / 2) { 
        uchar2 uv;
        surf2Dread(&uv, srcSurf_uv, x * sizeof(uchar2), y);
        surf2Dwrite(uv, dstSurf_uv, x * sizeof(uchar2), y);
    }
    
    // Copy or clear depth
    float depth;
    surf2Dread(&depth, srcSurf_depth, x * sizeof(float), y);
    surf2Dwrite(depth, dstSurf_depth, x * sizeof(float), y);
}


#define BLOCK_SIZE 16
#define MAX_MATCHES_PER_BLOCK 64
#define MAX_DISTANCE 100000.0f 


__global__ void set_distance_texture(
    cudaSurfaceObject_t depth_texture,
    MatchedKeypoint* matches,
    uint num_matches,
    int width,
    int height
){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    
    if (x >= width || y >= height) return;

    // Shared memory for matches
    __shared__ float2 s_keypoints[MAX_MATCHES_PER_BLOCK];
    __shared__ float s_depths[MAX_MATCHES_PER_BLOCK];
    
    float nearest_depth = 0.0f;
    float min_dist = MAX_DISTANCE;
    
    // Process matches in chunks to handle cases with many matches
    for (int chunk_start = 0; chunk_start < num_matches; chunk_start += MAX_MATCHES_PER_BLOCK) {
        const int chunk_size = min(MAX_MATCHES_PER_BLOCK, num_matches - chunk_start);
        
        // Cooperatively load chunk into shared memory
        if (tx + ty * BLOCK_SIZE < chunk_size) {
            int match_idx = chunk_start + tx + ty * BLOCK_SIZE;
            const MatchedKeypoint match = matches[match_idx];
            s_keypoints[tx + ty * BLOCK_SIZE] = match.world.image_coords;
            s_depths[tx + ty * BLOCK_SIZE] = match.world.position.y;
        }
        __syncthreads();
        
        // Find nearest match in this chunk
        #pragma unroll 4
        for (int i = 0; i < chunk_size; i++) {
            const float2 kp = s_keypoints[i];
            const float dx = kp.x - x;
            const float dy = kp.y - y;
            const float dist = dx * dx + dy * dy;
            
            if (dist < min_dist) {
                min_dist = dist;
                nearest_depth = s_depths[i];
            }
        }
        __syncthreads();
    }
    
    // Only write depth if we found a reasonably close match
    if (min_dist < MAX_DISTANCE) {
        surf2Dwrite(nearest_depth, depth_texture, x * sizeof(float), y);
    } else {
        surf2Dwrite(0.0f, depth_texture, x * sizeof(float), y);
    }
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
    const uint match_count,
    
    float4*  keypoint_positions,
    float4*  keypoint_colors,
    
    float4*  left_line_positions,
    float4* left_line_colors,
    
    float4* __restrict__ right_line_positions,
    float4* __restrict__ right_line_colors,

    const float* left_transform,
    const float* right_transform,

    bool show_connections,
    bool disable_depth
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= match_count) return;

    const MatchedKeypoint match = matches[idx];
    
    float4 match_color = generate_unique_color(idx, 0.5f);
    
    // Transform the world position
    keypoint_positions[idx] = make_float4(match.world.position.x, !disable_depth ? match.world.position.y - 0.01 : -0.01, match.world.position.z, 0.0f);
    keypoint_colors[idx] = make_float4(1.0f, 0.0f, 1.0f, 1.0f);

    if (show_connections){
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
    } else{
        left_line_positions[idx * 2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        left_line_positions[idx * 2 + 1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        left_line_colors[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // Transparent

        right_line_positions[idx * 2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        right_line_positions[idx * 2 + 1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        right_line_colors[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // Transparent

    }

}

// ============================================================= Visual Odometry =================================================================



__device__ float3 matmul3x3(const float* matrix, float3 vec) {
    float3 result;
    result.x = matrix[0] * vec.x + matrix[1] * vec.y + matrix[2] * vec.z;
    result.y = matrix[3] * vec.x + matrix[4] * vec.y + matrix[5] * vec.z;
    result.z = matrix[6] * vec.x + matrix[7] * vec.y + matrix[8] * vec.z;
    return result;
}

__device__ float3 cross_product(float3 a, float3 b) {
    float3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

__device__ float dot_product(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float computeTemporalCost(
    const MatchedKeypoint &prev,
    const MatchedKeypoint &curr,
    const TemporalParams &params
) {
    float3 world_diff;
    world_diff.x = prev.world.position.x - curr.world.position.x;
    world_diff.y = prev.world.position.y - curr.world.position.y;
    world_diff.z = prev.world.position.z - curr.world.position.z;
    float spatial_dist = sqrtf(world_diff.x * world_diff.x + 
                              world_diff.y * world_diff.y + 
                              world_diff.z * world_diff.z);
    
    if (spatial_dist > params.max_distance) {
        return 1e10f;
    }

    // 2. Descriptor distance
    int descDistance = hammingDistance(prev.left_desc, curr.left_desc);
    float descCost = (float)descDistance / 512.0f;

    if (descCost > params.max_hamming_dist) {
        return 1e10f;
    }

    // 3. Compare image space positions
    float2 prev_img = prev.world.image_coords;
    float2 curr_img = curr.world.image_coords;
    float img_dist = sqrtf((prev_img.x - curr_img.x) * (prev_img.x - curr_img.x) +
                          (prev_img.y - curr_img.y) * (prev_img.y - curr_img.y));

    float img_cost = img_dist / params.max_pixel_distance;

    // Combine costs with weights
    return params.spatial_weight * spatial_dist / params.max_distance +
           params.hamming_weight * descCost +
           params.img_weight * img_cost;
}


__global__ void matchCurrentToPrev(
    const MatchedKeypoint* curr_matches,
    uint curr_match_count,
    const MatchedKeypoint* prev_matches, 
    uint prev_match_count,
    BestMatch* curr_to_prev_matches,  // [curr_match_count]
    TemporalParams params
) {
    int curr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (curr_idx >= curr_match_count) return;

    const MatchedKeypoint curr = curr_matches[curr_idx];
    
    float best_cost = 1e9f;
    float second_best_cost = 1e9f;
    int best_match = -1;

    // Search for the best match in previous frame
    for (int prev_idx = 0; prev_idx < prev_match_count; prev_idx++) {
        const MatchedKeypoint prev = prev_matches[prev_idx];
        
        float cost = computeTemporalCost(prev, curr, params);

        if (cost < best_cost) {
            second_best_cost = best_cost;
            best_cost = cost;
            best_match = prev_idx;
        }
        else if (cost < second_best_cost) {
            second_best_cost = cost;
        }
    }

    // Apply ratio test
    if (second_best_cost < 1e9f && best_cost >= params.lowes_ratio * second_best_cost) {
        best_match = -1;  // Reject ambiguous matches
        best_cost = 1e9f;
    }

    // Apply absolute threshold
    if (best_cost >= params.cost_threshold) {
        best_match = -1;
        best_cost = 1e9f;
    }

    curr_to_prev_matches[curr_idx].bestIdx = best_match;
    curr_to_prev_matches[curr_idx].bestCost = best_cost;
}

__global__ void matchPrevToCurrent(
    const MatchedKeypoint* prev_matches,
    uint prev_match_count,
    const MatchedKeypoint* curr_matches,
    uint curr_match_count,
    BestMatch* prev_to_curr_matches,  // [prev_match_count]
    TemporalParams params
) {
    int prev_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (prev_idx >= prev_match_count) return;

    const MatchedKeypoint prev = prev_matches[prev_idx];
    
    float best_cost = 1e9f;
    float second_best_cost = 1e9f;
    int best_match = -1;

    // Search for the best match in current frame
    for (int curr_idx = 0; curr_idx < curr_match_count; curr_idx++) {
        const MatchedKeypoint curr = curr_matches[curr_idx];
        
        float cost = computeTemporalCost(prev, curr, params);

        if (cost < best_cost) {
            second_best_cost = best_cost;
            best_cost = cost;
            best_match = curr_idx;
        }
        else if (cost < second_best_cost) {
            second_best_cost = cost;
        }
    }

    // Apply ratio test
    if (second_best_cost < 1e9f && best_cost >= params.lowes_ratio * second_best_cost) {
        best_match = -1;  // Reject ambiguous matches
        best_cost = 1e9f;
    }

    // Apply absolute threshold
    if (best_cost >= params.cost_threshold) {
        best_match = -1;
        best_cost = 1e9f;
    }

    prev_to_curr_matches[prev_idx].bestIdx = best_match;
    prev_to_curr_matches[prev_idx].bestCost = best_cost;
}

__global__ void crossCheckTemporalMatches(
    const BestMatch* curr_to_prev_matches,
    const BestMatch* prev_to_curr_matches,
    const MatchedKeypoint* curr_matches,
    const MatchedKeypoint* prev_matches,
    uint curr_match_count,
    uint prev_match_count,
    TemporalMatch* temporal_matches,
    uint* temporal_match_count,
    TemporalParams params
) {
    int curr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (curr_idx >= curr_match_count) return;

    // Get the matched index in previous frame
    int prev_idx = curr_to_prev_matches[curr_idx].bestIdx;
    if (prev_idx < 0 || prev_idx >= prev_match_count) return;

    // Cross check - verify that prev_idx matches back to curr_idx
    if (prev_to_curr_matches[prev_idx].bestIdx == curr_idx) {
        // We have a consistent match

        // Calculate match confidence from costs
        float curr_to_prev_cost = curr_to_prev_matches[curr_idx].bestCost;
        float prev_to_curr_cost = prev_to_curr_matches[prev_idx].bestCost;
        float confidence = 1.0f / (0.5f * (curr_to_prev_cost + prev_to_curr_cost) + 1e-5f);

        if (confidence > params.min_confidence) {
            int match_idx = atomicAdd(temporal_match_count, 1);
            if (match_idx < min(curr_match_count, prev_match_count)) {
                temporal_matches[match_idx].prev_pos = prev_matches[prev_idx].world.position;
                temporal_matches[match_idx].current_pos = curr_matches[curr_idx].world.position;
                temporal_matches[match_idx].confidence = confidence;
            }
        }
    }
}

// Kernel to match keypoints between temporal frames
__global__ void matchTemporalKeypoints(
    const MatchedKeypoint* prev_matches,
    uint prev_match_count,
    const MatchedKeypoint* curr_matches, 
    uint curr_match_count,
    TemporalMatch* temporal_matches,
    uint* temporal_match_count,
    TemporalParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= curr_match_count) return;

    const MatchedKeypoint curr = curr_matches[idx];
    
    float best_cost = 1e9f;
    float second_best_cost = 1e9f;
    int best_match = -1;

    // Search for the closest match in previous frame
    for (int i = 0; i < prev_match_count; i++) {
        const MatchedKeypoint prev = prev_matches[i];
        
        // Use left camera descriptors for matching
        float cost = computeTemporalCost(
            prev,
            curr,
            params
        );

        if (cost < best_cost) {
            second_best_cost = best_cost;
            best_cost = cost;
            best_match = i;
        }
        else if (cost < second_best_cost) {
            second_best_cost = cost;
        }
    }

    // Apply Lowe's ratio test and absolute threshold
    if (best_match >= 0 && 
        best_cost < params.cost_threshold &&
        (second_best_cost >= 1e9f || best_cost < params.lowes_ratio * second_best_cost)) {
        
        int match_idx = atomicAdd(temporal_match_count, 1);
        if (match_idx < min(prev_match_count, curr_match_count)){
            temporal_matches[match_idx].prev_pos = prev_matches[best_match].world.position;
            temporal_matches[match_idx].current_pos = curr.world.position;
            temporal_matches[match_idx].confidence = 1.0f / (best_cost + 1e-5f);
        }
    }
}

__global__ void visualizeTemporalMatches(
    const TemporalMatch* matches,
    uint match_count,
    float4* keypoint_positions,
    float4* keypoint_colors,
    float4* connection_positions,
    float4* connection_colors,
    const float* prev_transform,
    const float* curr_transform,
    bool disable_depth
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= match_count) return;

    const TemporalMatch match = matches[idx];
    
    // Draw keypoint at prev position
    keypoint_positions[idx] = make_float4(
        match.prev_pos.x,
        !disable_depth ? match.prev_pos.y - 0.01 : -0.01,
        match.prev_pos.z,
        0.0f
    );
    keypoint_colors[idx] = make_float4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow

    // Draw connection line between prev and current positions
    float4 prev_pos = make_float4(
        match.prev_pos.x,
        !disable_depth ? match.prev_pos.y - 0.01 : -0.01,
        match.prev_pos.z,
        0.0f
    );
    float4 curr_pos = transform_point(curr_transform, 
        make_float4(match.current_pos.x, !disable_depth ? match.current_pos.y - 0.01 : -0.01 , match.current_pos.z, 0.0f));

    connection_positions[idx * 2 + 1] = prev_pos;
    connection_positions[idx * 2 ] = make_float4(curr_pos.x, curr_pos.y, curr_pos.z, 0);
    connection_colors[idx] = make_float4(0.0f, 1.0f, 1.0f, 0.5f); // Semi-transparent cyan
}

// ------------------------------------------------------------------
// Helpers for double-precision 3D math
// ------------------------------------------------------------------
__device__ inline double3 make_d3(float3 f)
{
    return make_double3((double)f.x, (double)f.y, (double)f.z);
}
__device__ inline double3 dsub(const double3 &a, const double3 &b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline double3 dadd(const double3 &a, const double3 &b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline double3 dscale(const double3 &a, double s)
{
    return make_double3(a.x*s, a.y*s, a.z*s);
}
__device__ inline double3 cross_d(const double3 &u, const double3 &v)
{
    return make_double3(
       u.y*v.z - u.z*v.y,
       u.z*v.x - u.x*v.z,
       u.x*v.y - u.y*v.x
    );
}
__device__ inline double length_d(const double3 &v)
{
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

// ------------------------------------------------------------------
// 3x3 symmetric eigen-decomposition in double (Jacobi for B = H^T H).
// For brevity, we keep exactly the same approach as prior examples.
// ------------------------------------------------------------------
__device__ void eigenDecomposeSym3Double(const double B[9], double eigVal[3], double eigVec[9])
{
    // Copy B => A, we’ll do in-place Jacobi on A
    // Use Q to accumulate eigenvectors
    double A[9];
    for (int i=0; i<9; i++){
        A[i] = B[i];
    }
    double Q[9] = {1,0,0, 0,1,0, 0,0,1}; // identity

    const int maxIter = 15;
    for(int iter=0; iter<maxIter; iter++){
        // largest off-diag element => pick p,q
        int p=0,q=1;
        double apqMax = fabs(A[1]);
        // check (0,2)
        double tmp = fabs(A[2]);
        if (tmp>apqMax){ apqMax = tmp; p=0; q=2; }
        // check (1,2)
        tmp = fabs(A[5]);
        if (tmp>apqMax){ apqMax = tmp; p=1; q=2; }
        if (apqMax < 1e-15) break; // done

        // Jacobi rotate p,q
        double app = A[p*3+p];
        double aqq = A[q*3+q];
        double apq = A[p*3+q];

        double tau  = (aqq - app)/(2.0 * apq);
        double t    = (tau >= 0.0)? 1.0/(fabs(tau)+sqrt(1.0+tau*tau))
                                  : -1.0/(fabs(tau)+sqrt(1.0+tau*tau));
        double c    = 1.0 / sqrt(1.0 + t*t);
        double s    = t*c;

        // rotate A in rows
        for(int k=0; k<3; k++){
            double A_pk = A[p*3 + k];
            double A_qk = A[q*3 + k];
            A[p*3 + k] = A_pk*c - A_qk*s;
            A[q*3 + k] = A_pk*s + A_qk*c;
        }
        // rotate A in columns
        for(int k=0; k<3; k++){
            double A_kp = A[k*3 + p];
            double A_kq = A[k*3 + q];
            A[k*3 + p] = A_kp*c - A_kq*s;
            A[k*3 + q] = A_kp*s + A_kq*c;
        }

        // rotate Q
        for(int k=0; k<3; k++){
            double Q_kp = Q[k*3 + p];
            double Q_kq = Q[k*3 + q];
            Q[k*3 + p] = Q_kp*c - Q_kq*s;
            Q[k*3 + q] = Q_kp*s + Q_kq*c;
        }
    }

    // diagonal => eigenvalues
    eigVal[0] = A[0];
    eigVal[1] = A[4];
    eigVal[2] = A[8];

    // sort them descending
    int idx[3] = {0,1,2};
    for(int i=0; i<2; i++){
        for(int j=0; j<2-i; j++){
            if(eigVal[j] < eigVal[j+1]){
                double tmp= eigVal[j];
                eigVal[j] = eigVal[j+1];
                eigVal[j+1]= tmp;
                int it= idx[j];
                idx[j]= idx[j+1];
                idx[j+1]= it;
            }
        }
    }

    // Re-map Q columns => eigVec
    // Q’s columns => eigenvectors
    for(int c=0; c<3; c++){
        int oldC = idx[c];
        eigVec[0 + 3*c] = Q[0 + 3*oldC];
        eigVec[1 + 3*c] = Q[1 + 3*oldC];
        eigVec[2 + 3*c] = Q[2 + 3*oldC];
    }
}

// ------------------------------------------------------------------
// A small helper to pick 3 distinct random indices in [0..maxval-1]
// using curand (Philox or XORWOW, etc.).
// We do a tiny loop if collisions happen.
// ------------------------------------------------------------------
__device__ inline void pick3DistinctIndices(curandStatePhilox4_32_10_t &rng,
                                           int maxval, int &i1, int &i2, int &i3)
{
    i1 = (int)(curand_uniform(&rng)*maxval);
    i2 = (int)(curand_uniform(&rng)*maxval);
    i3 = (int)(curand_uniform(&rng)*maxval);
    for(int tries=0; tries<5; tries++){
        if(i2==i1) i2 = (int)(curand_uniform(&rng)*maxval);
        if(i3==i1 || i3==i2) i3 = (int)(curand_uniform(&rng)*maxval);
        if(i1!=i2 && i2!=i3 && i1!=i3) break;
    }
}


// Picks N distinct random indices in [0..maxval-1].
__device__ inline void pickNDistinctIndices(
    curandStatePhilox4_32_10_t &rng,
    int maxval,
    int *out_idx,
    int N
)
{
    for (int i = 0; i < N; i++) {
        out_idx[i] = (int)(curand_uniform(&rng) * maxval);
        // Try a few times to ensure distinctness
        for (int tries = 0; tries < 5; tries++) {
            // Check against previously picked
            bool collision = false;
            for (int j = 0; j < i; j++) {
                if (out_idx[i] == out_idx[j]) {
                    collision = true;
                    break;
                }
            }
            if (collision) {
                out_idx[i] = (int)(curand_uniform(&rng) * maxval);
            }
            else {
                // no collision => done
                break;
            }
        }
    }
}



// ------------------------------------------------------------------
// Kabsch-based RANSAC kernel
//  * Everything (including RNG init) is inside the kernel
//  * Each block does exactly one RANSAC iteration
//  * We store the best pose globally
// ------------------------------------------------------------------
__global__ void estimateMotionRANSAC(
    const TemporalMatch* matches,
    uint match_count,
    TemporalParams params,
    CameraPose* best_pose,
    uint* inlier_count
)
{
    // We want one RANSAC iteration per block
    // We'll have multiple blocks => multiple guesses => 
    // the best guess is stored via atomic logic below.
    // We want one RANSAC iteration per block:
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    uint subset_size = params.ransac_points;
    
    // Shared memory for counting inliers and intermediate computations
    __shared__ int s_inliers;
    __shared__ double s_H[9];  // Cross-covariance matrix
    __shared__ double s_B[9];  // B = H^T * H
    __shared__ double3 s_centroidP, s_centroidC;
    __shared__ int s_idx[16];   // Random indices
    
    // Initialize shared memory
    if (threadId == 0) {
        s_inliers = 0;
        // We'll only init best_pose on block 0, thread 0
        if (blockId == 0) {
            *inlier_count = 0;
            for(int i=0; i<9; i++){
                best_pose->rotation[i] = (i%4==0) ? 1.f : 0.f;
            }
            best_pose->translation[0] = best_pose->translation[1] = best_pose->translation[2] = 0.f;
        }
    }
    __syncthreads();

    if(match_count < params.min_matches) {
        return; // nothing to do
    }

    // Initialize local RNG
    unsigned long long seed = clock64() + blockId * 1337ULL;
    curandStatePhilox4_32_10_t rngState;
    curand_init(seed, /*subsequence*/0, /*offset*/0, &rngState);

    // Thread 0 picks random indices
    if (threadId == 0) {
        pickNDistinctIndices(rngState, match_count, s_idx, subset_size);
    }
    __syncthreads();
    
    double3 sumPrev = make_double3(0.0, 0.0, 0.0);
    double3 sumCurr = make_double3(0.0, 0.0, 0.0);

    // Gather the points
    if (threadId == 0) {
        for (int i = 0; i < subset_size; i++) {
            const int mIdx = s_idx[i];
            double3 p = make_double3(
                (double)matches[mIdx].prev_pos.x,
                (double)matches[mIdx].prev_pos.y,
                (double)matches[mIdx].prev_pos.z
            );
            double3 c = make_double3(
                (double)matches[mIdx].current_pos.x,
                (double)matches[mIdx].current_pos.y,
                (double)matches[mIdx].current_pos.z
            );
            sumPrev.x += p.x; sumPrev.y += p.y; sumPrev.z += p.z;
            sumCurr.x += c.x; sumCurr.y += c.y; sumCurr.z += c.z;
        }
        double invN = 1.0 / (double)subset_size;
        s_centroidP = make_double3(sumPrev.x*invN, sumPrev.y*invN, sumPrev.z*invN);
        s_centroidC = make_double3(sumCurr.x*invN, sumCurr.y*invN, sumCurr.z*invN);
    }

    __syncthreads();

    // Initialize H matrix to zero
    if (threadId < 9) {
        s_H[threadId] = 0.0;
    }
    __syncthreads();

    // Compute cross-covariance matrix H (thread 0)
    if (threadId == 0) {
        auto accumulateH = [&](double3 pp, double3 cc)
        {
            double3 pC = make_double3(pp.x - s_centroidP.x, 
                                      pp.y - s_centroidP.y, 
                                      pp.z - s_centroidP.z);
            double3 cC = make_double3(cc.x - s_centroidC.x, 
                                      cc.y - s_centroidC.y, 
                                      cc.z - s_centroidC.z);

            s_H[0] += pC.x * cC.x;  s_H[1] += pC.x * cC.y;  s_H[2] += pC.x * cC.z;
            s_H[3] += pC.y * cC.x;  s_H[4] += pC.y * cC.y;  s_H[5] += pC.y * cC.z;
            s_H[6] += pC.z * cC.x;  s_H[7] += pC.z * cC.y;  s_H[8] += pC.z * cC.z;
        };

        for (int i = 0; i < subset_size; i++) {
            int mIdx = s_idx[i];
            double3 pp = make_double3(
                (double)matches[mIdx].prev_pos.x,
                (double)matches[mIdx].prev_pos.y,
                (double)matches[mIdx].prev_pos.z
            );
            double3 cc = make_double3(
                (double)matches[mIdx].current_pos.x,
                (double)matches[mIdx].current_pos.y,
                (double)matches[mIdx].current_pos.z
            );
            accumulateH(pp, cc);
        }
    }
    __syncthreads();

    // Compute B = H^T * H (thread 0)
    if (threadId == 0) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                double sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    // H^T => swap (r,k) => (k,r)
                    sum += s_H[k * 3 + r] * s_H[k * 3 + c];
                }
                s_B[r * 3 + c] = sum;
            }
        }
    }

    __syncthreads();

    // Compute rotation matrix R and translation vector t (thread 0)
    double eigVal[3], V[9];
    __shared__ double R_d[9];
    __shared__ double3 t_d;

    if (threadId == 0) {
        // Decompose s_B => V
        eigenDecomposeSym3Double(s_B, eigVal, V);

        // Compute singular values and U matrix
        double sigma[3];
        for(int i=0; i<3; i++) {
            sigma[i] = (eigVal[i] < 1e-15) ? 0.0 : sqrt(eigVal[i]);
        }

        // U = H * V * Sigma^-1
        double M[9], U[9];
        // M = H*V
        for(int r=0; r<3; r++) {
            for(int c=0; c<3; c++) {
                double sum = 0;
                for(int k=0; k<3; k++) {
                    sum += s_H[r*3 + k] * V[k*3 + c];
                }
                M[r*3+c] = sum;
            }
        }
        // Scale by 1/sigma
        for(int c=0; c<3; c++) {
            double invS = (sigma[c]<1e-15) ? 0.0 : 1.0/sigma[c];
            for(int r=0; r<3; r++) {
                U[r*3 + c] = M[r*3 + c] * invS;
            }
        }

        // R = U * V^T
        for(int r=0; r<3; r++) {
            for(int c=0; c<3; c++) {
                double sum = 0;
                for(int k=0; k<3; k++) {
                    sum += U[r*3 + k] * V[c*3 + k]; // V^T => swap indices
                }
                R_d[r*3 + c] = sum;
            }
        }

        // Check determinant and fix if necessary
        double det = R_d[0]*(R_d[4]*R_d[8] - R_d[5]*R_d[7])
                  - R_d[1]*(R_d[3]*R_d[8] - R_d[5]*R_d[6])
                  + R_d[2]*(R_d[3]*R_d[7] - R_d[4]*R_d[6]);
        if(det < 0.0) {
            // Flip last column
            for(int r=0; r<3; r++) {
                R_d[r*3 + 2] *= -1.0;
            }
        }

        // Compute translation
        double xT = R_d[0]*s_centroidP.x + R_d[1]*s_centroidP.y + R_d[2]*s_centroidP.z;
        double yT = R_d[3]*s_centroidP.x + R_d[4]*s_centroidP.y + R_d[5]*s_centroidP.z;
        double zT = R_d[6]*s_centroidP.x + R_d[7]*s_centroidP.y + R_d[8]*s_centroidP.z;
      
        t_d.x = s_centroidC.x - xT;
        t_d.y = s_centroidC.y - yT;
        t_d.z = s_centroidC.z - zT;
    }
    __syncthreads();

    if (threadId == 0) {
        s_inliers = 0;
    }
    __syncthreads();

    // Count inliers in parallel across all threads
    for(int i = threadId; i < match_count; i += blockDim.x) {
        double3 ptP = make_d3(matches[i].prev_pos);
        // Transform point using R and t
        double xT = R_d[0]*ptP.x + R_d[1]*ptP.y + R_d[2]*ptP.z + t_d.x;
        double yT = R_d[3]*ptP.x + R_d[4]*ptP.y + R_d[5]*ptP.z + t_d.y;
        double zT = R_d[6]*ptP.x + R_d[7]*ptP.y + R_d[8]*ptP.z + t_d.z;

        double3 cP = make_double3(
            (double)matches[i].current_pos.x,
            (double)matches[i].current_pos.y,
            (double)matches[i].current_pos.z
        );
        double dx = xT - cP.x;
        double dy = yT - cP.y;
        double dz = zT - cP.z;
        double dist = sqrt(dx*dx + dy*dy + dz*dz);

        if (dist < params.ransac_threshold) {
            atomicAdd(&s_inliers, 1);
        }
    }
    __syncthreads();

    // Update best pose if we found more inliers (thread 0)
    if(threadId == 0) {
        int block_inliers = s_inliers;
        float ratio = block_inliers / (float)match_count;

        // Atomic update of best result
        int oldVal = atomicMax((int*)inlier_count, block_inliers);
        if (block_inliers > oldVal) {
            // store R, t into best_pose
            for(int i=0; i<9; i++){
                best_pose->rotation[i] = (float)R_d[i];
            }
            best_pose->translation[0] = (float)t_d.x;
            best_pose->translation[1] = (float)t_d.y;
            best_pose->translation[2] = (float)t_d.z;
        }
    }
}

extern "C" {
    void init_gaussian_kernel(float sigma) {
        float h_gaussian_kernel[GAUSSIAN_KERNEL_SIZE];
        float sum = 0.0f;
        
        for (int i = -GAUSSIAN_KERNEL_RADIUS; i <= GAUSSIAN_KERNEL_RADIUS; i++) {
            float x = i;
            h_gaussian_kernel[i + GAUSSIAN_KERNEL_RADIUS] = expf(-(x * x) / (2 * sigma * sigma));
            sum += h_gaussian_kernel[i + GAUSSIAN_KERNEL_RADIUS];
        }
        
        for (int i = 0; i < GAUSSIAN_KERNEL_SIZE; i++) {
            h_gaussian_kernel[i] /= sum;
        }
        
        cudaMemcpyToSymbol(d_gaussian_kernel, h_gaussian_kernel, GAUSSIAN_KERNEL_SIZE * sizeof(float));
    }

    void launch_gaussian_blur(
        const uint8_t* input,
        uint8_t* temp1,
        uint8_t* temp2,
        int width,
        int height,
        int pitch,
        dim3 grid,
        dim3 block
    ) {
        gaussianBlurHorizontal<<<grid, block>>>(input, temp1, width, height, pitch);
        gaussianBlurVertical<<<grid, block>>>(temp1, temp2, width, height, pitch);
    }

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
    ) {
        detectFASTKeypoints<<<grid, block>>>(
            input, width, height, pitch,
            threshold, arc_length,
            positions, colors,
            descriptors, keypoint_count,
            max_keypoints
        );
    }

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
    ) {
        dim3 block = 512;

        {
            dim3 grid( (left_count + block.x - 1) / block.x );
            matchLeftToRight<<<grid, block>>>(
                left_positions, left_descriptors, left_count,
                right_positions, right_descriptors, right_count,
                params, matches_left
            );
        }

        {
            dim3 grid( (right_count + block.x - 1) / block.x );
            matchRightToLeft<<<grid, block>>>(
                right_positions, right_descriptors, right_count,
                left_positions, left_descriptors, left_count,
                params, matches_right
            );
        }
    }

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
    ) {
        crossCheckMatches<<<grid, block>>>(
            matches_left, matches_right,
            left_positions, right_positions,
            left_count, right_count,
            params,
            matched_pairs, out_count
        );
    }

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
    ) {
        visualizeTriangulation<<<grid, block>>>(
            matches, match_count,
            keypoint_positions, keypoint_colors,
            left_line_positions, left_line_colors,
            right_line_positions, right_line_colors,
            left_transform, right_transform,
            show_connections, disable_depth
        );
    }

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
    ) {
        setTextureKernel<<<grid, block>>>(
            y_surface, uv_surface, depth_surface,
            y_plane, uv_plane,
            width, height,
            y_linesize, uv_linesize
        );
    }

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
    ) {
        copySurfaceKernel<<<grid, block>>>(
            src_y, src_uv, src_depth,
            dst_y, dst_uv, dst_depth,
            width, height
        );
    }

    void launch_depth_texture_update(
        cudaSurfaceObject_t depth_surface,
        MatchedKeypoint* matches,
        uint num_matches,
        int width,
        int height,
        dim3 grid,
        dim3 block
    ) {
        set_distance_texture<<<grid, block>>>(
            depth_surface,
            matches,
            num_matches,
            width,
            height
        );
    }

  
    void launch_temporal_match_current_to_prev(
        const MatchedKeypoint* curr_matches,
        uint curr_match_count,
        const MatchedKeypoint* prev_matches, 
        uint prev_match_count,
        BestMatch* curr_to_prev_matches,
        TemporalParams params,
        dim3 grid,
        dim3 block
    ){
        matchCurrentToPrev<<<grid, block>>>(
            curr_matches, curr_match_count,
            prev_matches, prev_match_count,
            curr_to_prev_matches, params
        );
    }

    void launch_temporal_match_prev_to_current(
        const MatchedKeypoint* prev_matches,
        uint prev_match_count,
        const MatchedKeypoint* curr_matches,
        uint curr_match_count,
        BestMatch* prev_to_curr_matches,
        TemporalParams params,
        dim3 grid,
        dim3 block
    ){
        matchPrevToCurrent<<<grid, block>>>(
            prev_matches, prev_match_count,
            curr_matches, curr_match_count,
            prev_to_curr_matches, params
        );
    }

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
    ){
        crossCheckTemporalMatches<<<grid, block>>>(
            curr_to_prev_matches, prev_to_curr_matches,
            curr_matches, prev_matches,
            curr_match_count, prev_match_count,
            temporal_matches, temporal_match_count,
            params
        );
    }


    void launch_motion_estimation(
        const TemporalMatch* d_matches,
        uint match_count,
        CameraPose* d_best_pose,
        uint* d_inlier_count,
        TemporalParams* params,
        dim3 grid,
        dim3 block
    ) {  
        estimateMotionRANSAC<<<grid, block>>>(
            d_matches,
            match_count,
            *params,
            d_best_pose,
            d_inlier_count
        );
    }

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
    ) {
        visualizeTemporalMatches<<<grid, block>>>(
            matches,
            match_count,
            keypoint_positions,
            keypoint_colors,
            connection_positions,
            connection_colors,
            prev_transform,
            curr_transform,
            disable_depth
        );
    }

} 