#include "kernels.h"

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
    bool is_keypoint = (max_consecutive_brighter >= arc_length || max_consecutive_darker >= arc_length);

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

    bool show_connections
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= match_count) return;

    const MatchedKeypoint match = matches[idx];
    
    float4 match_color = generate_unique_color(idx, 0.5f);
    
    // Transform the world position
    keypoint_positions[idx] = make_float4(match.world.position.x, match.world.position.y - 0.01, match.world.position.z, 0.0f);
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

    if (descCost > params.stereo_params.max_hamming_dist) {
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
        best_cost < params.stereo_params.cost_threshold &&
        (second_best_cost >= 1e9f || best_cost < params.stereo_params.lowes_ratio * second_best_cost)) {
        
        int match_idx = atomicAdd(temporal_match_count, 1);
        temporal_matches[match_idx].prev_pos = prev_matches[best_match].world.position;
        temporal_matches[match_idx].current_pos = curr.world.position;
        temporal_matches[match_idx].confidence = 1.0f / (best_cost + 1e-5f);
    }
}


__device__ void svd_sort(float* u, float* v, float* w, int n) {
    // Sort singular values in descending order
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (w[j] < w[j + 1]) {
                // Swap singular values
                float temp = w[j];
                w[j] = w[j + 1];
                w[j + 1] = temp;
                
                // Swap corresponding columns in u and v
                for (int k = 0; k < n; k++) {
                    temp = u[k * n + j];
                    u[k * n + j] = u[k * n + j + 1];
                    u[k * n + j + 1] = temp;
                    
                    temp = v[k * n + j];
                    v[k * n + j] = v[k * n + j + 1];
                    v[k * n + j + 1] = temp;
                }
            }
        }
    }
}

__device__ bool compute_svd(
    float* H,        // Input 3x3 matrix in row-major order
    float* u,        // Output U matrix
    float* v,        // Output V matrix
    float* w,        // Output singular values
    const float eps = 1e-10f  // Numerical threshold
) {
    const int n = 3;  // Matrix size
    const int max_iter = 30;  // Maximum iterations
    
    // Initialize U as identity
    for (int i = 0; i < n * n; i++) {
        u[i] = (i % (n + 1) == 0) ? 1.0f : 0.0f;
    }
    
    // Copy H to V
    for (int i = 0; i < n * n; i++) {
        v[i] = H[i];
    }
    
    // Check input matrix for NaN/Inf
    for (int i = 0; i < n * n; i++) {
        if (!isfinite(H[i])) return false;
    }
    
    // Main Jacobi SVD iteration
    bool converged = false;
    for (int iter = 0; iter < max_iter && !converged; iter++) {
        float sum_offdiag = 0.0f;
        
        // Compute sum of off-diagonal elements
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum_offdiag += fabs(v[i * n + j]);
                }
            }
        }
        
        if (sum_offdiag < eps) {
            converged = true;
            break;
        }
        
        // Process all upper diagonal elements
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                float app = v[p * n + p];
                float apq = v[p * n + q];
                float aqq = v[q * n + q];
                
                float diff = aqq - app;
                if (fabs(apq) < eps * (fabs(app) + fabs(aqq))) {
                    continue;
                }
                
                // Compute Jacobi rotation
                float theta = 0.5f * atan2(2.0f * apq, diff + copysignf(
                    sqrtf(diff * diff + 4.0f * apq * apq), diff));
                float c = cosf(theta);
                float s = sinf(theta);
                
                if (!isfinite(c) || !isfinite(s)) return false;
                
                // Apply rotation to V
                for (int i = 0; i < n; i++) {
                    float vip = v[i * n + p];
                    float viq = v[i * n + q];
                    v[i * n + p] = vip * c - viq * s;
                    v[i * n + q] = vip * s + viq * c;
                }
                
                // Apply rotation to U
                for (int i = 0; i < n; i++) {
                    float uip = u[i * n + p];
                    float uiq = u[i * n + q];
                    u[i * n + p] = uip * c - uiq * s;
                    u[i * n + q] = uip * s + uiq * c;
                }
            }
        }
    }
    
    if (!converged) return false;
    
    // Extract singular values and normalize
    for (int i = 0; i < n; i++) {
        w[i] = sqrtf(v[i * n + i] * v[i * n + i]);
        
        // Check for near-zero singular values
        if (w[i] < eps) return false;
        
        // Normalize column in V
        float norm = 1.0f / w[i];
        for (int j = 0; j < n; j++) {
            v[j * n + i] *= norm;
        }
    }
    
    // Sort singular values in descending order
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (w[j] < w[j + 1]) {
                // Swap singular values
                float temp = w[j];
                w[j] = w[j + 1];
                w[j + 1] = temp;
                
                // Swap corresponding columns in u and v
                for (int k = 0; k < n; k++) {
                    temp = u[k * n + j];
                    u[k * n + j] = u[k * n + j + 1];
                    u[k * n + j + 1] = temp;
                    
                    temp = v[k * n + j];
                    v[k * n + j] = v[k * n + j + 1];
                    v[k * n + j + 1] = temp;
                }
            }
        }
    }
    
    return true;
}

__global__ void estimateMotionRANSAC(
    const TemporalMatch* matches,
    uint match_count,
    TemporalParams params,
    CameraPose* best_pose,
    float* inlier_count
) {
    // Local variables instead of shared memory
    float l_rotation[9];
    float l_translation[3];
    __shared__ int s_inliers;
    
    // Initialize shared memory and best pose
    if (threadIdx.x == 0) {
        s_inliers = 0;
        
        if (blockIdx.x == 0) {
            *inlier_count = 0.0f;
            // Initialize best_pose to identity
            for (int i = 0; i < 9; i++) {
                best_pose->rotation[i] = (i % 4 == 0) ? 1.0f : 0.0f;
            }
            for (int i = 0; i < 3; i++) {
                best_pose->translation[i] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    if (match_count < params.min_matches) return;
    
    // Randomly select 3 matches
    int idx1 = (blockIdx.x * 17 + 13) % match_count;
    int idx2 = (blockIdx.x * 23 + 7) % match_count;
    int idx3 = (blockIdx.x * 31 + 11) % match_count;
    
    if (idx1 == idx2 || idx2 == idx3 || idx1 == idx3) return;
    
    // Check if points form a valid triangle
    float3 p1 = matches[idx1].prev_pos;
    float3 p2 = matches[idx2].prev_pos;
    float3 p3 = matches[idx3].prev_pos;
    
    float3 v1 = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
    float3 v2 = make_float3(p3.x - p1.x, p3.y - p1.y, p3.z - p1.z);
    float3 cross = cross_product(v1, v2);
    float area = sqrtf(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z) * 0.5f;
    
    if (area < 1e-7f) return;  // Points are nearly collinear
    
    // Calculate centroid of points
    float3 centroid_prev = make_float3(0, 0, 0);
    float3 centroid_curr = make_float3(0, 0, 0);
    
    centroid_prev.x = (matches[idx1].prev_pos.x + matches[idx2].prev_pos.x + matches[idx3].prev_pos.x) / 3.0f;
    centroid_prev.y = (matches[idx1].prev_pos.y + matches[idx2].prev_pos.y + matches[idx3].prev_pos.y) / 3.0f;
    centroid_prev.z = (matches[idx1].prev_pos.z + matches[idx2].prev_pos.z + matches[idx3].prev_pos.z) / 3.0f;
    
    centroid_curr.x = (matches[idx1].current_pos.x + matches[idx2].current_pos.x + matches[idx3].current_pos.x) / 3.0f;
    centroid_curr.y = (matches[idx1].current_pos.y + matches[idx2].current_pos.y + matches[idx3].current_pos.y) / 3.0f;
    centroid_curr.z = (matches[idx1].current_pos.z + matches[idx2].current_pos.z + matches[idx3].current_pos.z) / 3.0f;
    
    // Check centroids for validity
    if (!isfinite(centroid_prev.x) || !isfinite(centroid_prev.y) || !isfinite(centroid_prev.z) ||
        !isfinite(centroid_curr.x) || !isfinite(centroid_curr.y) || !isfinite(centroid_curr.z)) {
        return;
    }
    
    // Calculate correlation matrix
    float H[9] = {0};
    
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        float3 p = matches[i == 0 ? idx1 : (i == 1 ? idx2 : idx3)].prev_pos;
        float3 c = matches[i == 0 ? idx1 : (i == 1 ? idx2 : idx3)].current_pos;
        
        p.x -= centroid_prev.x; p.y -= centroid_prev.y; p.z -= centroid_prev.z;
        c.x -= centroid_curr.x; c.y -= centroid_curr.y; c.z -= centroid_curr.z;
        
        if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z) ||
            !isfinite(c.x) || !isfinite(c.y) || !isfinite(c.z)) {
        }
        
        H[0] += p.x * c.x; H[1] += p.x * c.y; H[2] += p.x * c.z;
        H[3] += p.y * c.x; H[4] += p.y * c.y; H[5] += p.y * c.z;
        H[6] += p.z * c.x; H[7] += p.z * c.y; H[8] += p.z * c.z;
    }
        
    // Compute SVD
    float u[9], v[9], w[3];
    if (!compute_svd(H, u, v, w, 1E-12f)) return;
    
    // Compute rotation matrix R = V * U^T
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            l_rotation[i * 3 + j] = 0;
            for (int k = 0; k < 3; k++) {
                l_rotation[i * 3 + j] += v[i * 3 + k] * u[j * 3 + k];
            }
        }
    }
    
    // Ensure proper orientation (determinant = 1)
    float det = l_rotation[0] * (l_rotation[4] * l_rotation[8] - l_rotation[5] * l_rotation[7]) -
                l_rotation[1] * (l_rotation[3] * l_rotation[8] - l_rotation[5] * l_rotation[6]) +
                l_rotation[2] * (l_rotation[3] * l_rotation[7] - l_rotation[4] * l_rotation[6]);
    
    if (det < 0) {
        // Flip the sign of the last column of V
        for (int i = 0; i < 3; i++) {
            v[i * 3 + 2] = -v[i * 3 + 2];
        }
        
        // Recompute rotation matrix
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                l_rotation[i * 3 + j] = 0;
                for (int k = 0; k < 3; k++) {
                    l_rotation[i * 3 + j] += v[i * 3 + k] * u[j * 3 + k];
                }
            }
        }
    }
    
    
    // Compute translation
    float3 rotated_centroid = matmul3x3(l_rotation, centroid_prev);
    l_translation[0] = centroid_curr.x - rotated_centroid.x;
    l_translation[1] = centroid_curr.y - rotated_centroid.y;
    l_translation[2] = centroid_curr.z - rotated_centroid.z;
    
    // Count inliers
    s_inliers = 0;
    for (int i = threadIdx.x; i < match_count; i += blockDim.x) {
        float3 transformed = matmul3x3(l_rotation, matches[i].prev_pos);
        transformed.x += l_translation[0];
        transformed.y += l_translation[1];
        transformed.z += l_translation[2];
        
        float3 diff;
        diff.x = transformed.x - matches[i].current_pos.x;
        diff.y = transformed.y - matches[i].current_pos.y;
        diff.z = transformed.z - matches[i].current_pos.z;
        
        float error = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        
        if (error < params.ransac_threshold) {
            atomicAdd(&s_inliers, 1);
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float current_inliers = (float)s_inliers / match_count;
        if (current_inliers > *inlier_count) {
            *inlier_count = current_inliers;
            
            for (int i = 0; i < 9; i++) {
                best_pose->rotation[i] = l_rotation[i];
            }
            for (int i = 0; i < 3; i++) {
                best_pose->translation[i] = l_translation[i];
            }          
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
        dim3 grid,
        dim3 block
    ) {
        visualizeTriangulation<<<grid, block>>>(
            matches, match_count,
            keypoint_positions, keypoint_colors,
            left_line_positions, left_line_colors,
            right_line_positions, right_line_colors,
            left_transform, right_transform,
            show_connections
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
        cudaSurfaceObject_t dst_y,
        cudaSurfaceObject_t dst_uv,
        cudaSurfaceObject_t dst_depth,
        int width,
        int height,
        dim3 grid,
        dim3 block
    ) {
        copySurfaceKernel<<<grid, block>>>(
            src_y, src_uv,
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

    void launch_temporal_matching(
        const MatchedKeypoint* prev_matches,
        uint prev_match_count,
        const MatchedKeypoint* curr_matches,
        uint curr_match_count,
        TemporalMatch* temporal_matches,
        uint* temporal_match_count,
        TemporalParams params,
        dim3 grid,
        dim3 block
    ) {

        matchTemporalKeypoints<<<grid, block>>>(
            prev_matches, prev_match_count,
            curr_matches, curr_match_count,
            temporal_matches, temporal_match_count,
            params
        );
    }

    void launch_motion_estimation(
        const TemporalMatch* d_matches,
        uint match_count,
        TemporalParams params,
        CameraPose* best_pose,
        float* inlier_count,
        dim3 grid,
        dim3 block
    ) {
    
        estimateMotionRANSAC<<<1, 1>>>(
            d_matches,
            match_count,
            params,
            best_pose,
            inlier_count
        );
    }
} 