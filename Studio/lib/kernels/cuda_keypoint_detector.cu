#include "cuda_keypoint_detector.h"
#include <cuda_runtime.h>

// FAST circle offsets
__constant__ int2 fast_offsets[16] = {
    {3,  0},  {3,  1},  {2,  2},  {1,  3},
    {0,  3},  {-1, 3},  {-2, 2},  {-3, 1},
    {-3, 0},  {-3, -1}, {-2, -2}, {-1, -3},
    {0, -3},  {1, -3},  {2, -2},  {3,  -1}
};

// Device buffers
static int* d_keypoint_count = nullptr;
static KeyPoint* d_keypoints = nullptr;
static int max_frame_width = 0;
static int max_frame_height = 0;

__global__ void detectFASTKeypoints(
    const uint8_t* __restrict__ y_plane,
    int width,
    int height,
    int linesize,
    uint8_t threshold,
    KeyPoint* keypoints,
    int* keypoint_count,
    int max_keypoints
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 3 || y < 3 || x >= width - 3 || y >= height - 3) return;
    
    const uint8_t center = y_plane[y * linesize + x];
    
    int brighter = 0;
    int darker = 0;
    
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        const int2 offset = fast_offsets[i];
        const uint8_t pixel = y_plane[(y + offset.y) * linesize + (x + offset.x)];
        
        if (pixel > center + threshold) brighter++;
        else if (pixel < center - threshold) darker++;
        
        if (i >= 8 && brighter == 0 && darker == 0) return;
    }
    
    if (brighter >= 9 || darker >= 9) {
        const int idx = atomicAdd(keypoint_count, 1);
        if (idx < max_keypoints) {
            keypoints[idx] = {static_cast<float>(x), static_cast<float>(y)};
        }
    }
}

extern "C" {

int cuda_init_detector(int max_width, int max_height) {
    max_frame_width = max_width;
    max_frame_height = max_height;
    
    // Allocate device memory
   if (cudaMalloc(&d_keypoint_count, sizeof(int)) != cudaSuccess) {
        return -1;
    }
    if (cudaMalloc(&d_keypoints, 250000 * sizeof(KeyPoint)) != cudaSuccess) {
        cudaFree(d_keypoint_count);
        return -1;
    }

    return 0;
}

int cuda_detect_keypoints(
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    int width,
    int height,
    int y_linesize,
    int uv_linesize,
    uint8_t threshold,
    KeyPoint* keypoints,
    int max_keypoints,
    int* num_keypoints
) {
    if (width > max_frame_width || height > max_frame_height) return -1;
        
    // Reset keypoint counter
     // Reset keypoint counter
    cudaError_t error = cudaMemset(d_keypoint_count, 0, sizeof(int));
    if (error != cudaSuccess) {
        return -1;
    }

    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    detectFASTKeypoints<<<grid, block>>>(
        y_plane,
        width,
        height,
        y_linesize,
        threshold,
        d_keypoints,
        d_keypoint_count,
        max_keypoints
    );

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return -1;
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        return -1;
    }

    error = cudaMemcpy(num_keypoints, d_keypoint_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        return -1;
    }
    
    if (*num_keypoints > max_keypoints) *num_keypoints = max_keypoints;

    error = cudaMemcpy(keypoints, d_keypoints, *num_keypoints * sizeof(KeyPoint), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        return -1;
    }
    
    return 0;
}

void cuda_cleanup_detector(void) {
    if (d_keypoint_count) cudaFree(d_keypoint_count);
    if (d_keypoints) cudaFree(d_keypoints);
    
    d_keypoint_count = nullptr;
    d_keypoints = nullptr;
}

}