#include "cuda_keypoint_detector.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>


// FAST circle offsets
__constant__ int2 fast_offsets[16] = {
    {3,  0},  {3,  1},  {2,  2},  {1,  3},
    {0,  3},  {-1, 3},  {-2, 2},  {-3, 1},
    {-3, 0},  {-3, -1}, {-2, -2}, {-1, -3},
    {0, -3},  {1, -3},  {2, -2},  {3,  -1}
};

// Device buffers
static int* d_keypoint_count = nullptr;
static CUDAGLResources gl_resources = {};


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
    int* keypoint_count,
    int max_keypoints,
    float image_width,
    float image_height
) {
    __shared__ int block_counter;
    __shared__ float2 block_keypoints[256]; // Adjust size based on block size

    if (threadIdx.x == 0) {
        block_counter = 0;
    }
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 3 || y < 3 || x >= width - 3 || y >= height - 3) return;

    const uint8_t center = y_plane[y * linesize + x];

    int brighter = 0;
    int darker = 0;

    for (int i = 0; i < 16; i++) {
        const int2 offset = fast_offsets[i];
        const uint8_t pixel = y_plane[(y + offset.y) * linesize + (x + offset.x)];

        if (pixel > center + threshold) brighter++;
        else if (pixel < center - threshold) darker++;
    }

    bool is_keypoint = (brighter >= 9 || darker >= 9);

    if (is_keypoint) {
        int local_idx = atomicAdd(&block_counter, 1);
        if (local_idx < 256) { // Ensure local storage doesn't overflow
            block_keypoints[local_idx] = make_float2(x, y);
        }
    }
    __syncthreads();

   if (threadIdx.x == 0 && block_counter > 0) {
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
            }
        }
    }
}

extern "C" {

int cuda_init_detector() {
    if (cudaMalloc(&d_keypoint_count, sizeof(int)) != cudaSuccess) {
        return -1;
    }
    return 0;
}

int cuda_register_gl_buffers(unsigned int position_buffer, unsigned int color_buffer, int max_keypoints) {
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

    // Check if the CUDA device supports OpenGL interoperation
    int concurrentManagedAccess;
    error = cudaDeviceGetAttribute(&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, cudaDevice);
    if (error != cudaSuccess || !concurrentManagedAccess) {
        printf("Device does not support CUDA-GL interop\n");
        return -1;
    }

    // Register position buffer
    error = cudaGraphicsGLRegisterBuffer(
        &gl_resources.position_resource,
        position_buffer,
        cudaGraphicsRegisterFlagsWriteDiscard
    );
    if (error != cudaSuccess) {
        printf("Error registering Position Buffer %d => %d\n", position_buffer, error);
        return -1;
    }

    // Register color buffer
    error = cudaGraphicsGLRegisterBuffer(
        &gl_resources.color_resource,
        color_buffer,
        cudaGraphicsRegisterFlagsWriteDiscard
    );
    if (error != cudaSuccess) {
        cudaGraphicsUnregisterResource(gl_resources.position_resource);
        printf("Error registering Color Buffer %d => %d\n", color_buffer,error);
        return -1;
    }

    gl_resources.buffer_size = max_keypoints;
    return 0;
}

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
) {
    cudaError_t error;
 
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Map GL buffers for CUDA access
    error = cudaGraphicsMapResources(1, &gl_resources.position_resource);
    if (error != cudaSuccess) {
        printf("Failed to Map Position Resources: %d\n", error);
        return -1;
    };

    error = cudaGraphicsMapResources(1, &gl_resources.color_resource);
    if (error != cudaSuccess) {
        printf("Failed to Map Color Resources: %d\n", error);
        cudaGraphicsUnmapResources(1, &gl_resources.position_resource);
        return -1;
    }
        
    size_t bytes;
    error = cudaGraphicsResourceGetMappedPointer(
        (void**)&gl_resources.d_positions,
        &bytes,
        gl_resources.position_resource
    );
    if (error != cudaSuccess) {
        printf("Failed to get Positions Mapped Pointer: %d\n", error);
        goto cleanup;
    }

    error = cudaGraphicsResourceGetMappedPointer(
        (void**)&gl_resources.d_colors,
        &bytes,
        gl_resources.color_resource
    );
    if (error != cudaSuccess) {
        printf("Failed to get Colors Mapped Pointer: %d\n", error);
        goto cleanup;
    }


    // Reset keypoint counter
    error = cudaMemset(d_keypoint_count, 0, sizeof(int));
    if (error != cudaSuccess) {
        printf("Failed to reset keypoint count: %d\n", error);
        goto cleanup;
    }

    // Launch kernel
    detectFASTKeypoints<<<grid, block>>>(
        y_plane,
        width,
        height,
        y_linesize,
        threshold,
        gl_resources.d_positions,
        gl_resources.d_colors,
        d_keypoint_count,
        gl_resources.buffer_size,
        image_width,
        image_height
    );

    cudaDeviceSynchronize();

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Keypoint Detection Kernel failed: %d\n", error);
        goto cleanup;
    }

    // Get keypoint count
    error = cudaMemcpy(num_keypoints, d_keypoint_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Failed to get copy keypoint count: %d\n", error);
        goto cleanup;
    }


    cleanup:
        cudaGraphicsUnmapResources(1, &gl_resources.position_resource);
        cudaGraphicsUnmapResources(1, &gl_resources.color_resource);
    
    return error == cudaSuccess ? 0 : -1;
}

void cuda_unregister_gl_buffers(void) {
    if (gl_resources.position_resource) {
        cudaGraphicsUnregisterResource(gl_resources.position_resource);
    }
    if (gl_resources.color_resource) {
        cudaGraphicsUnregisterResource(gl_resources.color_resource);
    }
    gl_resources = {};
}

void cuda_cleanup_detector(void) {
    cuda_unregister_gl_buffers();
    if (d_keypoint_count) cudaFree(d_keypoint_count);
    d_keypoint_count = nullptr;
}


}