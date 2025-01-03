#include "keypoint_detector.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>

#define MAX_DETECTORS 16



// ============================================================= Detection =================================================================

// FAST circle offsets
__constant__ int2 fast_offsets[16] = {
    {3,  0},  {3,  1},  {2,  2},  {1,  3},
    {0,  3},  {-1, 3},  {-2, 2},  {-3, 1},
    {-3, 0},  {-3, -1}, {-2, -2}, {-1, -3},
    {0, -3},  {1, -3},  {2, -2},  {3,  -1}
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


// ============================================================= Matching =================================================================

// Matching parameters
struct MatchingParams {
    float baseline;
    float focal_length;
    float max_disparity;
    float epipolar_threshold;
};

// Structure to hold matched keypoint data
struct MatchedKeypoint {
    float3 left_pos;
    float3 right_pos;
    float3 world_pos;
    float disparity;
};

__global__ void matchKeypointsKernel(
    const float4* __restrict__ left_positions,
    const float4* __restrict__ right_positions,
    const int left_count,
    const int right_count,
    const MatchingParams params,
    MatchedKeypoint* matches,
    int* match_count,
    const int max_matches
) {
    const int left_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (left_idx >= left_count) return;

    __shared__ float min_costs[256]; 
    __shared__ int best_matches[256];

    min_costs[threadIdx.x] = INFINITY;
    best_matches[threadIdx.x] = -1;

    float3 left_pos = make_float3(
        left_positions[left_idx].x,
        left_positions[left_idx].y,
        left_positions[left_idx].z
    );

    // Find best match for this left keypoint
    for (int right_idx = 0; right_idx < right_count; right_idx++) {
        float3 right_pos = make_float3(
            right_positions[right_idx].x,
            right_positions[right_idx].y,
            right_positions[right_idx].z
        );

        // Check epipolar constraint
        float y_diff = fabsf(left_pos.y - right_pos.y);
        if (y_diff > params.epipolar_threshold) continue;

        // Calculate disparity (should be positive)
        float disparity = left_pos.x - right_pos.x;
        if (disparity <= 0 || disparity > params.max_disparity) continue;

        // Simple cost function based on y-difference
        float cost = y_diff;

        if (cost < min_costs[threadIdx.x]) {
            min_costs[threadIdx.x] = cost;
            best_matches[threadIdx.x] = right_idx;
        }
    }

    __syncthreads();

    // Store match if good enough
    if (best_matches[threadIdx.x] >= 0 && min_costs[threadIdx.x] < params.epipolar_threshold) {
        int match_idx = atomicAdd(match_count, 1);
        if (match_idx < max_matches) {
            float3 right_pos = make_float3(
                right_positions[best_matches[threadIdx.x]].x,
                right_positions[best_matches[threadIdx.x]].y,
                right_positions[best_matches[threadIdx.x]].z
            );
            
            float disparity = left_pos.x - right_pos.x;
            float depth = (params.baseline * params.focal_length) / disparity;
            
            // Calculate world position (using left camera as reference)
            float3 world_pos = make_float3(
                left_pos.x * depth / params.focal_length,
                left_pos.y * depth / params.focal_length,
                depth
            );

            matches[match_idx] = {
                left_pos,
                right_pos,
                world_pos,
                disparity
            };
        }
    }
}

// Kernel to generate visualization data
__global__ void generateVisualizationKernel(
    const MatchedKeypoint* matches,
    const int match_count,
    float4* positions,
    float4* colors,
    const float canvas_width,
    const float canvas_height
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= match_count) return;

    const MatchedKeypoint match = matches[idx];
    const int base_idx = idx * 3; // 3 vertices per match (triangle strip)

    // Center point (world position)
    positions[base_idx] = make_float4(match.world_pos.x, match.world_pos.y, match.world_pos.z, 1.0f);
    colors[base_idx] = make_float4(0.0f, 1.0f, 0.0f, 1.0f); // Green for center

    // Left keypoint
    positions[base_idx + 1] = make_float4(match.left_pos.x, match.left_pos.y, match.left_pos.z, 1.0f);
    colors[base_idx + 1] = make_float4(1.0f, 0.0f, 0.0f, 1.0f); // Red for left

    // Right keypoint
    positions[base_idx + 2] = make_float4(match.right_pos.x, match.right_pos.y, match.right_pos.z, 1.0f);
    colors[base_idx + 2] = make_float4(0.0f, 0.0f, 1.0f, 1.0f); // Blue for right
}



// ============================================================= Bindings =================================================================

extern "C" {

int cuda_create_detector(void) {
    int slot = find_free_detector_slot();
    if (slot < 0) {
        return -1;
    }

    if (cudaMalloc(&g_detectors[slot].d_keypoint_count, sizeof(int)) != cudaSuccess) {
        return -1;
    }

    g_detectors[slot].initialized = true;
    g_detectors[slot].id = g_next_detector_id++;
    return g_detectors[slot].id;
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

int cuda_detect_keypoints(
    int detector_id,
    uint8_t threshold,
    ImageParams* image
) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    
    if (!detector) return -1;

    cudaError_t error;
 
    dim3 block(16, 16);
    dim3 grid((image->width + block.x - 1) / block.x, (image->height + block.y - 1) / block.y);

    // Map GL buffers for CUDA access
    error = cudaGraphicsMapResources(1, &detector->gl_resources.position_resource);
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
        goto cleanup;
    }

    error = cudaGraphicsResourceGetMappedPointer(
        (void**)&detector->gl_resources.d_colors,
        &bytes,
        detector->gl_resources.color_resource
    );
    if (error != cudaSuccess) {
        printf("Failed to get Colors Mapped Pointer: %d\n", error);
        goto cleanup;
    }


    // Reset keypoint counter
    error = cudaMemset(detector->d_keypoint_count, 0, sizeof(int));
    if (error != cudaSuccess) {
        printf("Failed to reset keypoint count: %d\n", error);
        goto cleanup;
    }

    // Launch kernel
    detectFASTKeypoints<<<grid, block>>>(
        image->y_plane,
        image->width,
        image->height,
        image->y_linesize,
        threshold,
        detector->gl_resources.d_positions,
        detector->gl_resources.d_colors,
        detector->d_keypoint_count,
        detector->gl_resources.buffer_size,
        image->image_width,
        image->image_height
    );


    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Keypoint Detection Kernel failed: %d\n", error);
        goto cleanup;
    }


    printf("Copying keypoints for detector %d from %p to %p\n", 
    detector_id, 
    (void*)detector->d_keypoint_count,
    (void*)image->num_keypoints);

    // Get keypoint count
    error = cudaMemcpy(image->num_keypoints, detector->d_keypoint_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Failed to copy keypoint count: %d\n", error);
        goto cleanup;
    }

    printf("After copy: num_keypoints = %d\n", *image->num_keypoints);

    cleanup:
        cudaGraphicsUnmapResources(1, &detector->gl_resources.position_resource);
        cudaGraphicsUnmapResources(1, &detector->gl_resources.color_resource);
    
    return error == cudaSuccess ? 0 : -1;
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

void cuda_cleanup_detector(int detector_id) {
    DetectorInstance* detector = get_detector_instance(detector_id);
    if (!detector) return;

    cuda_unregister_gl_buffers(detector_id);
    if (detector->d_keypoint_count) cudaFree(detector->d_keypoint_count);
    detector->d_keypoint_count = nullptr;
}


int cuda_match_keypoints(
    int detector_id_left,
    int detector_id_right,
    float baseline,
    float focal_length,
    int* num_matches,
    uint8_t threshold,

    ImageParams* left,
    ImageParams* right
) {
    DetectorInstance* left_detector = get_detector_instance(detector_id_left);
    DetectorInstance* right_detector = get_detector_instance(detector_id_right);
    if (!left_detector || !right_detector) return -1;
  
    printf("Getting keypoints from left...\n");
    int result = cuda_detect_keypoints(
        detector_id_left,
        threshold,
        left
    );

    if (result < 0){
        printf("Failed to detect keypoints from left image");
    };

    printf("Getting keypoints from right...\n");
    result = cuda_detect_keypoints(
        detector_id_right,
        threshold,
        right
    );

    if (result < 0){
        printf("Failed to detect keypoints from right image");
    };

    cudaDeviceSynchronize();
    

    //Use left_detector as the basis for matching
    const int max_matches = (*left->num_keypoints);
    printf("Max matches allowed %d\n", max_matches);
   
    // Allocate device memory for matches
    MatchedKeypoint* d_matches;
    int* d_match_count;
    
    cudaMalloc(&d_matches, max_matches * sizeof(MatchedKeypoint));
    cudaMalloc(&d_match_count, sizeof(int));
    cudaMemset(d_match_count, 0, sizeof(int));

    // Set up matching parameters
    MatchingParams params = {
        .baseline = baseline,
        .focal_length = focal_length,
        .max_disparity = 100.0f,
        .epipolar_threshold = 2.0f
    };

    // Launch matching kernel
    dim3 block(256);
    dim3 grid((max_matches + block.x - 1) / block.x);
    
    // cudaError_t error;
    // error = cudaGraphicsMapResources(1, &left_detector->gl_resources.position_resource);
    // if (error != cudaSuccess) {
    //     printf("Failed to Map Position Resources: %d\n", error);
    //     return -1;
    // };
        
    // size_t bytes;
    // error = cudaGraphicsResourceGetMappedPointer(
    //     (void**)&left_detector->gl_resources.d_positions,
    //     &bytes,
    //     left_detector->gl_resources.position_resource
    // );
    // if (error != cudaSuccess) {
    //     printf("Failed to get Positions Mapped Pointer: %d\n", error);
    //     goto cleanup;
    // }


    // error = cudaGraphicsMapResources(1, &right_detector->gl_resources.position_resource);
    // if (error != cudaSuccess) {
    //     printf("Failed to Map Position Resources: %d\n", error);
    //     return -1;
    // };

    
    // error = cudaGraphicsResourceGetMappedPointer(
    //     (void**)&right_detector->gl_resources.d_positions,
    //     &bytes,
    //     right_detector->gl_resources.position_resource
    // );
    // if (error != cudaSuccess) {
    //     printf("Failed to get Positions Mapped Pointer: %d\n", error);
    //     goto cleanup;
    // }

 

    matchKeypointsKernel<<<grid, block>>>(
        left_detector->gl_resources.d_positions,
        right_detector->gl_resources.d_positions,
        *left->num_keypoints,
        *right->num_keypoints,
        params,
        d_matches,
        d_match_count,
        max_matches
    );

    // Get match count
    cudaMemcpy(num_matches, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);

    // // Generate visualization
    // generateVisualizationKernel<<<grid, block>>>(
    //     d_matches,
    //     *num_matches,
    //     detector->gl_resources.d_positions,
    //     detector->gl_resources.d_colors,
    //     detector->gl_resources.buffer_size,
    //     detector->gl_resources.buffer_size
    // );

    cudaFree(d_matches);
    cudaFree(d_match_count);

    // cleanup:
    //     cudaGraphicsUnmapResources(1, &left_detector->gl_resources.position_resource);
    //     cudaGraphicsUnmapResources(1, &right_detector->gl_resources.position_resource);
    
    // return error == cudaSuccess ? 0 : -1;
    return 0;
}

}