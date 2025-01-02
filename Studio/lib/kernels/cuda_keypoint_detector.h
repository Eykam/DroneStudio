#ifndef CUDA_KEYPOINT_DETECTOR_H
#define CUDA_KEYPOINT_DETECTOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Match your Zig struct
typedef struct {
    float x;
    float y;
} KeyPoint;

// Initialize CUDA resources
int cuda_init_detector(int max_width, int max_height, int max_keypoints);

// Process frame and detect keypoints
int cuda_detect_keypoints(
    const uint8_t* y_plane,      // Y plane data
    const uint8_t* uv_plane,     // UV plane data
    int width,                   // Frame width
    int height,                  // Frame height
    int y_linesize,             // Y plane linesize
    int uv_linesize,            // UV plane linesize
    uint8_t threshold,          // FAST detection threshold
    KeyPoint* keypoints,        // Output buffer for keypoints
    int max_keypoints,          // Maximum number of keypoints to detect
    int* num_keypoints          // Actual number of keypoints detected
);

// Clean up CUDA resources
void cuda_cleanup_detector(void);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KEYPOINT_DETECTOR_H