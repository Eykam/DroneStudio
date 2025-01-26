// cuda_wrapper.h
#define __CUDA_ARCH__ 0
#define __device__
#define __host__
#define __global__
#define __forceinline__
#define __device_builtin__
#define __builtin_align__(x)
#define __align__(x)
#define struct___device_builtin__ struct
#define __CUDA_INTERNAL_COMPILATION

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
