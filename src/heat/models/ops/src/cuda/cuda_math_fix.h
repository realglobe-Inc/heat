#ifndef CUDA_MATH_FIX_H
#define CUDA_MATH_FIX_H

#ifdef __CUDACC__
#include <cuda_runtime.h>

// Redirect CUDA's math functions to avoid conflict with glibc headers
#define rsqrt __cuda_rsqrt_hidden
#define rsqrtf __cuda_rsqrtf_hidden
#define sinpi __cuda_sinpi_hidden
#define cospi __cuda_cospi_hidden
#define sinpif __cuda_sinpif_hidden
#define cospif __cuda_cospif_hidden

#include <crt/math_functions.h>

#undef rsqrt
#undef rsqrtf
#undef sinpi
#undef cospi
#undef sinpif
#undef cospif
#endif

#endif
