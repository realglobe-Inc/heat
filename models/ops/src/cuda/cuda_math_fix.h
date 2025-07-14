// CUDA 12.x compatibility fix for math function conflicts
#ifndef CUDA_MATH_FIX_H
#define CUDA_MATH_FIX_H

// Prevent system math headers from being included
#define _BITS_MATHCALLS_H 1
#define __MATHCALLS_DECLARED 1

// Include CUDA headers first
#include <cuda_runtime.h>
#include <cuda.h>

// Prevent system math.h from declaring conflicting functions
#define sinpi __sinpi
#define cospi __cospi
#define sinpif __sinpif
#define cospif __cospif

// Undefine the guards so other headers can still include math functions
#undef _BITS_MATHCALLS_H
#undef __MATHCALLS_DECLARED

#endif // CUDA_MATH_FIX_H
