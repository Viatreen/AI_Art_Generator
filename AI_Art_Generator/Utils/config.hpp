#pragma once

// CUDA Headers
#include <cuda_fp16.h>

#ifdef NN_HALF_FLOAT
typedef __half nn_float;
#define make_full_precision(x)  __half2float(x)
#define make_nn_precision(x)    __float2half(x)
#endif

#ifdef NN_FULL_FLOAT
typedef float nn_float;
#define make_full_precision(x)  (x)
#define make_nn_precision(x)    (x)
#endif

#define BLOCK_SIZE      256