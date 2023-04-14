#pragma once

// CUDA Headers
#include <cuda_fp16.h>

typedef __half nn_float;

#define BLOCK_SIZE      256