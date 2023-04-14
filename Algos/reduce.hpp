#pragma once

// Project Headers
#include "AI_Art_Generator/Utils/config.hpp"

__device__ inline nn_float reduce(nn_float *in, size_t size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    return 0;
}