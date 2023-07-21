#pragma once

// Project Headers
#include "AI_Art_Generator/Utils/config.hpp"

__device__ inline void sum_in_place_up_to_4000_elements(nn_float *in, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= size) {
        return;
    }

    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    int log2_of_size = 31 - __clz(size);
    int largest_power_of_2 = 1 << log2_of_size;

    if (idx < size - largest_power_of_2) {
        in[idx] = in[idx] + in[idx + largest_power_of_2];
    }

    g.sync();

    #pragma unroll
    for (int i = largest_power_of_2 / 2; i > 0; i >>= 1) {
        if (idx < i) {
            in[idx] = in[idx] + in[idx + i];
        }
        g.sync();
    }
}