// Standard Library
#include <stdio.h>
#include <iostream>

// Nvidia
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

// Project Headers
#include "Algos/reduce.hpp"
#include "AI_Art_Generator/Utils/cuda_helper.hpp"
#include "AI_Art_Generator/Utils/config.hpp"

__global__ void reduce_test(int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    if (idx >= size) {
        return;
    }

    nn_float *gpu_test_data;
    if (idx == 0) {
        gpu_test_data = new nn_float(size);

        for (int i = 0; i < size; i++) {
            gpu_test_data[i] = 1;
        }
    }

    g.sync();

    nn_float result = reduce(gpu_test_data, size);

    if (idx == 0) {
        if (result == __float2half(0.f)) {
            printf("Reduce test pleases me\n");
        }
        else {
            printf("Reduce test is disappoint\n");
        }

        delete gpu_test_data;
    }
}

int main()
{
    std::cout << "Test_All begin" << std::endl;


    int size = 100;
    void* kernel_arguments[] = { (void*)&size };

    cudaLaunchCooperativeKernel((void*)reduce_test, num_blocks(size), BLOCK_SIZE, kernel_arguments);

    std::cout << "Test_All end" << std::endl;
    return 0;
}