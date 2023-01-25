// Standard Library
#include <stdio.h>
#include <iostream>

// Nvidia
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cooperative_groups.h>

// Project Headers
#include "AI_Art_Generator/Utils/error_check_cuda.hpp"
#include "AI_Art_Generator/Utils/gpu_info_print.hpp"

__global__ void my_kernel(int unused_param)  // unused_param - CUDA documentation does not inform how to launch a cooperative group kernel without function any arguments
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // cooperative_groups::grid_group g = cooperative_groups::this_grid();
    // printf("0 Thread ID: %d\n", idx);

    // if (idx % 32 == 0) {
    printf("1 Thread ID: %d\n", idx);
    // }

    // g.sync();

    // if (idx % 32 == 0) {
    printf("2 Thread ID: %d\n", idx);
    // }

    return;
}

int main(void)
{
    std::cout << "AI_Art Begin" << std::endl;

    int status = util::gpu_info_print();
    if (status) {
        return status; // No need for print statement. Print comes from within gpu_info_print
    }

    dim3 grid_dim(128, 1, 1);
    std::cout << "Blocks per grid: " << grid_dim.x << std::endl;
    dim3 block_dim(128, 1, 1);
    std::cout << "Threads per block: " << block_dim.x << std::endl;
    std::cout << "Total threads: " << grid_dim.x * block_dim.x << std::endl;
    int unused_param;   // unused_param - CUDA documentation does not inform how to launch a cooperative group kernel without function any arguments
    void *kernel_args[] = { (void*)&unused_param };
    cudaLaunchCooperativeKernel((void*)my_kernel, grid_dim, block_dim, kernel_args);

    cuda_check(cudaDeviceSynchronize());

    printf("Hello CUDA!\n");

    return 0;
}
