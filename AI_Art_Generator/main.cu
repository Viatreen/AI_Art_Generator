// Standard Library
#include <stdio.h>
#include <iostream>

// Nvidia
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cooperative_groups.h>

// Project Headers
#include "AI_Art_Generator/Utils/error_check_cuda.hpp"

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
    int device_number = 0;
    int supports_coop_launch = 0;
    cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, device_number);

    if(!supports_coop_launch) {
        std::cout << "This GPU does not support cooperative groups" << std::endl;
        return 1;
    }
    else {
        std::cout << "This GPU supports cooperative groups" << std::endl;
    }

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_number);
    dim3 grid_dim(256, 1, 1);
    std::cout << "Blocks per grid: " << grid_dim.x << std::endl;
    dim3 block_dim(128, 1, 1);
    std::cout << "Threads per block: " << block_dim.x << std::endl;
    std::cout << "Total threads: " << grid_dim.x * block_dim.x << std::endl;
    int unused_param;   // unused_param - CUDA documentation does not inform how to launch a cooperative group kernel without function any arguments
    void *kernel_args[] = { (void*)&unused_param };
    cudaLaunchCooperativeKernel((void*)my_kernel, grid_dim, block_dim, kernel_args);

    cuda_check(cudaDeviceSynchronize());

    printf("Hello CUDA!\n");

    #ifdef WIN32
    system("Pause");
    #endif

    return 0;
}
