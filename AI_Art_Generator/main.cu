// Standard Library
#include <stdio.h>
#include <iostream>

// Nvidia
#include <cuda_runtime.h>
#include <driver_types.h>

// Project Headers
#include "AI_Art_Generator/Utils/error_check_cuda.hpp"

__global__ void myKernel(void) {
    int idx = threadIdx.x;
    printf("Thread ID: %d\n", idx);

    return;
}

int main(void) {
    myKernel<<<1, 128>>>();
    cuda_check(cudaDeviceSynchronize());

    printf("Hello CUDA!\n");
    return 0;
}
