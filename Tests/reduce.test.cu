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
#include "AI_Art_Generator/Utils/error_check_cuda.hpp"

__global__ void reduce_test(nn_float *gpu_test_data, int size, float multiplier)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    if (idx >= size) {
        return;
    }

    if (idx == 0) {
        for (int i = 0; i < size; i++) {
            gpu_test_data[i] = make_nn_precision(multiplier);
        }
    }

    g.sync();

    sum_in_place_up_to_4000_elements(gpu_test_data, size);
    nn_float result = gpu_test_data[0];

    if (idx == 0) {
        float expected_result = size * multiplier;
        float actual_result = make_full_precision(result);

        if (expected_result == actual_result) {
            printf("PASS: Reduce test excellent.          Size: %6d. Multiplier: %2.2f. Expected result: %2.2f. Actual result: %2.2f\n", size, multiplier, expected_result, actual_result);
        }
        else {
            printf("!!!FAIL!!! Reduce test is disappoint. Size: %6d. Multiplier: %2.2f. Expected result: %2.2f. Actual result: %2.2f\n", size, multiplier, expected_result, actual_result);
        }
    }
}

void summation_test(int size, float multiplier)
{
    nn_float *gpu_test_data;
    cuda_check(cudaMalloc(&gpu_test_data, sizeof(nn_float) * size));

    void* kernel_arguments[] = { (void*)&gpu_test_data, (void*)&size, (void*)&multiplier };

    cudaLaunchCooperativeKernel((void*)reduce_test, num_blocks(size), BLOCK_SIZE, kernel_arguments);
    cuda_check_sync();

    cuda_check(cudaFree(gpu_test_data));
}

int main()
{
    std::cout << "Test_All begin" << std::endl;

    for (int i = 0; i < 3; i++) {
        float multiplier = 10.f + 5.f * i;

        summation_test(1 <<  0,     multiplier);
        summation_test(1 <<  1,     multiplier);
        summation_test(1 <<  2,     multiplier);
        summation_test(1 <<  3,     multiplier);
        summation_test(1 <<  4,     multiplier);
        summation_test(1 <<  5,     multiplier);
        summation_test(1 <<  6,     multiplier);
        summation_test(1 <<  7,     multiplier);
        summation_test(1 <<  8,     multiplier);
        summation_test(1 <<  9,     multiplier);
        summation_test(1 << 10,     multiplier);
        summation_test(1 << 11,     multiplier);
        summation_test(1 << 12,     multiplier);
        summation_test(1 << 13,     multiplier);
        summation_test(100,         multiplier);
        summation_test(1000,        multiplier);
        summation_test(2000,        multiplier);
        summation_test(4000,        multiplier);
        summation_test(4095,        multiplier);
        summation_test(5000,        multiplier);
        summation_test(8000,        multiplier);
        summation_test(10000,       multiplier);
        summation_test(100000,      multiplier);
        summation_test(1000000,     multiplier);
        summation_test(10000000,    multiplier);
    }

    std::cout << "Test_All end" << std::endl;
    return 0;
}
