// Standard Library
#include <iostream>

// Project Headers
#include "MNIST/MNIST_neural_network.hpp"
#include "AI_Art_Generator/Utils/gpu_info_print.hpp"
#include "AI_Art_Generator/Utils/cuda_helper.hpp"

int main()
{
    std::cout << "MNIST Begin" << std::endl;

    int status = util::gpu_info_print();
    if (status) {
        return status; // No need for print statement. Print comes from within gpu_info_print
    }

    neural_network::NN *nn;

    cudaMalloc(&nn, sizeof(neural_network::NN));

    neural_network::populate<<<number_blocks(INPUT_SIZE), BLOCK_SIZE>>>(nn);
    cuda_check_sync;
 
    void *kernel_args[] = { (void*)&nn };
    cudaLaunchCooperativeKernel((void*)neural_network::run, number_blocks(FIRST_LAYER_OUTPUT_SIZE), BLOCK_SIZE, kernel_args);
    cuda_check_sync;

    return 0;
}
