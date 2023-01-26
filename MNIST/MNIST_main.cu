// Standard Library
#include <iostream>

// Project Headers
#include "AI_Art_Generator/Utils/gpu_info_print.hpp"
#include "AI_Art_Generator/Utils/config.hpp"
#include "AI_Art_Generator/Utils/cuda_helper.hpp"
#include "AI_Art_Generator/Utils/error_check_cuda.hpp"
#include "AI_Art_Generator/Utils/file_writer_neural_net.hpp"
#include "MNIST/MNIST_neural_network.hpp"

int main()
{
    std::cout << "MNIST Begin" << std::endl;

    int status = util::gpu_info_print();
    if (status) {
        return status; // No need for print statement. Print comes from within gpu_info_print
    }

    neural_network::NN<nn_float> *nn;

    cudaMalloc(&nn, sizeof(neural_network::NN<nn_float>));

    neural_network::populate<<<num_blocks(INPUT_SIZE), BLOCK_SIZE>>>(nn);
    cuda_check_sync;
 
    void *kernel_args[] = { (void*)&nn };
    cudaLaunchCooperativeKernel((void*)neural_network::run, num_blocks(FIRST_LAYER_OUTPUT_SIZE), BLOCK_SIZE, kernel_args);
    cuda_check_sync;

    util::write_2_csv(nn);

    return 0;
}
