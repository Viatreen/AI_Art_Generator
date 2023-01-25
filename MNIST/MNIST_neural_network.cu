

// Nvidia
#include <cuda_runtime.h>
#include <cooperative_groups.h>


// File Header
#include "MNIST/MNIST_neural_network.hpp"

namespace neural_network {

__global__ void populate(NN *nn)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < INPUT_SIZE) {
        nn->input[idx] = nn_float(1);
    }

    if(idx < FIRST_LAYER_CHANNEL_AMOUNT * FIRST_LAYER_FILTER_SIZE) {
        int weight_index = idx % FIRST_LAYER_FILTER_SIZE;
        int channel_index = idx / FIRST_LAYER_FILTER_SIZE;
    
        nn->first_layer_filter[channel_index].weight[weight_index] = nn_float(1) / ( nn_float(FIRST_LAYER_FILTER_SIZE) + nn_float(1));
    }

    if(idx < FIRST_LAYER_OUTPUT_SIZE) {
        nn->first_layer_raw_output[idx] = nn_float(0);
        nn->first_layer_norm_output[idx] = nn_float(0);
        nn->first_layer_activation_output[idx] = nn_float(0);
    }

    return;
}

__global__ void run(NN *nn)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    for (int channel_index = 0; channel_index < FIRST_LAYER_CHANNEL_AMOUNT; channel_index++) {
        for (int weight_index = 0; weight_index < FIRST_LAYER_FILTER_SIZE; weight_index++) {
            if (idx < FIRST_LAYER_OUTPUT_SIZE) {
                int input_index_x = idx % FIRST_LAYER_OUTPUT_DIM + weight_index % FIRST_LAYER_FILTER_DIM;
                int input_index_y = idx / FIRST_LAYER_OUTPUT_DIM  + (weight_index / FIRST_LAYER_FILTER_DIM);

                int input_index = input_index_y * INPUT_DIM + input_index_x;

                if (input_index > INPUT_SIZE) {
                    printf("Thread ID: %d, intput size\n", idx);
                }

                nn->first_layer_raw_output[idx] += nn->input[input_index] * nn->first_layer_filter[channel_index].weight[weight_index];
            }

            g.sync();
        }
    }

    if (idx < FIRST_LAYER_OUTPUT_SIZE) {
        // nn->first_layer_raw_output[idx] = nn_float(5);
        printf("Output index %3d: %4.2f\n", idx, float(nn->first_layer_raw_output[idx]));
    }
}

} // Namespace neural_network
