

// Nvidia
#include <cuda_runtime.h>
#include <cooperative_groups.h>


// File Header
#include "MNIST/MNIST_neural_network.hpp"

namespace neural_network {

#ifdef ACTIVATION_NOTHING
__device__ inline nn_float activate(nn_float in)
{
    return in;
}
#elif ACTIVATION_MICK_RELU
__device__ inline nn_float activate(nn_float in)
{
    if(in < nn_float(1) && in > nn_float(-1)) {
        return in;
    }
    else if(in > nn_float(1)) {
        return ACTIVATION_SLOPE * in + ( nn_float(1) - ACTIVATION_SLOPE);
    }
    else {
        return ACTIVATION_SLOPE * in - ( nn_float(1) - ACTIVATION_SLOPE);
    }
}
#endif

__device__ inline nn_float reduce(nn_float *in, size_t size)
{
    
}

__global__ void populate(NN<nn_float> *nn)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < INPUT_SIZE) {
        nn->input[idx] = nn_float(0.01 * idx);
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

__global__ void run(NN<nn_float> *nn)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    // TODO: Optimize this
    // TODO: Create improved summation function. Optimize it too
    for (int channel_index = 0; channel_index < FIRST_LAYER_CHANNEL_AMOUNT; channel_index++) {
        #pragma unroll
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
        nn->first_layer_activation_output[idx] = activate(nn->first_layer_raw_output[idx]);
    }

    g.sync();

    return;
}

__host__ void convert_half_2_float(std::unique_ptr<neural_network::NN<__half>>& in, std::unique_ptr<neural_network::NN<float>>& out)
{
    for (size_t i = 0; i < INPUT_SIZE; i++)
    {
        out->input[i] = __half2float(in->input[i]);
    }

    for (size_t i = 0; i < FIRST_LAYER_CHANNEL_AMOUNT; i++)
    {
        for (size_t j = 0; j < FIRST_LAYER_FILTER_SIZE; j++)
        {
            out->first_layer_filter[i].weight[j] = __half2float(in->first_layer_filter[i].weight[j]);
        }
    }

    for (size_t i = 0; i < FIRST_LAYER_OUTPUT_SIZE; i++)
    {
        out->first_layer_raw_output[i] = __half2float(in->first_layer_raw_output[i]);
        out->first_layer_norm_output[i] = __half2float(in->first_layer_norm_output[i]);
        out->first_layer_activation_output[i] = __half2float(in->first_layer_activation_output[i]);
    }

    return;
}


} // Namespace neural_network
