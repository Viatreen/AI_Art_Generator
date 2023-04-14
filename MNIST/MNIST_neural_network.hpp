#pragma once

// CUDA Headers
#include <cuda_fp16.h>

// Project Headers
#include "AI_Art_Generator/Utils/config.hpp"

#define INPUT_DIM                     28
#define INPUT_SIZE                  ( INPUT_DIM * INPUT_DIM )
#define FIRST_LAYER_FILTER_DIM        3
#define FIRST_LAYER_FILTER_SIZE     ( FIRST_LAYER_FILTER_DIM * FIRST_LAYER_FILTER_DIM )
#define FIRST_LAYER_OUTPUT_DIM      ( INPUT_DIM - FIRST_LAYER_FILTER_DIM + 1 )
#define FIRST_LAYER_OUTPUT_SIZE     ( FIRST_LAYER_OUTPUT_DIM * FIRST_LAYER_OUTPUT_DIM )
#define FIRST_LAYER_CHANNEL_AMOUNT    32

#define ACTIVATION_SLOPE            ( nn_float(1) / nn_float(128) )
#define ACTIVATION_SLOPE_INV        ( nn_float(1) / ACTIVATION_SLOPE )

namespace neural_network {

template <int dim, typename t_float>
struct filter {
    t_float weight[dim * dim];
};

template<typename t_float>
struct NN {
    t_float                                 input[INPUT_SIZE];
    filter<FIRST_LAYER_FILTER_DIM, t_float> first_layer_filter[FIRST_LAYER_CHANNEL_AMOUNT];
    t_float                                 first_layer_raw_output[FIRST_LAYER_OUTPUT_SIZE];
    t_float                                 first_layer_norm_output[FIRST_LAYER_OUTPUT_SIZE];
    t_float                                 first_layer_activation_output[FIRST_LAYER_OUTPUT_SIZE];
};

__global__ void populate(NN<nn_float> *nn);
__global__ void run(NN<nn_float> *nn);
__host__ void convert_half_2_float(std::unique_ptr<neural_network::NN<__half>>& in, std::unique_ptr<neural_network::NN<float>>& out);


} // namespace neural_network