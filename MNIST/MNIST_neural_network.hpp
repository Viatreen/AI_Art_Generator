#pragma once

#include <cuda_fp16.h>

typedef __half nn_float;

#define BLOCK_SIZE      256

#define INPUT_DIM                     28
#define INPUT_SIZE                  ( INPUT_DIM * INPUT_DIM )
#define FIRST_LAYER_OUTPUT_DIM        26
#define FIRST_LAYER_OUTPUT_SIZE     ( FIRST_LAYER_OUTPUT_DIM * FIRST_LAYER_OUTPUT_DIM )
#define FIRST_LAYER_FILTER_DIM        3
#define FIRST_LAYER_FILTER_SIZE     ( FIRST_LAYER_FILTER_DIM * FIRST_LAYER_FILTER_DIM )
#define FIRST_LAYER_CHANNEL_AMOUNT    32

#define ACTIVATION_SLOPE_INV        ( nn_float(100) )
namespace neural_network {
struct filter3x3 {
    nn_float weight[FIRST_LAYER_FILTER_SIZE];
};

struct NN {
    nn_float  input[INPUT_SIZE];
    filter3x3 first_layer_filter[FIRST_LAYER_CHANNEL_AMOUNT];
    nn_float  first_layer_raw_output[FIRST_LAYER_OUTPUT_SIZE];
    nn_float  first_layer_norm_output[FIRST_LAYER_OUTPUT_SIZE];
    nn_float  first_layer_activation_output[FIRST_LAYER_OUTPUT_SIZE];
};

__global__ void populate(NN *nn);
__global__ void run(NN *nn);

} // namespace neural_network