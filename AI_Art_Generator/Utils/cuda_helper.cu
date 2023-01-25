

// File Header
#include "AI_Art_Generator/Utils/cuda_helper.hpp"

// Project Size
#include "AI_Art_Generator/Utils/config.hpp"

int num_blocks(int num_threads)
{
    return (num_threads - 1) / BLOCK_SIZE + 1;
}