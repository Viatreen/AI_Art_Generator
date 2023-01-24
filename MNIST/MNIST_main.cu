// Standard Library
#include <iostream>

// Project Headers
#include "MNIST\MNIST_neural_network.hpp"
#include "AI_Art_Generator\Utils\gpu_info_print.hpp"
int main()
{
    std::cout << "MNIST Begin" << std::endl;

    util::gpu_info_print();

    return 0;
}
