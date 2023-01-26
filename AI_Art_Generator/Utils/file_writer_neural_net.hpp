#pragma once

// Project Headers
#include "MNIST/MNIST_neural_network.hpp"


namespace util {

__host__ void write_2_csv(neural_network::NN<__half> *nn);

}   // namespace util
