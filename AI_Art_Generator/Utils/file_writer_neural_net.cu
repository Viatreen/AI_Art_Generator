// Standard Library
#include <algorithm>
#include <memory>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

// CUDA
#include <cuda_fp16.h>

// Project Headers
#include "AI_Art_Generator/Utils/config.hpp"

// File Header
#include "AI_Art_Generator/Utils/file_writer_neural_net.hpp"

// Project Headers
#include "MNIST/MNIST_neural_network.hpp"

#define DATA_PRINT_GAP   3

namespace util {

__host__ void write_2_csv(neural_network::NN<nn_float> *nn)
{
    std::unique_ptr<neural_network::NN<nn_float>> half_nn(new neural_network::NN<nn_float>);
    cudaMemcpy(half_nn.get(), nn, sizeof(neural_network::NN<nn_float>), cudaMemcpyDeviceToHost);

    std::unique_ptr<neural_network::NN<float>> full_nn(new neural_network::NN<float>);

    neural_network::convert_half_2_float(half_nn, full_nn);
    half_nn.reset();

    // Create and timestamp file
    time_t raw_time;
    tm* time_info;
    time(&raw_time);
    time_info = localtime(&raw_time);
    std::cout << "Writing neural network information to csv" << std::endl;
    std::cout << std::asctime(time_info);

    //boost::filesystem::path Destination = "Saves";
    //boost::filesystem::create_directory(Destination);

    std::stringstream filename_stream;

    // TODO: Check how to include a folder in a git commit but not its contents
    #ifdef _WIN32
        filename_stream << "Saves\\";
    #endif

    #ifdef __linux
        filename_stream << "Saves/";
    #endif

    filename_stream << "MNIST_NN_" << time_info->tm_year + 1900 << std::setw(2) << std::setfill('0')
    << time_info->tm_mon + 1 << std::setw(2) << time_info->tm_mday << "_" << std::setw(2) << time_info->tm_hour
    << std::setw(2) << time_info->tm_min << std::setw(2) << time_info->tm_sec << ".csv";



    std::stringstream csv_file_contents;
    // Print data labels
    csv_file_contents << "Input:,";
    for (size_t i = 0; i < INPUT_DIM + DATA_PRINT_GAP - 1; i++)
    {
        csv_file_contents << ",";
    }

    csv_file_contents << "1st:,";
    for (size_t i = 0; i < FIRST_LAYER_FILTER_DIM + DATA_PRINT_GAP - 1; i++)
    {
        csv_file_contents << ",";
    }

    csv_file_contents << "Raw:,";
    for (size_t i = 0; i < FIRST_LAYER_OUTPUT_DIM + DATA_PRINT_GAP - 1; i++)
    {
        csv_file_contents << ",";
    }

    csv_file_contents << "Activated:\n";

    // Print data
    size_t max_dim = std::max({INPUT_DIM, (FIRST_LAYER_FILTER_DIM + 1) * FIRST_LAYER_CHANNEL_AMOUNT - 1, FIRST_LAYER_OUTPUT_DIM});
    for (size_t row = 0; row < max_dim; row++)
    {
        size_t column_amount = INPUT_DIM + DATA_PRINT_GAP + FIRST_LAYER_FILTER_DIM + DATA_PRINT_GAP + FIRST_LAYER_OUTPUT_DIM + DATA_PRINT_GAP + FIRST_LAYER_OUTPUT_DIM;
        for (size_t column = 0; column < column_amount; column++)
        {

            int layer_1_data_gap_1      = INPUT_DIM + DATA_PRINT_GAP;
            int layer_1_filter_col      = layer_1_data_gap_1 + FIRST_LAYER_FILTER_DIM;
            int layer_1_data_gap_2      = layer_1_filter_col + DATA_PRINT_GAP;
            int layer_1_raw_col         = layer_1_data_gap_2 + FIRST_LAYER_OUTPUT_DIM;
            int layer_1_data_gap_3      = layer_1_raw_col + DATA_PRINT_GAP;
            int layer_1_activated_col   = layer_1_data_gap_3 + FIRST_LAYER_OUTPUT_DIM;

            // Input Array
            if (column < INPUT_DIM) {
                if (row < INPUT_DIM) {
                    csv_file_contents << full_nn->input[INPUT_DIM * row + column] << ",";
                }
                else {
                    csv_file_contents << ",";
                }
            }

            // Data gap
            else if (column < layer_1_data_gap_1) {
                csv_file_contents << ",";
            }

            // Filters
            else if (column < layer_1_filter_col) {
                if ((row != 0 && (row + 1) % (FIRST_LAYER_FILTER_DIM + 1) == 0) || (row > (FIRST_LAYER_FILTER_DIM + 1) * FIRST_LAYER_CHANNEL_AMOUNT - 1)) {
                    csv_file_contents << ",";
                }
                else {
                    int channel_index = row / (FIRST_LAYER_FILTER_DIM + 1);
                    int weight_index_x = (column - (layer_1_data_gap_1));
                    int weight_index_y = row % (FIRST_LAYER_FILTER_DIM + 1);
                    int weight_index = FIRST_LAYER_FILTER_DIM * weight_index_y + weight_index_x;

                    csv_file_contents << full_nn->first_layer_filter[channel_index].weight[weight_index] << ",";
                }
            }

            // Data gap
            else if (column < layer_1_data_gap_2) {
                csv_file_contents << ",";
            }

            // Raw output data
            else if (column < layer_1_raw_col) {
                if (row < FIRST_LAYER_OUTPUT_DIM) {
                    int index = column - (layer_1_data_gap_2) + FIRST_LAYER_OUTPUT_DIM * row;

                    csv_file_contents << full_nn->first_layer_raw_output[index] << ",";
                }
                else {
                    csv_file_contents << ",";
                }
            }

            // Data gap
            else if (column < layer_1_data_gap_3) {
                csv_file_contents << ",";
            }

            // Activated output data
            else if (column < layer_1_activated_col) {
                if (row < FIRST_LAYER_OUTPUT_DIM) {
                    int index = column - (layer_1_data_gap_3) + FIRST_LAYER_OUTPUT_DIM * row;

                    csv_file_contents << full_nn->first_layer_activation_output[index] << ",";
                }
                else {
                    csv_file_contents << ",";
                }
            }

            else {
                csv_file_contents << ",";
            }


        }
        csv_file_contents << "\n";
    }
    

    std::ofstream csv_file;
    csv_file.open(filename_stream.str());
    if (!csv_file.is_open()) {
        std::cout << "ERROR: Could not open file: " << filename_stream.str() << std::endl;
        return;
    }

    csv_file << csv_file_contents.str();

    csv_file.close();


}

}   // namespace util

