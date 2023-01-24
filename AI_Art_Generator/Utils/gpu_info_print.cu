// Standard Library
#include <iostream>

namespace util
{

int gpu_info_print()
{
    int device_number = 0;
    int supports_coop_launch = 0;
    cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, device_number);

    if(!supports_coop_launch) {
        std::cout << "This GPU does not support cooperative groups" << std::endl;
        return 1;
    }
    else {
        std::cout << "This GPU supports cooperative groups" << std::endl;
    }

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, device_number);

    return 0;
}

}