#pragma once

#include <driver_types.h>

#ifndef _DEBUG

#define cuda_check(x) (x)
#define cuda_check_sync()

#else

#define cuda_check(x) cuda_check_expanded(x, #x, __FILE__, __LINE__)
#define cuda_check_sync() cuda_check(cudaDeviceSynchronize())

#endif  // _DEBUG

void cuda_check_expanded(cudaError_t result, const char *function_name, const char *filename, int line_number);
