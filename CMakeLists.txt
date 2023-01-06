cmake_minimum_required(VERSION 3.17)

# if (MSVC)
#   set(CudaToolkitDir "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0")
#   set(CMAKE_CUDA_COMPILER "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin\\nvcc.exe")
#   set(CMAKE_CXX_COMPILER "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.34.31933\\bin\\Hostx64\\x64\\cl.exe")
#   set(CUDA_TOOLKIT_ROOT_DIR "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0")
# endif()

project(AI_Art_Generator LANGUAGES CXX CUDA)
message("Project name: " ${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if (MSVC)
    set(_CRT_SECURE_NO_WARNINGS 1)
    set(CURL_STATICLIB 1)
endif()

include_directories(
    /
)

set(sources
    AI_Art_Generator/main.cu
)

add_executable(${CMAKE_PROJECT_NAME} ${sources})

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr> # -g -G --ptxas-options=–allow-expensive-optimizations=true>
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75) # 61 = GTX 1080 Ti

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
