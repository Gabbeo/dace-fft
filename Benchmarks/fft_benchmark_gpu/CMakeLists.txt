cmake_minimum_required(VERSION 3.14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(dace_fft_benchmark)
enable_language(CXX C CUDA)
find_package(OpenMP REQUIRED)

add_executable(dace_fft_benchmark src/main.cpp src/cufft_test.cu src/utils.cpp)

set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/lib /usr/local/cuda-10.2/lib64)
target_include_directories(dace_fft_benchmark PUBLIC /usr/local/cuda-10.2/include)

target_link_libraries(dace_fft_benchmark ${OpenMP_CXX_FLAGS})
target_link_libraries(dace_fft_benchmark cufft)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")






