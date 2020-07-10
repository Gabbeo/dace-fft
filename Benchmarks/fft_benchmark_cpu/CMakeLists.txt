cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(dace_fft_benchmark)
find_package(OpenMP REQUIRED)

add_executable(dace_fft_benchmark src/main.cpp src/fftw_test.cpp src/utils.cpp)

target_link_libraries(dace_fft_benchmark ${OpenMP_CXX_FLAGS})
target_link_libraries(dace_fft_benchmark fftw3 fftw3f fftw3_omp fftw3f_omp)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")






