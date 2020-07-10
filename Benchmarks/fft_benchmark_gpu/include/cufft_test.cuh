#ifndef CUFFT_TEST_CUH
#define CUFFT_TEST_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <complex>
#include "utils.h"
#include <vector>
#include <chrono>

void cufft_float(std::vector<std::complex<float>*> inputs, std::vector<std::complex<float>*> outputs, std::vector<benchmark_info>& cufft_info, settings_s& settings);
void cufft_double(std::vector<std::complex<double>*> inputs, std::vector<std::complex<double>*> outputs, std::vector<benchmark_info>& cufft_info, settings_s& settings);

#endif