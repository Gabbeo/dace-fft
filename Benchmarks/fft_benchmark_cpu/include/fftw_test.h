#ifndef FFTW_TEST_H
#define FFTW_TEST_H

#include <omp.h>
#include <fftw3.h>
#include <iostream>
#include <complex>
#include <vector>
#include "utils.h"
#include <chrono>
#include <cstring>


void fftw_float(std::vector<std::complex<float>*> inputs, std::vector<std::complex<float>*> outputs, std::vector<benchmark_info>& fftw_info, settings_s& settings);
void fftw_double(std::vector<std::complex<double>*> inputs, std::vector<std::complex<double>*> outputs, std::vector<benchmark_info>& fftw_info, settings_s& settings);

#endif 