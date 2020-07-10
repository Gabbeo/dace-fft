#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <tuple>

#include <sys/time.h>

class benchmark_info {
private:
    size_t fft_length;
    long flop_count;
    size_t r, k;       // Only for DaCe implementation
    std::vector<double> timings, flops;
public:
    benchmark_info(size_t fft_length, long flop_count, size_t r, size_t k);
    benchmark_info(size_t fft_length, long flop_count);
    void add_timing(double time);
    double t_avg();
    double t_sd();
    double t_min();
    double t_max();
    double flops_avg();
    double flops_sd();
    double flops_min();
    double flops_max();
    const std::vector<double>& getTimings();
    const std::vector<double>& getFlops();
    size_t get_fft_length();
    size_t get_r();
    size_t get_k();
};

struct settings_s {
    std::vector<std::tuple<size_t, size_t>> DACE_SETTINGS;
    std::vector<size_t> FFT_LENGTHS;
    size_t NUM_ITERATIONS; // Warm-up run excluded.
    size_t N_STEPS;

    double VAL_MAX;
    double VAL_MIN;
    double TOL_FP;
    float TOL_DP;

    bool DEBUG;

    settings_s() : DACE_SETTINGS{std::tuple<size_t, size_t>(2,2)},
        FFT_LENGTHS{4}, NUM_ITERATIONS(100), VAL_MAX(10.0f), 
        VAL_MIN(-10.0f), TOL_FP(1e-7), TOL_DP(1e-14), DEBUG(false) {}
};

#endif