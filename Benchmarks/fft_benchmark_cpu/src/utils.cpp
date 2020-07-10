#include "../include/utils.h"

benchmark_info::benchmark_info(size_t fft_length, long flop_count, size_t r, size_t k)
    : fft_length(fft_length), flop_count(flop_count), r(r), k(k) {}

benchmark_info::benchmark_info(size_t fft_length, long flop_count)
    : fft_length(fft_length), flop_count(flop_count) {}

void benchmark_info::add_timing(double time) {
    timings.push_back(time);
    if (std::isinf(time < 1e-10)) {
        std::cerr << "Recorded time span is too short!" << std::endl;
    }
    double flops_cur = flop_count / time;
    flops.push_back(flops_cur);
}

double benchmark_info::t_avg() {
    if (timings.size() > 0) {
        return std::accumulate(timings.begin(), timings.end(), 0.0) / ((double)timings.size());
    }
    else {
        return 0.0;
    }
}

double benchmark_info::t_sd() {
    double avg = this->t_avg();
    double accumulator = 0.0f;
    std::for_each (timings.begin(), timings.end(), [&](const double timing){
        accumulator += (timing - avg) * (timing - avg);
    });

    return sqrt(accumulator / ((double) timings.size() - 1));
}

double benchmark_info::t_min() {
    if (timings.size() > 0) {
        return *std::min_element(timings.begin(), timings.end());
    }
    else {
        return 0.0;
    }
}

double benchmark_info::t_max() {
    if (timings.size() > 0) {
        return *std::max_element(timings.begin(), timings.end());
    }
    else {
        return 0.0;
    }
}

double benchmark_info::flops_avg() {
    if (flops.size() > 0) {
        return std::accumulate(flops.begin(), flops.end(), 0.0) / ((double)flops.size());
    }
    else {
        return 0.0;
    }
}

double benchmark_info::flops_sd() {
    double avg = this->flops_avg();
    double accumulator = 0.0f;
    std::for_each (flops.begin(), flops.end(), [&](const double flop){
        accumulator += (flop - avg) * (flop - avg);
    });

    return sqrt(accumulator / ((double) flops.size() - 1));
}

double benchmark_info::flops_min() {
    if (flops.size() > 0) {
        return *std::min_element(flops.begin(), flops.end());
    }
    else {
        return 0.0;
    }
}

double benchmark_info::flops_max() {
    if (flops.size() > 0) {
        return *std::max_element(flops.begin(), flops.end());
    }
    else {
        return 0.0;
    }
}

const std::vector<double>& benchmark_info::getTimings() {
    return timings;
}

const std::vector<double>& benchmark_info::getFlops() {
    return flops;
}

size_t benchmark_info::get_fft_length() {
    return fft_length;
}

size_t benchmark_info::get_r() {
    return r;
}

size_t benchmark_info::get_k() {
    return k;
}