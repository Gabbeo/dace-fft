/**
    THIS FILE IS PROVIDED AS AN EXAMPLE
*/

#include <stdlib.h>
#include <dace/dace.h>
extern "C" {
#include "own_fft.h"
}
#include <cmath>
#include <iostream>
#include <iomanip>
#include <map>
#include <tuple>
#include <omp.h>
#include <fstream>
#include <random>
#include <fftw3.h>
#include <chrono>

int main(int argc, char** argv) {
    const size_t max_r = 256;
    const size_t max_k = 30;
    const size_t max_t = 24; // Dependent on system
    const size_t min_len = 32; // Should be equal to minimum in important_lengths.
    const size_t max_len = 2 * 32768; // Should be double maximum in important_lengths.
    const size_t repetitions = 100; // Number of times to run for average.
    const std::vector<size_t> important_lengths = 
        {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

    const bool debug = true;
    const bool debug2 = false;

    std::cout << "%%% PROGRAM START %%%" << std::endl;

    // File for raw measurements of the dace implementation.
    std::ofstream dace_res_file;
    dace_res_file.open("dace_benchmark_result.csv");
    dace_res_file << "r,k,n,number_of_threads,correct,avg_time,std_dev_time";
    for (size_t i = 1; i <= repetitions; i++) dace_res_file << ",time_iteration_" << i;
    dace_res_file << std::endl;

    // File for best dace results for each length. 
    std::ofstream dace_res_best_all_lengths;
    dace_res_best_all_lengths.open("dace_best_times_all_lengths.csv");

    // This file is a bit special.
    // For each length in important_lengths it finds the dace
    // combination that has the smallest runtime but a size 
    // equal or larger to the length
    std::ofstream dace_res_important_len;
    dace_res_important_len.open("dace_best_times_important_lengths.csv");

    //       N                  R,      K,     Threads, Avg time
    std::map<size_t, std::tuple<size_t, size_t, size_t, double>> dace_perf_map;
 
    for (size_t R = 2; R <= max_r; R++) {
        std::cout << "Running with radix: " << R << std::endl;
        for (size_t K = 1; K <= max_k; K++) {
            size_t N = pow(R, K);
            size_t N_div_R = pow(R, K-1);
            // Skip this iteration if too big/little.
            if (N < min_len || N > max_len) continue;
            
            dace::complex128* __restrict__ x = (dace::complex128*) calloc(N, sizeof(dace::complex128));
            dace::complex128* __restrict__ y_dace = (dace::complex128*) calloc(N, sizeof(dace::complex128));

            // Init the input data.
            std::random_device rand_dev;
            std::mt19937 e2(rand_dev());
            std::uniform_real_distribution<double> distrib(-10, 10); // Generate vals from -10.0 to 10.0
            for (int i = 0; i < N; i++) {
                x[i] = dace::complex128(distrib(e2), distrib(e2));
                if (debug2) std::cout << x[i] << std::endl;
            }

          
            
            // TEST DACE IMPLEMENTATION
            for (size_t T = 1; T <= max_t; T++) {
                omp_set_num_threads(T);

                // Time program
                std::vector<double> timings = std::vector<double>(); 
                std::chrono::high_resolution_clock::time_point t1, t2;
                __dace_init_own_fft(x, y_dace, K, N, N_div_R, R);
                __program_own_fft(x, y_dace, K, N, N_div_R, R); // Warmup
                for (size_t i = 0; i < repetitions; i++) {
                    t1 = std::chrono::high_resolution_clock::now();
                    __program_own_fft(x, y_dace, K, N, N_div_R, R);
                    t2 = std::chrono::high_resolution_clock::now();
                    double dur = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
                    timings.push_back(dur);
                }

                // Calculate statistics
                double avg = std::accumulate(timings.begin(), timings.end(), 0.0) / ((double)timings.size());
                double accumulator = 0.0f;
                std::for_each (timings.begin(), timings.end(), [&](const double timing){
                    accumulator += (timing - avg) * (timing - avg);
                });
                double sd = sqrt(accumulator / ((double) timings.size() - 1));
               bool corr = true;
                
                // Save if it's the current best result for the input length AND correct.
                if (corr && dace_perf_map.find(N) != dace_perf_map.end()) {
                    if (avg < std::get<3>(dace_perf_map[N])) {
                        dace_perf_map[N] = std::make_tuple(R, K, T, avg);
                    }
                }
                else {
                    dace_perf_map[N] = std::make_tuple(R, K, T, avg);
                }
                
                if (debug) {
                    std::cout << "R: " <<std::setw(3)<<std::left<< R << " K: " <<std::setw(3)<<std::left<< K 
                        << " N: " <<std::setw(5)<<std::left<< N << " T: " <<std::setw(5)<<std::left<< T
                        << " avg_time: " <<std::setw(11)<<std::left<< avg << " sd_time: " <<std::setw(11)<<std::left<< sd
                        << " correct: " << corr << std::endl;
                }

                dace_res_file << R << "," << K << "," << N << "," << T << ","
                            << corr << "," << avg << "," << sd;
                for (size_t i = 0; i < timings.size(); i++) dace_res_file << "," << timings[i];
                dace_res_file << std::endl;

                __dace_exit_own_fft(x, y_dace, K, N, N_div_R, R);
            }
            
            free(x);
            free(y_dace);

        }
    }


    dace_res_best_all_lengths << "r,k,n,number_of_threads,avg_time" << std::endl;
    std::map<size_t, std::tuple<size_t, size_t, size_t, double>> best_times_import_lens;
    std::cout << std::endl << "\nBEST TIMES FOR ALL SIZES" << std::endl;
    for (auto perf : dace_perf_map) {
        auto best = perf.second;
        int N = pow(std::get<0>(best), std::get<1>(best));

        std::cout << "R: " <<std::setw(3)<<std::left<< std::get<0>(best) << " K: " <<std::setw(3)<<std::left<< std::get<1>(best)
            << " N: " <<std::setw(5)<<std::left<< N << " T: " <<std::setw(5)<<std::left<< std::get<2>(best)
            << " avg_time: " <<std::setw(11)<<std::left<< std::get<3>(best) << std::endl;

        dace_res_best_all_lengths << std::get<0>(best) << "," << std::get<1>(best) << "," << N << "," << std::get<2>(best) << ","
            << std::get<3>(best) << std::endl;

        // If this is the fastest for an important length then add it.
        for (auto len : important_lengths) {
            if (N > len && best_times_import_lens.find(len) == best_times_import_lens.end()) {
                best_times_import_lens[len] = best;
            }
            else if (N > len && std::get<3>(best) < std::get<3>(best_times_import_lens[len])) {
                best_times_import_lens[len] = best;
            }
        }
    }

    dace_res_important_len << "dace_r,dace_k,dace_n,dace_num_threads,imp_len,dace_avg_time" << std::endl;  
    std::cout << std::endl << "\nBEST TIMES FOR IMPORTANT SIZES" << std::endl;
    for (auto perf : best_times_import_lens) {
        auto len = perf.first;
        auto best = perf.second;
        int N = pow(std::get<0>(best), std::get<1>(best));

        dace_res_important_len << std::get<0>(best) << "," << std::get<1>(best) << "," << N << "," 
            << std::get<2>(best) <<"," << len << "," << std::get<3>(best) << std::endl;

        std::cout << "R: " <<std::setw(3)<<std::left<< std::get<0>(best) << " K: " <<std::setw(3)<<std::left<< std::get<1>(best)
            << " N: " <<std::setw(5)<<std::left<< N << " Target len: " <<std::setw(5)<<std::left<< len 
            << " T: " <<std::setw(5)<<std::left<< std::get<2>(best)
            << " avg_time: " <<std::setw(11)<<std::left<< std::get<3>(best) << std::endl;
    }


    dace_res_best_all_lengths.close();
    dace_res_important_len.close();
    dace_res_file.close();
    return 0;
}
