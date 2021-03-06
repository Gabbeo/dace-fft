#include <iostream>
#include <cstring>
#include <complex>
#include <cstdlib>
#include <random>
#include <chrono>
#include <fstream>

#include <omp.h>
#include <fftw3.h>

#include "../include/fftw_test.h"
#include "../include/utils.h"

#define USE_SP 0
#if USE_SP
#define PRECISION float
#else
#define PRECISION double
#endif

int main() {
    std::cout << "%%% PROGRAM START %%%" << std::endl;
    omp_set_num_threads(24);

    // What ffts should be tested?
    
    // GPU lengths
    std::vector<size_t> lengths = {32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,81,243,729,2187,6561,19683,59049,125,625,3125,15625,36,216,1296,7776,46656,49,343,2401,16807,100,1000,10000,121,1331,14641,144,1728,20736,169,2197,28561,196,2744,38416,225,3375,50625,289,4913,324,5832,361,6859,400,8000,441,9261,484,10648,529,12167,576,13824,676,17576,784,21952,841,24389,900,27000,961,29791,33,1089,35937,34,1156,39304,35,1225,42875,37,1369,50653,38,1444,54872,39,1521,59319,40,1600,64000,41,1681,42,1764,43,1849,44,1936,45,2025,46,2116,47,2209,48,2304,50,2500,51,2601,52,2704,53,2809,54,2916,55,3025,56,3136,57,3249,58,3364,59,3481,60,3600,61,3721,62,3844,63,3969,65,4225,66,4356,67,4489,68,4624,69,4761,70,4900,71,5041,72,5184,73,5329,74,5476,75,5625,76,5776,77,5929,78,6084,79,6241,80,6400,82,6724,83,6889,84,7056,85,7225,86,7396,87,7569,88,7744,89,7921,90,8100,91,8281,92,8464,93,8649,94,8836,95,9025,96,9216,97,9409,98,9604,99,9801,101,10201,102,10404,103,10609,104,10816,105,11025,106,11236,107,11449,108,11664,109,11881,110,12100,111,12321,112,12544,113,12769,114,12996,115,13225,116,13456,117,13689,118,13924,119,14161,120,14400,122,14884,123,15129,124,15376,126,15876,127,16129,129,16641,130,16900,131,17161,132,17424,133,17689,134,17956,135,18225,136,18496,137,18769,138,19044,139,19321,140,19600,141,19881,142,20164,143,20449,145,21025,146,21316,147,21609,148,21904,149,22201,150,22500,151,22801,152,23104,153,23409,154,23716,155,24025,156,24336,157,24649,158,24964,159,25281,160,25600,161,25921,162,26244,163,26569,164,26896,165,27225,166,27556,167,27889,168,28224,170,28900,171,29241,172,29584,173,29929,174,30276,175,30625,176,30976,177,31329,178,31684,179,32041,180,32400,181,32761,182,33124,183,33489,184,33856,185,34225,186,34596,187,34969,188,35344,189,35721,190,36100,191,36481,192,36864,193,37249,194,37636,195,38025,197,38809,198,39204,199,39601,200,40000,201,40401,202,40804,203,41209,204,41616,205,42025,206,42436,207,42849,208,43264,209,43681,210,44100,211,44521,212,44944,213,45369,214,45796,215,46225,217,47089,218,47524,219,47961,220,48400,221,48841,222,49284,223,49729,224,50176,226,51076,227,51529,228,51984,229,52441,230,52900,231,53361,232,53824,233,54289,234,54756,235,55225,236,55696,237,56169,238,56644,239,57121,240,57600,241,58081,242,58564,244,59536,245,60025,246,60516,247,61009,248,61504,249,62001,250,62500,251,63001,252,63504,253,64009,254,64516,255,65025};
    // CPU lengths 
    //std::vector<size_t> lengths = {32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,100,121,125,128,144,169,196,216,225,243,256,289,324,343,361,400,441,484,512,529,576,625,676,729,784,841,900,961,1000,1024,1089,1156,1225,1296,1331,1369,1444,1521,1600,1681,1728,1764,1849,1936,2025,2116,2187,2197,2209,2304,2401,2500,2601,2704,2744,2809,2916,3025,3125,3136,3249,3364,3375,3481,3600,3721,3844,3969,4096,4225,4356,4489,4624,4761,4900,4913,5041,5184,5329,5476,5625,5776,5832,5929,6084,6241,6400,6561,6724,6859,6889,7056,7225,7396,7569,7744,7776,7921,8000,8100,8281,8464,8649,8836,9025,9216,9261,10000,10648,12167,13824,14641,15625,16384,16807,17576,19683,20736,21952,24389,27000,28561,29791,32768,35937,38416,39304,42875,46656,50625,50653,54872,59049,59319,64000,65536};
    
    std::vector<std::tuple<size_t,size_t>> dace = {};
    // Settings
    settings_s settings = settings_s();
    settings.NUM_ITERATIONS = 100; // Warm-up run excluded.
    settings.DEBUG = true;
    settings.FFT_LENGTHS = lengths;
    settings.DACE_SETTINGS = dace;

    settings.VAL_MAX = 10.0f;
    settings.VAL_MIN = -10.0f;
    settings.TOL_FP = 1e-3;
    settings.TOL_DP = 1e-7;

#if USE_SP
    std::cout << "%%% SINGLE PRECISION TESTING %%%" << std::endl;
#else
    std::cout << "%%% DOUBLE PRECISION TESTING %%%" << std::endl;
#endif

    /**
     * Test clock
     */
    {
        using namespace std::chrono;
        std::cout << "%%% START TESTING CLOCK %%%" << std::endl;
        const size_t n_clock_tests = 1000;
        std::vector<high_resolution_clock::time_point> clock_timings 
            = std::vector<high_resolution_clock::time_point>(n_clock_tests + 1); 

        for (size_t i = 0; i <= n_clock_tests; i++) {
            clock_timings[i] = high_resolution_clock::now();
        }

        duration<double> min_dur = std::chrono::seconds(10);
        duration<double> max_dur = std::chrono::seconds(0);
        duration<double> sum = std::chrono::nanoseconds(0);

        for (size_t i = 0; i < n_clock_tests; i++) {
            duration<double> dur = duration_cast<duration<double>>(
                    clock_timings[i+1] - clock_timings[i]);
            sum += dur;

            if (dur < min_dur) min_dur = dur;
            if (dur > max_dur) max_dur = dur;
        }

        double avg_granularity = sum.count() / (clock_timings.size() - 1);

        std::cout << "Minimum clock granularity: " << min_dur.count() * 1e9 
            << " nsec." << std::endl;
        std::cout << "Maximum clock granularity: " << max_dur.count() * 1e9 
            << " nsec." << std::endl;
        std::cout << "Average clock granularity: " << avg_granularity * 1e9
            << " nsec." << std::endl;
        std::cout << "%%% END TESTING CLOCK %%%" << std::endl << std::endl;
    }
    
    // Allocate memory
    std::cout << "%%% START DATA SETUP %%%" << std::endl;
    auto inputs = std::vector<std::complex<PRECISION>*>();
    auto outputs_dace_cpu = std::vector<std::complex<PRECISION>*>();
    auto outputs_fftw = std::vector<std::complex<PRECISION>*>();

    std::random_device rand_dev;
    std::mt19937 e2(rand_dev());
    std::uniform_real_distribution<PRECISION> distrib(settings.VAL_MIN, settings.VAL_MAX);

    // Allocate using fftw to ensure that data is aligned properly
    for (auto fft_length : settings.FFT_LENGTHS) {
#if USE_SP
        fftwf_complex* fftw_input = fftwf_alloc_complex(fft_length);
        fftwf_complex* fftw_output_fftw = fftwf_alloc_complex(fft_length);
#else
        fftw_complex* fftw_input = fftw_alloc_complex(fft_length);
        fftw_complex* fftw_output_fftw = fftw_alloc_complex(fft_length);
#endif
        // Zero all data.
        memset(fftw_input, 0.0f, 2 * sizeof(PRECISION) * fft_length);
        memset(fftw_output_fftw, 0.0f, 2 * sizeof(PRECISION) * fft_length);

        // Generate input data
        for (size_t i = 0; i < fft_length; i++) {
            fftw_input[i][0] = distrib(e2); // Real
            fftw_input[i][1] = distrib(e2); // Imaginary
        }

        if (settings.DEBUG && 0) {
            std::cout << "Input data:" << std::endl; 
            for (size_t i = 0; i < fft_length; i++) {
                std::cout << "y["<<i<<"]: " << fftw_input[i][0] << ", " << fftw_input[i][1] << std::endl; 
            }
        }

        // Cast to standardized format and pointer.
        inputs.push_back(reinterpret_cast<std::complex<PRECISION>*>(fftw_input));
        outputs_fftw.push_back(reinterpret_cast<std::complex<PRECISION>*>(fftw_output_fftw));
    }

    std::vector<benchmark_info> timings_fftw = std::vector<benchmark_info>();
    std::cout << "%%% END DATA SETUP %%%" << std::endl << std::endl;

    /**
     *  FFTW
     */
    std::cout << "%%% BEGIN TESTING FFTW %%%" << std::endl;
#if USE_SP
    fftw_float(inputs, outputs_fftw, timings_fftw, settings);
#else
    fftw_double(inputs, outputs_fftw, timings_fftw, settings);
#endif
    std::cout << "%%% END TESTING FFTW %%%" << std::endl << std::endl;

    // Write timings to file.
    std::ofstream fftw_results_file;
    fftw_results_file.open("results/fftw_results_file.csv");
    fftw_results_file << "n,avg_time,std_dev_time";
    for (size_t i = 1; i <= settings.NUM_ITERATIONS; i++) fftw_results_file << "," << "iteration" << i;
    fftw_results_file << std::endl;
    for (auto benchmark : timings_fftw) {
        fftw_results_file << benchmark.get_fft_length() 
            << "," << benchmark.t_avg() << "," << benchmark.t_sd();
        for (auto timing : benchmark.getTimings()) {
            fftw_results_file << "," << timing;
        }
        fftw_results_file << std::endl;
    }


    fftw_results_file.close();

    // Free pointers.
    for (auto ptr : inputs) fftw_free(ptr);
    for (auto ptr : outputs_fftw) fftw_free(ptr);

    std::cout << "%%% PROGRAM END %%%" << std::endl;
}
