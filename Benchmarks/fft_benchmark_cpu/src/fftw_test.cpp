#include "../include/fftw_test.h"

void fftw_float(std::vector<std::complex<float>*> inputs, std::vector<std::complex<float>*> outputs, std::vector<benchmark_info>& fftw_info, settings_s& settings) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());

    std::chrono::high_resolution_clock::time_point t1, t2;
    for (size_t i = 0; i < settings.FFT_LENGTHS.size(); i++) {
        auto fft_length = settings.FFT_LENGTHS[i];
        double flop_count = 5 * fft_length * std::log2(fft_length);
        benchmark_info benchmark = benchmark_info(fft_length, flop_count);

        fftwf_complex* original_fftw_input = reinterpret_cast<fftwf_complex*>(inputs[i]);
        fftwf_complex* original_fftw_output = reinterpret_cast<fftwf_complex*>(outputs[i]);
        fftwf_complex* fftw_input = fftwf_alloc_complex(fft_length * sizeof(fftwf_complex));
        fftwf_complex* fftw_output = fftwf_alloc_complex(fft_length * sizeof(fftwf_complex));
        
        std::string wisdom_p = "wisdom/float" + std::to_string(fft_length) + ".fftw";
        fftwf_plan plan;
        plan = fftwf_plan_dft_1d(fft_length, fftw_input, fftw_output, FFTW_FORWARD, FFTW_MEASURE);
        // planning overwrites data, move real data into buffers.
        memcpy(fftw_input, original_fftw_input, fft_length * sizeof(fftwf_complex));
        fftwf_execute(plan); // Warmup

        for (size_t i = 0; i < settings.NUM_ITERATIONS; i++) {
            t1 = std::chrono::high_resolution_clock::now();
            fftwf_execute(plan);
            t2 = std::chrono::high_resolution_clock::now();
            double dur = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
            benchmark.add_timing(dur);
        }

        fftwf_destroy_plan(plan);

        if (settings.DEBUG) {
            std::cout << "FFTW, length: " << fft_length << ", time avg: " 
                << benchmark.t_avg() << "s, time sd: " << benchmark.t_sd() 
                << "s, flop/s avg: " << benchmark.flops_avg() << ", flop/s sd: "
                << benchmark.flops_sd() << "." << std::endl;
        } else {
            if (i % 50 == 0) {
                std::cout << "Completed step " << i << " of benchmarking"
                    << " FFTW with fft_length " << fft_length << "." << std::endl;
            }
        }

        // planning overwrites data, move real data into buffers.
        memcpy(original_fftw_output, fftw_output, fft_length * sizeof(fftwf_complex));
        fftw_info.push_back(benchmark);
    }

    // Cleanup
    fftwf_cleanup_threads();
    fftwf_cleanup();
}

void fftw_double(std::vector<std::complex<double>*> inputs, std::vector<std::complex<double>*> outputs, std::vector<benchmark_info>& fftw_info, settings_s& settings) {
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());

    std::chrono::high_resolution_clock::time_point t1, t2;
    for (size_t i = 0; i < settings.FFT_LENGTHS.size(); i++) {
        auto fft_length = settings.FFT_LENGTHS[i];
        double flop_count = 5 * fft_length * std::log2(fft_length);
        benchmark_info benchmark = benchmark_info(fft_length, flop_count);

        fftw_complex* original_fftw_input = reinterpret_cast<fftw_complex*>(inputs[i]);
        fftw_complex* original_fftw_output = reinterpret_cast<fftw_complex*>(outputs[i]);
        fftw_complex* fftw_input = fftw_alloc_complex(fft_length * sizeof(fftw_complex));
        fftw_complex* fftw_output = fftw_alloc_complex(fft_length * sizeof(fftw_complex));
        
        std::string wisdom_p = "wisdom/double" + std::to_string(fft_length) + ".fftw";
        fftw_plan plan;
        fftw_import_wisdom_from_filename(wisdom_p.c_str());
        if (fft_length == 88*88 || fft_length == 68*68 || fft_length == 136*136) {
            plan = fftw_plan_dft_1d(fft_length, fftw_input, fftw_output, FFTW_FORWARD, FFTW_ESTIMATE);
        }
        else {
            plan = fftw_plan_dft_1d(fft_length, fftw_input, fftw_output, FFTW_FORWARD, FFTW_MEASURE);
        }
        // planning overwrites data, move real data into buffers.
        memcpy(fftw_input, original_fftw_input, fft_length * sizeof(fftw_complex));
        fftw_execute(plan); // Warmup

        for (size_t i = 0; i < settings.NUM_ITERATIONS; i++) {
            t1 = std::chrono::high_resolution_clock::now();
            fftw_execute(plan);
            t2 = std::chrono::high_resolution_clock::now();
            double dur = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
            benchmark.add_timing(dur);
        }

        fftw_export_wisdom_to_filename(wisdom_p.c_str());
        fftw_destroy_plan(plan);

        if (settings.DEBUG) {
            std::cout << "FFTW, length: " << fft_length << ", time avg: " 
                << benchmark.t_avg() << "s, time sd: " << benchmark.t_sd() 
                << "s, flop/s avg: " << benchmark.flops_avg() << ", flop/s sd: "
                << benchmark.flops_sd() << "." << std::endl;
        } else {
            if (i % 50 == 0) {
                std::cout << "Completed step " << i << " of benchmarking"
                    << " FFTW with fft_length " << fft_length << "." << std::endl;
            }
        }

        // planning overwrites data, move real data into buffers.
        memcpy(original_fftw_output, fftw_output, fft_length * sizeof(fftw_complex));
        fftw_info.push_back(benchmark);
    }

    // Cleanup
    fftw_cleanup_threads();
    fftw_cleanup();
}
