#include "../include/cufft_test.cuh"

#define N_TRANSFORMS 1

void cufft_float(std::vector<std::complex<float>*> inputs, std::vector<std::complex<float>*> outputs, std::vector<benchmark_info>& cufft_info, settings_s& settings) {
    size_t max_fft_length = *std::max_element(settings.FFT_LENGTHS.begin(), settings.FFT_LENGTHS.end());;
    
    cufftComplex *in_data, *out_data;
    std::chrono::high_resolution_clock::time_point t1, t2;

    for (size_t i = 0; i < settings.FFT_LENGTHS.size(); i++) {
        size_t fft_length = settings.FFT_LENGTHS[i];
        double flop_count = 5 * fft_length * std::log2(fft_length);
        benchmark_info benchmark = benchmark_info(fft_length, flop_count);

        cufftComplex* original_cufft_input = reinterpret_cast<cufftComplex*>(inputs[i]);
        cufftComplex* original_cufft_output = reinterpret_cast<cufftComplex*>(outputs[i]);

        cufftHandle plan;
        cufftPlan1d(&plan, fft_length, CUFFT_C2C, N_TRANSFORMS);

        cudaMalloc((void**) &in_data, sizeof(cufftComplex) * fft_length);
        cudaMalloc((void**) &out_data, sizeof(cufftComplex) * fft_length);
        cufftExecC2C(plan, in_data, out_data, CUFFT_FORWARD); // Warmup
        cudaFree(in_data);
        cudaFree(out_data);

        for (size_t iter = 0; iter < settings.NUM_ITERATIONS; iter++) {
            t1 = std::chrono::high_resolution_clock::now();
            cudaMalloc((void**) &in_data, sizeof(cufftComplex) * fft_length);
            cudaMalloc((void**) &out_data, sizeof(cufftComplex) * fft_length);
            cudaMemcpy(in_data, original_cufft_input, fft_length * sizeof(cufftComplex), cudaMemcpyHostToDevice);
            cufftExecC2C(plan, in_data, out_data, CUFFT_FORWARD); 
            cudaMemcpy(original_cufft_output, out_data, fft_length * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
            cudaFree(in_data);
            cudaFree(out_data);
            cudaDeviceSynchronize();
            t2 = std::chrono::high_resolution_clock::now();
            double dur = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
            benchmark.add_timing(dur);
        }

        if (settings.DEBUG) {
            std::cout << "cuFFT, length: " << fft_length << ", time avg: " 
                << benchmark.t_avg() << "s, time sd: " << benchmark.t_sd() 
                << "s, flop/s avg: " << benchmark.flops_avg() << ", flop/s sd: "
                << benchmark.flops_sd() << "." << std::endl;
        } else {
            if (i % 50 == 0) {
                std::cout << "Completed step " << i << " of benchmarking"
                    << " cuFFT with fft_length " << fft_length << "." << std::endl;
            }
        }

        cufft_info.push_back(benchmark);

        cufftDestroy(plan);
    }
}

void cufft_double(std::vector<std::complex<double>*> inputs, std::vector<std::complex<double>*> outputs, std::vector<benchmark_info>& cufft_info, settings_s& settings) {
    size_t max_fft_length = *std::max_element(settings.FFT_LENGTHS.begin(), settings.FFT_LENGTHS.end());;
    
    cufftDoubleComplex *in_data, *out_data;
    std::chrono::high_resolution_clock::time_point t1, t2;

    for (size_t i = 0; i < settings.FFT_LENGTHS.size(); i++) {
        size_t fft_length = settings.FFT_LENGTHS[i];
        double flop_count = 5 * fft_length * std::log2(fft_length);
        benchmark_info benchmark = benchmark_info(fft_length, flop_count);

        cufftDoubleComplex* original_cufft_input = reinterpret_cast<cufftDoubleComplex*>(inputs[i]);
        cufftDoubleComplex* original_cufft_output = reinterpret_cast<cufftDoubleComplex*>(outputs[i]);

        cufftHandle plan;
        cufftPlan1d(&plan, fft_length, CUFFT_Z2Z, N_TRANSFORMS);

        cudaMalloc((void**) &in_data, sizeof(cufftDoubleComplex) * fft_length);
        cudaMalloc((void**) &out_data, sizeof(cufftDoubleComplex) * fft_length);
        cufftExecZ2Z(plan, in_data, out_data, CUFFT_FORWARD); // Warmup
        cudaFree(in_data);
        cudaFree(out_data);

        for (size_t iter = 0; iter < settings.NUM_ITERATIONS; iter++) {
            t1 = std::chrono::high_resolution_clock::now();
            cudaMalloc((void**) &in_data, sizeof(cufftDoubleComplex) * fft_length);
            cudaMalloc((void**) &out_data, sizeof(cufftDoubleComplex) * fft_length);
            cudaMemcpy(in_data, original_cufft_input, fft_length * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
            cufftExecZ2Z(plan, in_data, out_data, CUFFT_FORWARD); 
            cudaMemcpy(original_cufft_output, out_data, fft_length * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
            cudaFree(in_data);
            cudaFree(out_data);
            cudaDeviceSynchronize();
            t2 = std::chrono::high_resolution_clock::now();
            double dur = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
            benchmark.add_timing(dur);
        }

        if (settings.DEBUG) {
            std::cout << "cuFFT, length: " << fft_length << ", time avg: " 
                << benchmark.t_avg() << "s, time sd: " << benchmark.t_sd() 
                << "s, flop/s avg: " << benchmark.flops_avg() << ", flop/s sd: "
                << benchmark.flops_sd() << "." << std::endl;
        } else {
            if (i % 50 == 0) {
                std::cout << "Completed step " << i << " of benchmarking"
                    << " cuFFT with fft_length " << fft_length << "." << std::endl;
            }
        }

        cufft_info.push_back(benchmark);

        cufftDestroy(plan);
    }

}