#!/bin/bash

import numpy as np
import dace
from numpy.fft import fft
import os


N = dace.symbol('N')
R = dace.symbol('R')
K = dace.symbol('K')

############################################################################
@dace.program(dace.complex128[N], dace.complex128[N])
def own_fft(x, y):
    dtype = dace.complex128
    
    # Generate radix dft matrix
    dft_mat = dace.define_local([R, R], dtype=dtype)
    @dace.map(_[0:R, 0:R])
    def dft_mat_gen(ii, jj):
        omega >> dft_mat[ii, jj]
        omega = exp(-dace.complex128(0, 2 * 3.14159265359 * ii * jj / R))
        
    tmp = dace.define_local([N], dtype=dtype)
    @dace.map(_[0:N])
    def move_x_to_y(ii):
        x_in << x[ii]
        y_out >> y[ii]
        
        y_out = x_in
        
    # Calculate indices CPU
    c_r_i = dace.define_local([K], dtype=dace.int64)
    c_r_k_i_1 = dace.define_local([K], dtype=dace.int64)
    
    @dace.map(_[0:K])
    def calc_indices_cpu(ii):
        c_r_i_out >> c_r_i[ii]
        c_r_k_i_1_out >> c_r_k_i_1[ii]
        
        c_r_i_out = R ** ii
        c_r_k_i_1_out = R ** (K - ii - 1)
        
    # Calculate indices GPU
    g_r_i = dace.define_local([K], dtype=dace.int64)
    g_r_i_1 = dace.define_local([K], dtype=dace.int64)
    g_r_k_1 = dace.define_local([K], dtype=dace.int64) 
    g_r_k_i_1 = dace.define_local([K], dtype=dace.int64)
    
    @dace.map(_[0:K])
    def calc_indices_cpu(ii):
        g_r_i_out >> g_r_i[ii]
        g_r_i_1_out >> g_r_i_1[ii]
        g_r_k_1_out >> g_r_k_1[ii]
        g_r_k_i_1_out >> g_r_k_i_1[ii]
        
        g_r_i_out = R ** ii
        g_r_i_1_out = R ** (ii + 1)
        g_r_k_1_out = R ** (K - 1)
        g_r_k_i_1_out = R ** (K - ii - 1)
        
    # Main Stockham loop
    for i in range(K):
        # STRIDE PERMUTATION AND TWIDDLE FACTOR MULTIPLICATION
        tmp_perm = dace.define_local([N], dtype=dtype)
        @dace.map(_[0:R, 0:c_r_i[i], 0:c_r_k_i_1[i]])
        def permute(ii, jj, kk):
            r_k_i_1_in << g_r_k_i_1[i]
            r_i_in << g_r_i[i]
            r_i_1_in << g_r_i_1[i]
            y_in << y[r_k_i_1_in * (jj * R + ii) + kk]
            tmp_out >> tmp_perm[r_k_i_1_in * (ii * r_i_in + jj) + kk]
    
            tmp_out = y_in * exp(dace.complex128(0, -2 * 3.14159265359 * ii * jj / r_i_1_in))


        # ---------------------------------------------------------------------
        # Vector DFT multiplication
        x_packed = dace.define_local([R,2],dtype=dtype)
        y_packed = dace.define_local([R,2],dtype=dtype)
        for jj in range(R ** (K-1)):
            @dace.map(_[0:R])
            def pack_matrices(ii):
                g_r_k_1_in << g_r_k_1[i]
                tmp_in << tmp_perm[jj + ii * g_r_k_1_in]
                x_packed_out >> x_packed[ii, 0]
                
                x_packed_out = tmp_in
        
            y_packed[:] = dft_mat @ x_packed
            
            @dace.map(_[0:R])
            def unpack_matrices(ii):
                g_r_k_1_in << g_r_k_1[i]
                y_out >> y[jj + ii * g_r_k_1_in]
                y_packed_in << y_packed[ii, 0]
                
                y_out = y_packed_in

if __name__ == "__main__":
    print("==== Program start ====")
    os.environ["OMP_NUM_THREADS"] = "12"
    dace.Config.set('profiling', value=True)
    dace.Config.set('treps', value=1000)

    r = 2
    k = 4
    n = r ** k
    print('FFT on vector of length %d' % n)

    N.set(n)
    R.set(r)
    K.set(k)

    X = np.random.rand(n).astype(np.complex128) + 1j * np.random.rand(n).astype(np.complex128) 
    Y_own = np.zeros_like(X, dtype=np.complex128)


    own_fft(X, Y_own, N=N, K=K, R=R)
    Y_np = fft(X)
    
    if dace.Config.get_bool('profiling'):
        dace.timethis('FFT', 'numpy_fft', (n**2), fft, X)
    
    print("\n### RESULTS ###")
    print("X:", np.array_str(X, precision=2, suppress_small=True))
    print("Y_own: ", np.array_str(Y_own, precision=10, suppress_small=True))
    print("Y_np: ", np.array_str(Y_np, precision=10, suppress_small=True))
        
    eps = 1e-10
    diff = np.linalg.norm(Y_own - Y_np) / n
    if diff < eps:
        print("Difference:", diff)
    else:
        print("\033[91mDifference:", diff, "\033[0m")

    print("==== Program end ====")
    exit(0 if diff <= 1e-3 else 1)



