#!/bin/bash

import numpy as np
import dace
from numpy.fft import fft
import os


N = dace.symbol('N')
R = dace.symbol('R')
K = dace.symbol('K')

############################################################################
@dace.program(dace.complex64[N], dace.complex64[N])
def own_fft(x, y):
    dtype = dace.complex64
    
    # Generate radix dft matrix
    dft_mat = dace.define_local([R, R], dtype=dtype)
    @dace.map(_[0:R, 0:R])
    def dft_mat_gen(i, j):
        omega >> dft_mat[i, j]
        omega = exp(-dace.complex64(0, 2 * 3.14159265359 * i * j / R))
        
    tmp = dace.define_local([N], dtype=dtype)
    @dace.map(_[0:N])
    def move_x_to_y(i):
        x_in << x[i]
        y_out >> y[i]
        
        y_out = x_in
        
    # Calculate indices        
    r_i = dace.define_local([K], dtype=dace.int64)
    r_i_1 = dace.define_local([K], dtype=dace.int64)
    r_k_1 = dace.define_local([K], dtype=dace.int64) 
    r_k_i_1 = dace.define_local([K], dtype=dace.int64)

    
    @dace.map(_[0:K])
    def calc_index(i):
        # Permutations
        r_i_o >> r_i[i]
        r_i_1_o >> r_i_1[i]
        r_k_1_o >> r_k_1[i]
        r_k_i_1_o >> r_k_i_1[i]
        
        r_i_o = R ** i
        r_i_1_o = R ** (i + 1)
        r_k_i_1_o = R ** (K - i - 1)
        r_k_1_o = R ** (K - 1)        

    # Main Stockham loop
    for i in range(K):
        # STRIDE PERMUTATION
        tmp_perm = dace.define_local([N], dtype=dtype)
        @dace.map(_[0:R, 0:r_i[i], 0:r_k_i_1[i]])
        def permute(ii, jj, kk):
            r_k_i_1_in << r_k_i_1[i]
            r_i_in << r_i[i]
            y_in << y[r_k_i_1_in * (jj * R + ii) + kk]
            tmp_out >> tmp_perm[r_k_i_1_in * (ii * r_i_in + jj) + kk]
    
            tmp_out = y_in
            
        # ---------------------------------------------------------------------
        # TWIDDLE FACTOR MULTIPLICATION
        D = dace.define_local([N], dtype=dace.complex64)
        @dace.map(_[0:R, 0:r_i[i], 0:r_k_i_1[i]])
        def generate_twiddles(ii, jj, kk):
            r_i_1_in << r_i_1[i]
            r_i_in << r_i[i]
            r_k_i_1_in << r_k_i_1[i]
            twiddle_o >> D[r_k_i_1_in * (ii * r_i_in + jj) + kk]
            twiddle_o = exp(dace.complex64(0, -2 * 3.14159265359 * ii * jj / r_i_1_in))
            
        tmp_twid = dace.define_local([N], dtype=dtype)
        @dace.map(_[0:N])
        def twiddle_multiplication(ii):
            tmp_in << tmp_perm[ii]
            D_in << D[ii]
            tmp_out >> tmp_twid[ii]
            
            tmp_out = tmp_in * D_in

        # ---------------------------------------------------------------------
        # Vector DFT multiplication
        tmp_y = dace.define_local([N, N], dtype=dace.complex64)
        @dace.map(_[0:r_k_1[i], 0:R, 0:R])
        def tensormult(ii, jj, kk):
            r_k_1_in << r_k_1[i]
            dft_in << dft_mat[jj, kk]
            tmp_in << tmp_twid[ii + r_k_1_in * kk]
            tmp_y_out >> tmp_y[ii + r_k_1_in * jj, ii + r_k_1_in * kk]

            tmp_y_out = dft_in * tmp_in
            
        tmp_red = dace.define_local([N], dtype=dtype)
        dace.reduce(lambda a, b: a + b, tmp_y, tmp_red, axis=1, identity=0)
        
        @dace.map(_[0:N])
        def move_to_y(i):
            tmp_in << tmp_red[i]
            y_out >> y[i]
            
            y_out = tmp_in
        
if __name__ == "__main__":
    print("==== Program start ====")
    os.environ["OMP_NUM_THREADS"] = "12"
    dace.Config.set('profiling', value=True)
    dace.Config.set('treps', value=10)

    r = 2
    k = 8
    n = r ** k
    print('FFT on vector of length %d' % n)

    N.set(n)
    R.set(r)
    K.set(k)

    X = np.random.rand(n).astype(np.complex64) + 1j * np.random.rand(n).astype(np.complex64) 
    Y_own = np.zeros_like(X, dtype=np.complex64)


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
