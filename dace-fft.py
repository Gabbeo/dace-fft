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
        
    # Main Stockham loop
    # NOTE REVERSE LOOP
    for i in range(K):
        # STRIDE PERMUTATION
        # Permute_tensorprod_I(x, r, k, i)
        n_p = dace.define_local([1], dtype=dace.int64)
        I_len = dace.define_local([1], dtype=dace.int64)
        @dace.tasklet
        def calc_index():
            n_p_o >> n_p[0]
            I_len_o >> I_len[0]
            n_p_o = R ** i
            I_len_o = R ** (K - i - 1)
    
        tmp_perm = dace.define_local([N], dtype=dtype)
        @dace.map(_[0:R, 0:n_p[0], 0:I_len[0]])
        def permute(ii, jj, kk):
            I_len_in << I_len[0]
            n_p_in << n_p[0]
            y_in << y[I_len_in * (jj * R + ii) + kk]
            tmp_out >> tmp_perm[I_len_in * (ii * n_p_in + jj) + kk]
    
            tmp_out = y_in
            
        # ---------------------------------------------------------------------
        # TWIDDLE FACTOR MULTIPLICATION
        # D = Generate_twiddles(n, i, r, k, x.dtype, True)
        D = dace.define_local([N], dtype=dace.complex64)
        L_q = dace.define_local([1], dtype=dace.int64)
        Lq_rq = dace.define_local([1], dtype=dace.int64)
        n_Lq = dace.define_local([1], dtype=dace.int64)
        
        @dace.tasklet
        def calc_indices():
            L_q_o >> L_q[0]
            Lq_rq_o >> Lq_rq[0]
            n_Lq_o >> n_Lq[0]
            
            L_q_o = R ** (i + 1)
            Lq_rq_o = R ** i
            n_Lq_o = R ** (K - i - 1)
            
            
        @dace.map(_[0:R, 0:Lq_rq[0], 0:n_Lq[0]])
        def generate_twiddles(ii, jj, kk):
            L_q_in << L_q[0]
            Lq_rq_in << Lq_rq[0]
            n_Lq_in << n_Lq[0]
            twiddle_o >> D[n_Lq_in * (ii * Lq_rq_in + jj) + kk]
            twiddle_o = exp(dace.complex64(0, -2 * 3.14159265359 * ii * jj / L_q_in))
            

        # x = D_op(D, x, n)
        tmp_twid = dace.define_local([N], dtype=dtype)
        @dace.map(_[0:N])
        def twiddle_multiplication(i):
            tmp_in << tmp_perm[i]
            D_in << D[i]
            tmp_out >> tmp_twid[i]
            
            tmp_out = tmp_in * D_in

        # ---------------------------------------------------------------------
        # Vector DFT multiplication
        # x = A_tensorprod_I(dft_m, x, r, r_k_m1)
        tmp_y = dace.define_local([N, N], dtype=dace.complex64)
        r_k_m1 = dace.define_local([1], dtype=dace.int64)
        @dace.tasklet
        def calc_index():
            r_k_m1_out >> r_k_m1[0]
            
            r_k_m1_out = R ** (K - 1)
        
        @dace.map(_[0:r_k_m1[0], 0:R, 0:R])
        def tensormult(ii, jj, kk):
            r_k_m1_in << r_k_m1[0]
            dft_in << dft_mat[jj, kk]
            tmp_in << tmp_twid[ii + r_k_m1_in * kk]
            tmp_y_out >> tmp_y[ii + r_k_m1_in * jj, ii + r_k_m1_in * kk]

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
    os.environ["OMP_NUM_THREADS"] = "1"
    dace.Config.set('profiling', value=True)
    dace.Config.set('treps', value=1)

    r = 2
    k = 2
    n = r ** k
    print('FFT on real vector of length %d' % n)

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