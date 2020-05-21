#!/bin/bash

import numpy as np
from numpy.fft import fft
import os

def I_tensorprod_A(A, x, m, n):
    y = np.zeros(m * n, dtype=x.dtype)
    for i in range(m):
        for j in range(n):
            idx_y = i * n + j
            for k in range(n):
                idx_x = i * n + k
                y[idx_y] += A[j, k] * x[idx_x]
    return y

def A_tensorprod_I(A, x, m, n):
    y = np.zeros(m * n, dtype=x.dtype)
    for i in range(n):
        for j in range(m):
            idx_y = i + n * j
            for k in range(m):
                idx_x = i + n * k
                y[idx_y] += A[j, k] * x[idx_x]
    return y

def D_op(D, x, n):
    y = np.zeros(n, dtype=x.dtype)
    for i in range(n):
        y[i] = D[i] * x[i]
    return y

def Permute(x, m, n):
    y = np.zeros(n * m, dtype=x.dtype)
    for i in range(m):
        for j in range(n):
            idx_x = n * i + j
            idx_y = i + m * j
            y[idx_y] = x[idx_x]
    return y

def Permute_tensorprod_I(x, m, n, k, debug):
    y = np.zeros(m * n * k, dtype=x.dtype)
    for i in range(m):
        for j in range(n):
            for l in range(k):
                idx_x = k * (n * i + j) + l
                idx_y = k * (i + m * j) + l
                y[idx_y] = x[idx_x]

                if debug:
                    print("idx_x:", idx_x, "| idx_y:", idx_y)

    return y

def Gorlach_Permute_tensorprod_I(x, i, r, k, debug):
    n = int(round(pow(r, k)))
    y = np.zeros(n, dtype=x.dtype)

    u = k - i
    L_u = int(round(pow(r, u)))

    L_star = int(round(L_u / r))
    I_len = int(round(n / L_u))

    print("u:", u, "| L_u:", L_u, "| L_star:", L_star, "| I_len:", I_len)

    for i in range(L_star):
        for j in range(I_len):
            for l in range(r):
                idx_x = (r*i + l) * I_len + j
                idx_y = (i + l * L_star) * I_len + j

                if debug:
                    print("idx_x:", idx_x, "| idx_y:", idx_y)

                y[idx_y] = x[idx_x]

    return y

def Generate_DFT(r, dtype):
    dft = np.zeros((r,r), dtype=dtype)
    for i in range(r):
        for j in range(r):
            dft[i,j] = np.exp(-2j * np.pi * i * j / r)
    return dft

def Generate_twiddles(n, i, r, k, dtype, debug):
    if debug:
        print()

    # Twiddles
    r_u = r
    u = k - i
    L_u = int(round(pow(r, u)))
    Lu_ru = int(round(pow(r, u - 1))) # L_u divided by r_u
    n_Lu = int(round(pow(r, i)))
    short_twiddle = np.ones(L_u, dtype=dtype)

    if debug:
        print("r_u:", r_u, "| L_u:", L_u, "| Lu_ru:", Lu_ru, "| n_Lu:", n_Lu)

    for i in range(r_u):
        for j in range(Lu_ru):
            idx = i * Lu_ru + j
            # The L_u factor here is not clear from Gorlach's paper.
            short_twiddle[idx] = np.exp(-2j * np.pi * i * j / L_u)

            if debug:
                print("w_", L_u, "**", i*j)

    # Kronecker product
    D = np.zeros(n, dtype=dtype)
    for i in range(L_u):
        for j in range(n_Lu):
            if debug:
                print("i:", i, "| j:", j)
            idx = i * n_Lu + j
            D[idx] = short_twiddle[i]

    if debug:
        print("D: ", np.array_str(D, precision=4, suppress_small=False))
        
    return D

def own_fft(x, n, r, k):
    dtype = x.dtype
    dft_m = Generate_DFT(r, dtype)

    # Main Stockham loop
    # NOTE REVERSE LOOP
    for i in range(k-1, -1, -1):
        r_ki_m1 = int(round(pow(r, k - i - 1)))
        r_k_m1 = int(round(pow(r, k - 1)))
        r_i = int(round(pow(r, i)))

        print("\n\n%%% NEW ITERATION %%%")
        print("i:", i, "| k:", k)

        # Strided read
        #x = Permute_tensorprod_I(x, r, r_ki_m1, r_i, True)
        x = Gorlach_Permute_tensorprod_I(x, i, r, k, True)

        # Twiddle factor multiplication
        D = Generate_twiddles(n, i, r, k, x.dtype, True)
        x = D_op(D, x, n)

        # Vector DFT multiplication
        x = A_tensorprod_I(dft_m, x, r, r_k_m1)

        print("%%%%%%%%%%%%%%%%%%%%%%%")

    return x

if __name__ == "__main__":
    print("==== Program start ====")
    os.environ["OMP_NUM_THREADS"] = "12"

    r = 6
    k = 5
    n = int(round(pow(r, k)))
    print('FFT on real vector of length %d' % n)

    X = np.random.rand(n).astype(np.complex128) + 1j * np.random.rand(n).astype(np.complex128) 

    Y_own = own_fft(X, n, r, k)
    Y_np = fft(X)
    
    print("\n### RESULTS ###")
    print("Y_own: ", np.array_str(Y_own, precision=2, suppress_small=True))
    print("Y_np: ", np.array_str(Y_np, precision=2, suppress_small=True))
        
    eps = 1e-10
    diff = np.linalg.norm(Y_own - Y_np) / n
    if diff < eps:
        print("Difference:", diff)
    else:
        print("\033[91mDifference:", diff, "\033[0m")

    print("==== Program end ====")
    exit(0 if diff <= 1e-3 else 1)