#!/usr/bin/env python
from __future__ import print_function

import dace
import mkl
import numpy as np
import os
import scipy.linalg as linalg
import csv

from numpy.fft import fft

#####################################

N = dace.symbol('N')

@dace.program(dace.complex128[N], dace.complex128[N])
def DFT(X, Y):
    # Generate DFT matrix
    dft_mat = dace.define_local([N, N], dtype=dace.complex128)
    @dace.map(_[0:N, 0:N])
    def dft_mat_gen(i, j):
        omega >> dft_mat[i, j]
        
        omega = exp(-dace.complex128(0, 2 * 3.14159265359 * i * j) / dace.complex128(N))
    
    # Matrix multiply input vector with DFT matrix
    tmp = dace.define_local([N, N], dtype=dace.complex128)
    @dace.map(_[0:N, 0:N])
    def dft_tasklet(k, n):
        x << X[n]
        omega << dft_mat[k, n]
        out >> tmp[k, n]
        
        out = x * omega
    
    dace.reduce(lambda a, b: a + b, tmp, Y, axis=1, identity=0)

#####################################

def scipy_dft(X, size):
    dft_mat = linalg.dft(size)
    return np.dot(dft_mat, X)

if __name__ == "__main__":
    print("==== Program start ====")
    
    os.environ["OMP_PROC_BIND"] = "true"
    dace.Config.set('profiling', value=True)
    dace.Config.set('treps', value=100)
    
    size = 128
    
    N.set(size)
    print('\nDFT on real vector of length %d' % (N.get()))
        
    # Initialize arrays: Randomize A and B, zero C
    X = np.random.rand(N.get()).astype(np.complex128)
    Y_dace = np.zeros_like(X, dtype=np.complex128)
    Y_np = fft(X)
        
    DFT(X, Y_dace)

