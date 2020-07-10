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

    with dace.tasklet(language=dace.Language.CPP, code_global='#include <mkl.h>'):
        x << X; omega << dft_mat; y >> Y
        '''
        dace::complex128 alpha(1,0), beta(0,0);
        cblas_zgemv(CblasRowMajor, CblasNoTrans, N, N, &alpha, omega, N, x, 1, &beta, y, 1);
        '''

#####################################

def scipy_dft(X, size):
    dft_mat = linalg.dft(size)
    return np.dot(dft_mat, X)

if __name__ == "__main__":
    print("==== Program start ====")
    
    #os.environ["OMP_PROC_BIND"] = "true"
    os.environ["OMP_NUM_THREADS"] = "24"
    dace.Config.set('profiling', value=True)
    dace.Config.set('treps', value=100)
    
    size = 4096
    
    N.set(size)
    print('\nDFT on real vector of length %d' % (N.get()))
        
    # Initialize arrays: Randomize A and B, zero C
    X = np.random.rand(N.get()).astype(np.complex128)
    Y_dace = np.zeros_like(X, dtype=np.complex128)
    Y_np = fft(X)
        
    DFT(X, Y_dace)
    
    diff = np.linalg.norm(Y_dace - Y_np) / size
    print(diff)

