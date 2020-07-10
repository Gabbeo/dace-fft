import numpy as np
import dace
from numpy.fft import fft
import os

N, R, K = (dace.symbol(name) for name in ['N','R','K'])

@dace.program(dace.complex128[N], dace.complex128[N])
def stockhamFFT(x, y):
    dtype = dace.complex128
    
    # Generate radix dft matrix
    dft_mat = dace.define_local([R, R], dtype=dace.complex128)
    @dace.map(_[0:R, 0:R])
    def dft_mat_gen(ii, jj):
        omega >> dft_mat[ii, jj]
        omega = exp(-dace.complex128(0, 2 * 3.14159265359 * ii * jj / R))
        
    tmp = dace.define_local([N], dtype=dace.complex128)
    @dace.map(_[0:N])
    def move_x_to_y(ii):
        x_in << x[ii]
        y_out >> y[ii]
        
        y_out = x_in
        
    # Calculate indices
    r_i = dace.define_local([K], dtype=dace.int64)
    r_i_1 = dace.define_local([K], dtype=dace.int64)
    r_k_1 = dace.define_local([K], dtype=dace.int64)
    r_k_i_1 = dace.define_local([K], dtype=dace.int64)
    
    @dace.map(_[0:K])
    def calc_indices(ii):
        r_i_out >> r_i[ii]
        r_i_1_out >> r_i_1[ii]
        r_k_1_out >> r_k_1[ii]
        r_k_i_1_out >> r_k_i_1[ii]
        
        r_i_out = R ** ii
        r_i_1_out = R ** (ii + 1)
        r_k_1_out = R ** (K - 1)
        r_k_i_1_out = R ** (K - ii - 1)

    # Main Stockham loop
    for i in range(K):
        # STRIDE PERMUTATION
        tmp_perm = dace.define_local([N], dtype=dace.complex128)
        @dace.map(_[0:R, 0:r_i[i], 0:r_k_i_1[i]])
        def permute(ii, jj, kk):
            r_k_i_1_in << r_k_i_1[i]
            r_i_in << r_i[i]
            y_in << y[r_k_i_1_in * (jj * R + ii) + kk]
            tmp_out >> tmp_perm[r_k_i_1_in * (ii * r_i_in + jj) + kk]
    
            tmp_out = y_in
            
        # ---------------------------------------------------------------------
        # TWIDDLE FACTOR MULTIPLICATION
        D = dace.define_local([N], dtype=dace.complex128)
        @dace.map(_[0:R, 0:r_i[i], 0:r_k_i_1[i]])
        def generate_twiddles(ii, jj, kk):
            r_i_1_in << r_i_1[i]
            r_i_in << r_i[i]
            r_k_i_1_in << r_k_i_1[i]
            twiddle_o >> D[r_k_i_1_in * (ii * r_i_in + jj) + kk]
            twiddle_o = exp(dace.complex64(0, -2 * 3.14159265359 * ii * jj / r_i_1_in))
            
        tmp_twid = dace.define_local([N], dtype=dace.complex128)
        @dace.map(_[0:N])
        def twiddle_multiplication(i):
            tmp_in << tmp_perm[i]
            D_in << D[i]
            tmp_out >> tmp_twid[i]
            
            tmp_out = tmp_in * D_in

        # ---------------------------------------------------------------------
        # Vector DFT multiplication
        tmp_y = dace.define_local([N, N], dtype=dace.complex128)
        @dace.map(_[0:r_k_1[i], 0:R, 0:R])
        def tensormult(ii, jj, kk):
            r_k_1_in << r_k_1[i]
            dft_in << dft_mat[jj, kk]
            tmp_in << tmp_twid[ii + r_k_1_in * kk]
            tmp_y_out >> tmp_y[ii + r_k_1_in * jj, ii + r_k_1_in * kk]

            tmp_y_out = dft_in * tmp_in

        dace.reduce(lambda a, b: a + b, tmp_y, y, axis=1, identity=0)

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

    stockhamFFT(X, Y_own, N=N, K=K, R=R)
    Y_np = fft(X)
    
    if dace.Config.get_bool('profiling'):
        dace.timethis('FFT', 'numpy_fft', (n**2), fft, X)
    
    print("\n### RESULTS ###")
    print("X:", np.array_str(X, precision=2, suppress_small=True))
    print("Y_own: ", np.array_str(Y_own, precision=10, suppress_small=True))
    print("Y_np: ", np.array_str(Y_np, precision=10, suppress_small=True))
        
    diff = np.linalg.norm(Y_own - Y_np) / n
    print("Difference:", diff)

    print("==== Program end ====")
    exit(0 if diff <= 1e-3 else 1)
