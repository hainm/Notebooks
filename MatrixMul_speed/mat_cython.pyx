import numpy as np
cimport numpy as np
import time
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def matmul_i_then_j(double[:,:] ma1, double[:,:] ma2, int row1, int col1, int row2, int col2):
    cdef:
        int i, j, k
    cdef double[:,:] outmat = np.zeros((row1, col2))
    
    if col1 != row2:
        raise ValueError("Matrix 1 col# should be equal to row#")

    t0 = time.time()
    #expensive O(N^3) loop (check the yellow color in "outmat[i][j] += ma1[i][k] * ma2[k][j]")
    for i in range(row1):
        for j in range(col2):
            for k in range(row2):
                #this is slow
                outmat[i][j] += ma1[i][k] * ma2[k][j]
    t1 = time.time() - t0
    return np.array(outmat), t1

@cython.boundscheck(False)
@cython.wraparound(False)
def matmul_ij(double[:,:] ma1, double[:,:] ma2, int row1, int col1, int row2, int col2):
    cdef:
        int i, j, k
    cdef double[:,:] outmat = np.zeros((row1, col2))
    
    if col1 != row2:
        raise ValueError("Matrix 1 col# should be equal to row#")

    t0 = time.time()
    #less expensive O(N^3) loop (there is not yellow color for "outmat[i, j] += ma1[i, k] * ma2[k, j]")
    for i in range(row1):
        for j in range(col2):
            for k in range(row2):
                #this is fast
                outmat[i, j] += ma1[i, k] * ma2[k, j]
    t1 = time.time() - t0
    return np.array(outmat), t1

@cython.boundscheck(False)
@cython.wraparound(False)
#mix C and Fortran index
def matmul_ij_mixCF(double[:,::1] ma1, double[::1,:] ma2, int row1, int col1, int row2, int col2):
    cdef:
        int i, j, k
    cdef double[:,:] outmat = np.zeros((row1, col2))
    
    if col1 != row2:
        raise ValueError("Matrix 1 col# should be equal to row#")

    t0 = time.time()
    #less expensive O(N^3) loop (there is not yellow color for "outmat[i, j] += ma1[i, k] * ma2[k, j]")
    for i in range(row1):
        for j in range(col2):
            for k in range(row2):
                #this is fast
                outmat[i, j] += ma1[i, k] * ma2[k, j]
    t1 = time.time() - t0
    return np.array(outmat), t1

