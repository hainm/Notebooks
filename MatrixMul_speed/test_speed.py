import cython
import numba
from numba import autojit
import numpy as np
import platform
from mat_cython import *

print "INFO"
print "--"*20
print "* numpy version: %s" % np.version.version
print "* cython version: %s" % cython.__version__
print "* numba version: %s" % numba.__version__
print '* system   :', platform.system()
print '* release  :', platform.release()
print '* version  :', platform.version()
print '* machine  :', platform.machine()
print '* processor:', platform.processor()
print "--"*20
print

@autojit
def matmul_numba(ma1, ma2, row1, col1, row2, col2):
    outmat = np.zeros((row1, col2))
    
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
    return outmat, t1

if __name__ == '__main__':
    import sys
    N = int(sys.argv[1]) 
    X = np.random.rand(N*N).reshape(N,N)
    Y = np.random.rand(N*N).reshape(N,N)
    n, m = X.shape
    q, w = Y.shape
    
    print "RESULT"
    print "--"*20
    print "matrix size: %d x %d" % (N, N)
    output_matmul_i_then_j, t_i_then_j = matmul_i_then_j(X, Y, n, m, q, w)
    print "        * C[i][j] style:         %f (s)" % t_i_then_j
    
    output_matmul_ij, t_ij = matmul_ij(X, Y, n, m, q, w)
    print "        * C[i, j] style:         %f (s)" % t_ij
    
    output_matmul_ij_mixCF, t_ij_mixCF = matmul_ij(X, Y, n, m, q, w)
    print "        * C[i, j] style, mix CF: %f (s)" % t_ij_mixCF
    
    output_matmul_numba, t_numba = matmul_ij(X, Y, n, m, q, w)
    print "        * numba:                 %f (s)" % t_numba
    
    #compare to numpy
    Xmat = np.asmatrix(X, dtype=np.float64)
    Ymat = np.asmatrix(Y, dtype=np.float64)
    t0 = time.time()
    outputnp =  Xmat * Ymat
    t_numpy = time.time() - t0
    print "        * numpy time:            %f (s)" % t_numpy
    
    #make sure to reprofuce numpy result
    #print np.any(np.abs(np.array(outputnp) - output_matmul_i_then_j) < 1e-6)
    #print np.any(np.abs(np.array(outputnp) - output_matmul_ij) < 1e-6)
    #print np.any(np.abs(np.array(outputnp) - output_matmul_ij_mixCF) < 1e-6)
    #print np.any(np.abs(np.array(outputnp) - output_matmul_numba) < 1e-6)
    
    #relative speed
    d = {'C[i][j] style': t_i_then_j/t_numpy,
         'C[i, j] style': t_ij/t_numpy,
         'C[i, j] style, mixCF': t_ij_mixCF/t_numpy,
         'numba': t_numba/t_numpy,
         'numpy':         t_numpy/t_numpy}
    
    index = np.arange(len(d.keys()))
    speed = [1./value for key, value in d.iteritems()]
    
    print "Relative speed, matrix size = %d x %d" % (N, N)
    for key, value in d.iteritems():
        print "        * %20s: %3.3f" % (key, 1./value)
