Compare matrix multiplication speed between Cython, Numba and Numpy
------------------------------------------------------------------

Cython: Two matrix indexing styles:
* C[i][j]
* C[i, j]

Numba: autojit

Numpy: mat * mat

Compile
------
./install.sh

Run
---
python ./test_speed.py 1500

(1500 = matrix size)
