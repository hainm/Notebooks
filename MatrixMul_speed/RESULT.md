python ./test_speed.py 1500
> INFO
> ----------------------------------------
> * numpy version: 1.9.0
> * cython version: 0.21
> * numba version: 0.15.1
> * system   : Linux
> * release  : 3.13.0-24-generic
> * version  : #46-Ubuntu SMP Thu Apr 10 19:11:08 UTC 2014
> * machine  : x86_64
> * processor: x86_64
> ----------------------------------------
> 
> RESULT
> ----------------------------------------
> matrix size: 1500 x 1500
>         * C[i][j] style:         176.108312 (s)
>         * C[i, j] style:         28.757729 (s)
>         * C[i, j] style, mix CF: 28.207343 (s)
>         * numba:                 29.545591 (s)
>         * numpy time:            0.933783 (s)
> Relative speed, matrix size = 1500 x 1500
>         *                numpy: 1.000
>         *                numba: 0.032
>         * C[i, j] style, mixCF: 0.033
>         *        C[i, j] style: 0.032
>         *        C[i][j] style: 0.005

