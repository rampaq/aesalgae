Source codes and data for thesis
=======================

Master's thesis:
Algebraic Cryptanalysis of Small Scale Variants of the AES via Reduction Theory

Author: Tomáš Faikl

FNSPE in Prague, FIT

## Installation:

Create `conda` environment and install SageMath via `conda`.
Optionally install also `galois` for linear algebra over Numpy matrices in finite fields.

Compile [CodeRed](https://github.com/lducas/CodeRed/) reduction library in packaged format in `packages/`. Then, install all supplied python packages via `packages/install-pkgs.sh`.
If needed, adjust the number of cores in `packages/aesalgae/config.py`.

Finally, run 

```sh
# run SR(2,4,2,4), PC=32, r=1, preprocessing: HME -> LLL; Groebner basis using F4 with 24 threads and GPU
sage scripts/aes.py -n 2 -r 4 -c 2 -e 4 --pc-pairs 32 -T24 --preprocessing monomelim,lll --pre-reduce-mult 1

# run SR(3,2,2,4), PC=1, no preprocessing; Groebner basis using F4 with 24 threads and GPU
sage scripts/aes.py -n 3 -r 2 -c 2 -e 4 --pc-pairs 1 -T24
```
