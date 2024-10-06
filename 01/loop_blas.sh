#!/bin/bash

for (( i=1; i <= 5; i++ ))
do
    echo "$i"
    srun -n 1 ./gemm_blas.out >> results.txt
done
