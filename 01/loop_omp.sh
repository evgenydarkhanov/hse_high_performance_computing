#!/bin/bash

for ncores in 1 2 4 8 16
do
    echo "ncores: $ncores"
    echo "ncores: $ncores" >> results.txt
    for (( i=1; i<=5; i++ ))
    do
        echo "attempt: $i"
        OMP_NUM_THREADS=$ncores srun --ntasks=1 --cpus-per-task=$ncores ./gemm_omp.out >> results.txt
    done
done
