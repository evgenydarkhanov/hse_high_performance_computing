#!/bin/bash
#SBATCH --job-name=gemm_omp
#SBATCH --output=gemm_omp.out
#SBATCH --error=gemm_omp.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


g++ -fopenmp gemm_omp.cpp -o gemm_omp.out

for threads in 1 2 4 8 16
do
    echo "$threads"
    OMP_NUM_THREADS=$threads ./gemm_omp.out >> results.txt
done
