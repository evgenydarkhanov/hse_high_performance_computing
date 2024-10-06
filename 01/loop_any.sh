#!/bin/bash

OMP_NUM_THREADS=48 srun --ntasks=1 --cpus-per-task=48 ./gemm_any.out >> results.txt
echo "done any"
