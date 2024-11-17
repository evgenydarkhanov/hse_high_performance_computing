NVCC=nvcc

build_global:
	$(NVCC) gemm_global.cu -o gemm_global.out

global: build_global
	srun ./gemm_global.out

build_global_kernel_only:
	$(NVCC) gemm_global_kernel_only.cu -o gemm_global_kernel_only.out

global_kernel_only: build_global_kernel_only
	srun ./gemm_global_kernel_only.out

build_pinned:
	$(NVCC) gemm_pinned.cu -o gemm_pinned.out

pinned: build_pinned
	srun ./gemm_pinned.out

build_unified:
	$(NVCC) gemm_unified.cu -o gemm_unified.out

unified: build_unified
	srun ./gemm_unified.out

build_streams:
	$(NVCC) gemm_streams.cu -o gemm_streams.out

streams: build_streams
	srun ./gemm_streams.out

build_shared:
	$(NVCC) gemm_shared.cu -o gemm_shared.out

shared: build_shared
	srun ./gemm_shared.out

build_cublas:
	$(NVCC) gemm_cublas.cu -lcublas -o gemm_cublas.out

cublas: build_cublas
	srun ./gemm_cublas.out

build_openmp:
	nvc -mp=gpu gemm_omp.c -o gemm_omp.out

openmp: build_openmp
	srun ./gemm_omp.out

clean:
	rm *.out
