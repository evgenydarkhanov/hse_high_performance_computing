build_global:
	nvcc gemm_global.cu -o gemm_global.out

global: build_global
	srun ./gemm_global.out

build_global_kernel_only:
	nvcc gemm_global_kernel_only.cu -o gemm_global_kernel_only.out

global_kernel_only: build_global_kernel_only
	srun ./gemm_global_kernel_only.out

build_pinned:
	nvcc gemm_pinned.cu -o gemm_pinned.out

pinned: build_pinned
	srun ./gemm_pinned.out

build_unified:
	nvcc gemm_unified.cu -o gemm_unified.out

unified: build_unified
	srun ./gemm_unified.out

build_streams:
	nvcc gemm_streams.cu -o gemm_streams.out

streams: build_streams
	srun ./gemm_streams.out

build_cublas:
	nvcc gemm_cublas.cu -lcublas -o gemm_cublas.out

cublas: build_cublas
	srun ./gemm_cublas.out

clean:
	rm *.out
