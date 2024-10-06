run_serial: build_serial
	echo "----- Running serial -----"
	./loop_serial.sh

build_serial:
	echo "----- Building serial -----"
	g++ gemm_serial.cpp -o gemm_serial.out

run_omp: build_omp
	echo "----- Running OpenMP -----"
	./loop_omp.sh

build_omp:
	echo "----- Building OpenMP ------"
	g++ -fopenmp gemm_omp.cpp -o gemm_omp.out

run_blas: build_blas
	echo "----- Running BLAS -----"
	./loop_blas.sh

build_blas:
	echo "----- Building BLAS -----"
	g++ -lopenblas -fopenmp gemm_blas.cpp -o gemm_blas.out

run_mkl: build_mkl
	echo "----- Running MKL ------"
	./loop_mkl.sh

build_mkl:
	echo "----- Building MKL -----"
	g++ gemm_mkl.cpp -fopenmp -lmkl_gf_lp64 -lmkl_core -lmkl_gnu_thread -o gemm_mkl.out

run_any: build_any
	echo "----- Running any -----"
	./loop_any.sh

build_any:
	echo "----- Building any -----"
	g++ -fopenmp gemm_any.cpp -o gemm_any.out

clean:
	echo "----- Cleaning -----"
	rm -rf *.out
