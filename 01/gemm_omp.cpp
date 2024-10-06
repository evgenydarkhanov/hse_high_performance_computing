#include <iostream>
#include <ctime>
#include <cstdlib>
#include <omp.h>

const int N = 1500;

void fill_matrix(double* matrix){
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = rand() % 100;
    }
}

void gemm_serial(int M, int N, int K, double* A, double* B, double* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0;
            for (int k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];

            }
        }
    }
}

void gemm_omp(int M, int N, int K, double* A, double* B, double* C) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0;
            for (int k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];

            }
        }
    }
}


int main()
{
    srand(time(nullptr));
    double* A = new double [N * N];
    double* B = new double [N * N];
    double* C = new double [N * N];

    fill_matrix(A);
    fill_matrix(B);

    double start_time = omp_get_wtime();
    gemm_omp(N, N, N, A, B, C);
    double end_time = omp_get_wtime();

    double total_time = (end_time - start_time) * 1e6;
    std::cout << "total time: " << total_time  << " microseconds" << std::endl;
    
    /*
    // INCREMETING NCORES
    for (int ncores = 1; ncores <= num_threads; ncores++) {
        for (int i = 1; i <= 5; i++) {
            omp_set_num_threads(ncores);

            std::cout << "threads: " << ncores << ", attempt: " << i << std::endl;

            clock_t start_time = clock();
            gemm_omp(N, N, N, A, B, C);
            clock_t end_time = clock();

            double total_time = double(end_time - start_time) * (1000000.0 / CLOCKS_PER_SEC);
            std::cout << "total time: " << total_time  << " microseconds" << std::endl;

        }
    }
    */


    /*
    // COMPARISON
    double* D = new double [N * N];

    gemm_serial(N, N, N, A, B, D);

    double delta = 0;
    for (int i = 0; i < N * N; ++i) {
        delta += std::abs(C[i] - D[i]);
    }

    if (delta > 0.0001) {
        std::cout << "error " << std::endl;
    }
    else {
        std::cout << "good" << std::endl;
    }

    delete[] D;
    */

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
