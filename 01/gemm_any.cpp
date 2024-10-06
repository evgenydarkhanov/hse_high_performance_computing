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
    int num_threads = omp_get_max_threads();

    srand(time(nullptr));
    double* A = new double [N * N];
    double* B = new double [N * N];
    double* C = new double [N * N];

    fill_matrix(A);
    fill_matrix(B);

    
    double times;
    
    // можно установить произвольное количество потоков через OMP_NUM_THREADS при запуске

    for (int ncores = 1; ncores <= num_threads; ncores++) {
        omp_set_num_threads(ncores);
        times = omp_get_wtime();
        gemm_omp(N, N, N, A, B, C);
        times = omp_get_wtime() - times;

        std::cout << "ncores: " << ncores << ", total time: " << times * 1e6  << " microseconds" << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
