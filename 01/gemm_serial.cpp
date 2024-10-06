#include <iostream>
#include <ctime>
#include <cstdlib>

const int N = 1500;

void fill_matrix(double* matrix){
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = rand() % 100;
    }
}

void matmul(int M, int N, int K, double* A, double* B, double* C) {
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

    clock_t start_time = clock();
    matmul(N, N, N, A, B, C);
    clock_t end_time = clock();

    double total_time = double(end_time - start_time) * (1000000.0 / CLOCKS_PER_SEC);
    std::cout << "total time: " << total_time << " microseconds" << std::endl;

    /*
    // for comparison
    double* D = new double [N * N];

    matmul(N, N, N, A, B, D);

    double delta = 0;
    for (int i = 0; i < N * N; ++i) {
        delta += abs(C[i] - D[i]);
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
