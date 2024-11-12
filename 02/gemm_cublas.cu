#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int N = 1024;

void fill_matrix(double *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = (i + 1) * (j + 1);
        }
    }
}

void matmul_cpu(int rowsA, int colsA, int colsB, double *A, double *B, double *C)
{
    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < colsB; j++)
        {
            C[i * colsB + j] = 0;
            for (int k = 0; k < colsA; k++)
            {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];

            }
        }
    }
}

int main()
{
    int cols = N;
    int rows = N;
    size_t matrix_size = cols * rows * sizeof(double);

    // creating host matrices
    double *h_A = (double*)malloc(matrix_size);
    double *h_B = (double*)malloc(matrix_size);
    double *h_C = (double*)malloc(matrix_size);

    // filling matrices with random numbers
    fill_matrix(h_A, rows, cols); 
    fill_matrix(h_B, rows, cols);

    // creating device matrices
    double *d_A, *d_B, *d_C;
    cudaMalloc( (void**)&d_A, matrix_size );
    cudaMalloc( (void**)&d_B, matrix_size );
    cudaMalloc( (void**)&d_C, matrix_size );
    
    // time
    cudaEvent_t start_gpu, stop_gpu;
    float gpu_time;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    
    // creating cublas context
    cublasHandle_t handle;
    cublasCreate(&handle);
   
    // copying the data
    cudaEventRecord(start_gpu, 0);
    cublasSetMatrix(N, N, sizeof(double), h_A, N, d_A, N);
    cublasSetMatrix(N, N, sizeof(double), h_B, N, d_B, N);
    
    // multiplication
    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    cublasGetMatrix(N, N, sizeof(double), d_C, N, h_C, N);
   
    cublasDestroy(handle);
    cudaEventRecord(stop_gpu, 0);

    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    printf("GPU time (cuBLAS, setMatrix, kernel, getMatrix): %.3f ms\n", gpu_time);

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    /*
    // checking with CPU multiplication
    double *h_D = (double*)malloc(matrix_size);
    
    cudaEvent_t start_cpu, stop_cpu;
    float cpu_time;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);

    cudaEventRecord(start_cpu, 0);
    matmul_cpu(rows, cols, cols, h_A, h_B, h_D);
    cudaEventRecord(stop_cpu, 0);

    cudaEventSynchronize(stop_cpu);
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);

    printf("CPU time: %.3f ms\n", cpu_time);
    
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);

    printf("checking\n");
    double delta = 0;
    for (int i = 0; i < cols * rows; i++)
    {
        delta += fabs(h_D[i] - h_C[i]);
    }
    if (delta > 0.00001)
    {
        printf("error %f\n", delta);
    }
    else
    {
        printf("good %f\n", delta);
    }

    free(h_D);
    */
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
