#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

const int N = 1024;

void fill_matrix(double *matrix, int rows, int cols)
{
    for (int i = 0; i < cols * rows; i++)
    {
        matrix[i] = rand() % 100;
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

void matmul_openmp(int rowsA, int colsA, int colsB, double *A, double *B, double *C)
{
    #pragma omp target teams distribute parallel for map(to: A[0:colsA*rowsA], B[0:colsA*colsB]) map(from: C[0:rowsA*colsB])
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
    double *h_C_omp = (double*)malloc(matrix_size);

    // filling matrices with random numbers
    fill_matrix(h_A, rows, cols); 
    fill_matrix(h_B, rows, cols);

    // multiplication
    double omp_time;
    double start_omp = omp_get_wtime();
    matmul_openmp(rows, cols, cols, h_A, h_B, h_C_omp);
    omp_time = omp_get_wtime() - start_omp;

    printf("OpenMP time: %.3f ms\n", omp_time * 1000);

    /*
    // checking with CPU multiplication
    double *h_C_cpu = (double*)malloc(matrix_size);
    
    double cpu_time;
    double start_cpu = omp_get_wtime();
    matmul_cpu(rows, cols, cols, h_A, h_B, h_C_cpu);
    cpu_time = omp_get_wtime() - start_cpu;

    printf("CPU time: %.3f ms\n", cpu_time * 1000);

    printf("checking\n");
    double delta = 0;
    for (int i = 0; i < cols * rows; i++)
    {
        delta += fabs(h_C_cpu[i] - h_C_omp[i]);
    }
    if (delta > 0.00001)
    {
        printf("error %f\n", delta);
    }
    else
    {
        printf("good %f\n", delta);
    }
    free(h_C_cpu);
    */
    free(h_A);
    free(h_B);
    free(h_C_omp);

    return 0;
}
