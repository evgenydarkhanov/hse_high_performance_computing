#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

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

__global__ void matmul_gpu(double *A, double *B, double *C, int colsA, int colsB)
{
    int idx_a = blockDim.y * blockIdx.y + threadIdx.y;  // row from A
    int idx_b = blockDim.x * blockIdx.x + threadIdx.x;  // col from B
    int idx_c = colsA * idx_a + idx_b;                  // elem from C
    
    if (idx_a < N && idx_b < N)
    {
        double sum = 0;
        for (int k = 0; k < colsA; k++)
        {
            sum += A[idx_a * colsA + k] * B[idx_b + k * colsB];
        }
        C[idx_c] = sum;
    }
}

int main()
{
    int cols = N;
    int rows = N;
    size_t matrix_size = cols * rows * sizeof(double);
    int device;
    
    cudaGetDevice(&device);
    printf("Device: %d\n", device);

    // creating matrices in unified memory
    double *A, *B, *C;
    cudaMallocManaged( &A, matrix_size );
    cudaMallocManaged( &B, matrix_size );
    cudaMallocManaged( &C, matrix_size );

    // filling matrices with random numbers
    fill_matrix(A, rows, cols); 
    fill_matrix(B, rows, cols);
    
    // early copying
    cudaMemPrefetchAsync( A, matrix_size, device );
    cudaMemPrefetchAsync( B, matrix_size, device );
    cudaMemPrefetchAsync( C, matrix_size, device );

    // block and grid sizes
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocksPerGrid = dim3(GRID_SIZE, GRID_SIZE, 1);

    // time
    cudaEvent_t start_gpu, stop_gpu;
    float gpu_time;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // kernel
    cudaEventRecord(start_gpu, 0);
    matmul_gpu<<< blocksPerGrid, threadsPerBlock >>>(A, B, C, cols, cols);
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop_gpu, 0);
    
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    printf("GPU time (unified memory, kernel): %.3f ms\n", gpu_time);
    
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    /*
    // checking with CPU multiplication
    double *D = (double*)malloc(matrix_size);
    
    cudaEvent_t start_cpu, stop_cpu;
    float cpu_time;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);

    cudaEventRecord(start_cpu, 0);
    matmul_cpu(rows, cols, cols, A, B, D);
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
        delta += fabs(D[i] - C[i]);
    }
    if (delta > 0.00001)
    {
        printf("error %f\n", delta);
    }
    else
    {
        printf("good %f\n", delta);
    }
    free(D);
    */

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
