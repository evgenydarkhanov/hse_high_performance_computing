#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

const int N = 1024;

void fill_matrix(double *matrix, int rows, int cols)
{
    for (int i = 0; i < cols * rows; i++)
    {
        matrix[i] = rand() % 10 + 1;
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

__global__ void matmul_gpu(double *A, double *B, double *C, int size)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;  // row from A
    int col = blockDim.x * blockIdx.x + threadIdx.x;  // col from B
    
    if (row < size && col < size)
    {
        double sum = 0;
        for (int k = 0; k < size; k++)
        {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
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

    // block and grid sizes
    int BLOCK_SIZE = 32;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocksPerGrid( (N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    // creating streams
    int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    // timing
    cudaEvent_t start_gpu, stop_gpu;
    float gpu_time;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu, 0);

    // creating device matrices
    double *d_A[num_streams], *d_B, *d_C[num_streams];
    for (int i = 0; i < num_streams; i++)
    {
        cudaMalloc( (void**)&d_A[i], matrix_size/num_streams );
        cudaMalloc( (void**)&d_C[i], matrix_size/num_streams );

        cudaMemcpyAsync( d_A[i], h_A + i * (N/num_streams)*N, matrix_size/num_streams, cudaMemcpyHostToDevice, streams[i] );
    }

    cudaMalloc( (void**)&d_B, matrix_size/num_streams );
    cudaMemcpyAsync( d_B, h_B, matrix_size, cudaMemcpyHostToDevice );
    
    // kernel
    for (int i = 0; i < num_streams; i++)
    {
        matmul_gpu<<< blocksPerGrid, threadsPerBlock, 0, streams[i] >>>(d_A[i], d_B, d_C[i], N/num_streams);
        cudaMemcpyAsync( h_C + i * (N/num_streams)*N, d_C[i], matrix_size/num_streams, cudaMemcpyDeviceToHost, streams[i] );
        //cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < num_streams; i++)
    {
        cudaStreamSynchronize(streams[i]);
        cudaFree(d_A[i]);
        cudaFree(d_C[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_B);
    cudaEventRecord(stop_gpu, 0);

    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    printf("GPU time (streams, copyHTD, kernel, copyDTH): %.3f ms\n", gpu_time);

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

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
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            delta += fabs(h_D[i * cols + j] - h_C[i * cols + j]);
        }
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
