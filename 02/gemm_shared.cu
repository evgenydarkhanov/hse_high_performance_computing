#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

const int N = 1024;
const int BLOCK_SIZE = 32;
const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
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

__global__ void matmul_gpu(double *A, double *B, double *C, int size) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Разделяемая память для подматриц
    __shared__ double shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double shared_B[BLOCK_SIZE][BLOCK_SIZE];
    
    double sum = 0.0;

    // Перемножение блоков
    for (int k = 0; k < (size + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) 
    {
        // Загрузка данных в разделяемую память
        if (row < size && (k * BLOCK_SIZE + threadIdx.x) < size) 
        {
            shared_A[threadIdx.y][threadIdx.x] = A[row * size + (k * BLOCK_SIZE + threadIdx.x)];
        }
        else
        {
            shared_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < size && (k * BLOCK_SIZE + threadIdx.y) < size)
        {
            shared_B[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * size + col];
        }
        else
        {
            shared_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads(); // Синхронизация потоков
        
        // Умножение блоков
        for (int n = 0; n < BLOCK_SIZE; n++)
        {
            sum += shared_A[threadIdx.y][n] * shared_B[n][threadIdx.x];
        }

        __syncthreads(); // Синхронизация потоков
     }

     // Запись результата в глобальную память
     if (row < size && col < size)
     {
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

    // creating device matrices
    double *d_A, *d_B, *d_C;
    cudaMalloc( (void**)&d_A, matrix_size );
    cudaMalloc( (void**)&d_B, matrix_size );
    cudaMalloc( (void**)&d_C, matrix_size );
    
    // block and grid sizes
    dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocksPerGrid = dim3(GRID_SIZE, GRID_SIZE, 1);

    // time
    cudaEvent_t start_gpu, stop_gpu;
    float gpu_time;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // copying the data
    cudaEventRecord(start_gpu, 0);
    cudaMemcpy( d_A, h_A, matrix_size, cudaMemcpyHostToDevice );
    cudaMemcpy( d_B, h_B, matrix_size, cudaMemcpyHostToDevice );
    // cudaMemcpy( d_C, h_C, matrix_size, cudaMemcpyHostToDevice );

    // kernel
    matmul_gpu<<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, d_C, cols);

    cudaMemcpy( h_C, d_C, matrix_size, cudaMemcpyDeviceToHost );
    cudaEventRecord(stop_gpu, 0);

    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    printf("GPU time (shared memory, copyHTD, kernel, copyDTH): %.3f ms\n", gpu_time);

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
