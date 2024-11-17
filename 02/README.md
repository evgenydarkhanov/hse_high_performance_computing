## Параллельное умножение матриц

### Состав директории

- `gemm_cublas.cu`
- `gemm_global.cu`
- `gemm_global_kernel_only.cu`
- `gemm_omp.c`
- `gemm_pinned.cu`
- `gemm_shared.cu`
- `gemm_streams.cu`
- `gemm_unified.cu`
- `Makefile` - для запуска соответствующего `.cu/.c`-файла

Перед запуском требуется аллоцировать GPU и загрузить NVIDIA HPC SDK

```bash
salloc -n --gpus=1
module load nvidia_sdk/nvhpc/23.5
```

### Global Memory

- `make global` - измеряется время с учётом копирования массивов с хоста на девайс и обратно 
- `make global_kernel_only` - измеряется время только перемножения

### Pinned Memory

- `make pinned`

### Unified Memory

- `make unified`

### Streams

- `make streams`

### Shared Memory

- `make shared`

### cuBLAS

- `make cublas`

### OpenMP

- `make openmp`

### cleaning

- `make clean` - для удаления `.out`-файлов
