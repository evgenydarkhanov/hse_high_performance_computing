## Параллельное умножение матриц

### Состав директории

- `gemm_cublas.cu`
- `gemm_global.cu`
- `gemm_global_kernel_only.cu`
- `gemm_pinned.cu`
- `gemm_unified.cu`
- `Makefile` - для запуска соответствующего `.cu`-файла

Перед запуском требуется аллоцировать GPU и загрузить NVIDIA HPC SDK

```bash
salloc -n --gpus=1 -A proj_1593
module load nvidia_sdk/nvhpc/23.5
```

### Global Memory

- `make global` - измеряется время с учётом копирования массивов с хоста на девайс и обратно 
- `make global_kernel_only` - измеряется время только перемножения

### Pinned Memory

- `make pinned`

### Unified Memory

- `make unified`

### cuBLAS

- `make cublas`
