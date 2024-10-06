## Параллельное умножение матриц

### Состав директории

Различные версии матричного умножения

- `gemm_blas.cpp`
- `gemm_mkl.cpp`
- `gemm_omp.cpp`
- `gemm_serial.cpp`

Makefile и bash-скрипты для запуска соответствующего `.cpp`-файла и записи результатов в `results.txt`

- `loop_blas.sh`
- `loop_mkl.sh`
- `loop_omp.sh`
- `loop_serial.sh`
- `run_omp.sh` (что-то не так)
- `Makefile` 
- `results.txt`

Перед запуском нужно сделать bash-скрипты исполняемыми
```
chmod +x loop_*.sh
./loop_*sh
```

### Serial

`make run_serial`

### Parallel OpenMP

`make run_omp`

### OpenBLAS
```
module load OpenBlas/v0.3.18
make run_blas
```

### MKL
```
module load INTEL/oneAPI_2022
module load mkl/latest
make run_mkl
```
