## Параллельное умножение матриц

Различные версии матричного умножения

- `gemm_any.cpp`
- `gemm_blas.cpp`
- `gemm_mkl.cpp`
- `gemm_omp.cpp`
- `gemm_serial.cpp`

Makefile и bash-скрипты для запуска соответствующего `.cpp`-файла и записи результатов в `results.txt`. Анализ в `results.ipynb`

- `loop_any.sh`
- `loop_blas.sh`
- `loop_mkl.sh`
- `loop_omp.sh`
- `loop_serial.sh`
- `run_omp.sh` (что-то не так)
- `Makefile` 
- `results.txt`
- `results.ipynb`

Перед запуском нужно сделать bash-скрипты исполняемыми
```bash
chmod +x loop_*.sh
./loop_*sh
```

### Serial

- `make run_serial`

### Parallel OpenMP

- `make run_omp`

### OpenBLAS
```bash
module load OpenBlas/v0.3.18
make run_blas
```

### MKL
```bash
module load INTEL/oneAPI_2022
module load mkl/latest
make run_mkl
```

### Any

умножение с произвольным количеством потоков, которое передаётся через `OMP_NUM_THREADS` в `loop_any.sh`
- `make run_any`


`make clean` - для удаления `*.out`-файлов
