## Распараллеливание уравнения теплопроводности

### Состав директории

- `heat_serial.cpp`
- `heat_parallel.cpp`
- `heat_async.cpp`
- `heat_collect.cpp`
- `test.txt` - записаны выводы в консоль файлов `heat_serial.cpp`, `heat_parallel.cpp`
- `temperatures.txt` - записаны изменения температур после каждой итерации
- `Makefile` - для запуска соответствующего `.cpp`-файла
- `results.ipynb`

Перед запуском требуется загрузить openmpi3

```bash
module load openmpi
```

### Serial

- `make serial`

### Parallel

- `make parallel`

### Async

- `make async`

### Collect

- `make collect`

### cleaning

- `make clean` - для удаления `.out`-файлов
