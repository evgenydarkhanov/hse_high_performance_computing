N = 4

serial: build_serial
	srun -A proj_1593 heat_serial.out

build_serial:
	mpicxx heat_serial.cpp -o heat_serial.out

parallel: build_parallel
	srun -N 1 --ntasks-per-node=$(N) -A proj_1593 heat_parallel.out

build_parallel:
	mpicxx heat_parallel.cpp -o heat_parallel.out

async: build_async
	srun -N 1 --ntasks-per-node=$(N) -A proj_1593 heat_async.out

build_async:
	mpicxx heat_async.cpp -o heat_async.out

collect: build_collect
	srun -N 1 --ntasks-per-node=$(N) -A proj_1593 heat_collect.out

build_collect:
	mpicxx heat_collect.cpp -o heat_collect.out

clean:
	rm -r *.out
