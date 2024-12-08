#include <iostream>
#include <mpi.h>
#include <vector>

// parameters
const double L = 1.0;         // rod's length
const double k = 1.0;         // coef
const double h = 0.02;        // space step
const double tau = 0.0002;    // time step
const double T = 0.1;         // time

const int TAG = 123;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Status status;

    double start, finish, time;
    int N = static_cast<int>(L / h);  // num of points
    int local_N = N / size;

    std::vector<double> u(N, 0.0);
    std::vector<double> u_new(N, 0.0);
    std::vector<double> u_local(local_N + 2, 0.0);
    std::vector<double> u_local_new(local_N + 2, 0.0);

    if (rank == 0)
    {
        for (int i = 1; i < N - 1; i++)
        {
            u[i] = 1.0;
        }
    }

    MPI_Scatter(u.data() + 1, local_N, MPI_DOUBLE, u_local.data() + 1, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    start = MPI_Wtime();
    for (double t = 0; t < T; t += tau)
    {
        if (rank > 0)
        {
            MPI_Send(&u_local[1], 1, MPI_DOUBLE, rank - 1, TAG, MPI_COMM_WORLD);            // sending left border
            MPI_Recv(&u_local[0], 1, MPI_DOUBLE, rank - 1, TAG, MPI_COMM_WORLD, &status);   // receiving left border
        }
        if (rank < size - 1)
        {
            MPI_Send(&u_local[local_N], 1, MPI_DOUBLE, rank + 1, TAG, MPI_COMM_WORLD);      // sending right border
            MPI_Recv(&u_local[local_N + 1], 1, MPI_DOUBLE, rank + 1, TAG, MPI_COMM_WORLD, &status); // receiving right border
        }

        // computation
        for (int i = 1; i <= local_N; i++)
        {
            u_local_new[i] = u_local[i] + ((k * tau) / (h * h)) * (u_local[i + 1] - 2 * u_local[i] + u_local[i - 1]);
        }

        std::swap(u_local_new, u_local);
    }

    MPI_Gather(u_local.data() + 1, local_N, MPI_DOUBLE, u.data() + 1, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    finish = MPI_Wtime();
    time = finish - start;

    if (rank == 0)
    {
        std::cout << time << " sec" << std::endl;
    }
    /*
    if (rank == 0)
    {
        std::cout << time << " sec" << std::endl;
        std::cout << "rank: " << rank << std::endl;
        for (int i = 0; i < N; i++)
        {
            std::cout << u[i] << " ";
        }
        std::cout << std::endl;
    }
    */
    MPI_Finalize();

    return 0;
}
