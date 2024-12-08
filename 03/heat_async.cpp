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
    MPI_Request send_request, recv_request;

    double start, finish, time;
    int N = static_cast<int>(L / h);  // num of points
    int local_N = N / size;

    std::vector<double> u(N, 0.0);
    std::vector<double> u_local(local_N + 2, 0.0);
    std::vector<double> u_local_new(local_N + 2, 0.0);

    if (rank == 0)
    {
        for (int i = 1; i < N - 1; i++)
        {
            u[i] = 1.0;
        }
    }

    // sending
    for (int i = 0; i < size; i++)
    {
        /*
        if(i == rank)
        {
            for (int j = 1; j <= local_N; j++)
            {
                u_local[j] = u[j + i * local_N];
            }
        } 
        */
        if (rank == 0)
        {
            MPI_Send(u.data() + 1 + i * local_N, local_N, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
        }
        else if (rank == i)
        {
            MPI_Recv(u_local.data() + 1, local_N, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, &status);
        }
    }

    start = MPI_Wtime();
    for (double t = 0; t < T; t += tau)
    {
        if (rank > 0)
        {
            MPI_Isend(&u_local[1], 1, MPI_DOUBLE, rank - 1, TAG, MPI_COMM_WORLD, &send_request);        // sending left border
            MPI_Irecv(&u_local[0], 1, MPI_DOUBLE, rank - 1, TAG, MPI_COMM_WORLD, &recv_request);        // receiving left border
        }
        if (rank < size - 1)
        {
            MPI_Isend(&u_local[local_N], 1, MPI_DOUBLE, rank + 1, TAG, MPI_COMM_WORLD, &send_request);      // sending right border
            MPI_Irecv(&u_local[local_N + 1], 1, MPI_DOUBLE, rank + 1, TAG, MPI_COMM_WORLD, &recv_request);  // receiving right border
        }
        // waiting
        if (rank > 0)
        {
            MPI_Wait(&send_request, &status);
            MPI_Wait(&recv_request, &status);
        }

        if (rank < size - 1)
        {
            MPI_Wait(&send_request, &status);
            MPI_Wait(&recv_request, &status);
        }
        // computation
        for (int i = 1; i <= local_N; i++)
        {
            u_local_new[i] = u_local[i] + ((k * tau) / (h * h)) * (u_local[i + 1] - 2 * u_local[i] + u_local[i - 1]);
        }

        std::swap(u_local_new, u_local);
    }

    if (rank == 0)
    {
        for (int i = 1; i <= local_N; i++)
        {
            u[i] = u_local[i];
        }
        for (int i = 1; i < size; i++)
        {
            MPI_Recv(u.data() + 1 + i * local_N, local_N, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD, &status);
        }
    }
    else
    {
        MPI_Send(u_local.data() + 1, local_N, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
    }

    finish = MPI_Wtime();
    time = finish - start;
    if (rank == 0)
    {
        std::cout << time << " sec" << std::endl;
    }
    
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
    
    MPI_Finalize();

    return 0;
}
