#include <iostream>
#include <mpi.h>
#include <vector>
#include <cmath>

/*
// default parameters
const double L = 1.0;         // rod's length
const double k = 1.0;         // coef
const double h = 0.02;        // space step
const double tau = 0.0002;    // time step
const double T = 0.01;         // time
*/

const int num_points = 500;

const double L = 1.0;
const double k = 1.0;
const double h = L / num_points;
const double tau = (h * h) / (2 * k);
const double T = 500 * tau;

const int TAG = 123;
const int PI = 3.141592;
/*
double analytical_solution(double x, double t, double L, double k)
{
    double sum = 0.0;
    for (int m = 0; m < 100; m++)
    {
        double m_part = 2 * m + 1;
        double add = (exp((-k * PI * PI * m_part * m_part * t) / (L * L)) / m_part) * sin((PI * m_part * x) / L);
        sum += add;
    }
    sum *= (4/PI);
    return sum;
}
*/
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
    /*    
    if (rank == 0)
    {
        std::cout << time << " sec" << std::endl;
        std::cout << "rank: " << rank << std::endl;
        for (int i = 0; i <= 10; i++)
        {
            double x = i * L / 10;
            double numerical_temp = u[static_cast<int>(x / h)];
            double analytical_temp = analytical_solution(x, T, L, k);
            std::cout << x << "\tnumerical: "  << numerical_temp << "\tanalytical: " << analytical_temp << std::endl;
        }
        std::cout << std::endl;
    }
    */
    MPI_Finalize();

    return 0;
}
