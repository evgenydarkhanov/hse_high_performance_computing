#include <iostream>
#include <vector>
#include <mpi.h>

/*
// default parameters
const double L = 1.0;         // rod's length
const double k = 1.0;         // coef
const double h = 0.02;        // space step
const double tau = 0.0002;    // time step
const double T = 0.01;        // time
*/

const int num_points = 50;

const double L = 1.0;
const double k = 1.0;
const double h = L / num_points;
const double tau = (h * h) / (2 * k);
const double T = 500 * tau;

void calculate_temperatures(double* u, double* u_new, int N, double k, double tau, double h)
{
    u[0] = 0.0;
    u_new[N - 1] = 0.0;
    for (int i = 1; i < N - 1; i++)
    {
        u_new[i] = u[i] + ((k * tau) / (h * h)) * (u[i + 1] - 2 * u[i] + u[i - 1]);
    }
}
/*
void write_to_file(double* u, int N, int step, std::ofstream& output_file)
{
    // need #include <fstream>
    for (int i = 0; i < N; i++)
    {
        output_file << u[i] << " ";
    }
    output_file << std::endl;
}
*/
int main()
{
    double start, finish, time;
    int N = static_cast<int> (L / h);      // num of space and time points

    // std::ofstream output_file("temperatures.txt");

    std::vector<double> u(N, 0.0);
    std::vector<double> u_new(N, 0.0);  // arrays for temperatures

    for (int i = 1; i < N - 1; i++)     // start condition
    {
        u[i] = 1.0;
    }

    // write_to_file(u, N, 0, output_file);

    // COMPUTATION
    start = MPI_Wtime();
    for (double t = 0; t < T; t += tau)
    {
        calculate_temperatures(u.data(), u_new.data(), N, k, tau, h);
        // write_to_file(u_new, N, n, output_file);

        // copying new values into the old array
        for (int i = 0; i < N; i++)
        {
            u[i] = u_new[i];
        }
    }

    finish = MPI_Wtime();
    time = finish - start;
    std::cout << time << " sec"<< std::endl;
    // output_file.close();

    return 0;
}
