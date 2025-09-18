#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Hello from Llaminar! Process " << rank
              << " of " << size << " processes." << std::endl;

    // TODO: Implement LLM inferencing engine with COSMA integration

    // Finalize MPI
    MPI_Finalize();

    return 0;
}