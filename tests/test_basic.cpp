#include "../src/common.h"
#include "../src/argument_parser.h"
#include <mpi.h>
#include <iostream>

// Basic MPI test function
void runBasicTest(int rank, int size)
{
    std::cout << "Hello from Llaminar! Process " << rank
              << " of " << size << " processes." << std::endl;
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse arguments
    ArgumentParser parser(argc, argv);
    LlaminarParams params;

    if (!parser.parse(params))
    {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        std::cout << "\n=== Basic MPI Test ===" << std::endl;
        std::cout << "Testing basic MPI functionality with " << size << " processes" << std::endl;
    }

    runBasicTest(rank, size);

    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "\n✓ BASIC MPI TEST SUCCESS: All processes responded" << std::endl;
        std::cout << "======================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}