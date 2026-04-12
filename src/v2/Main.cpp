/**
 * @file Main.cpp
 * @brief Llaminar v2 entry point
 */

#include "app/AppLifecycle.h"
#include <cstdlib>
#include <unistd.h>
#include <iostream>

int main(int argc, char *argv[])
{
    llaminar2::AppLifecycle app;
    int result = app.run(argc, argv);

    // Flush output streams before exit
    std::cout.flush();
    std::cerr.flush();

    if (result == 0)
    {
        // Skip C++ static destructors via _exit() to avoid RCCL/ROCm CLR bugs.
        // RCCL's internal atexit handler segfaults when cleaning up communicators
        // created via ncclCommInitAll on MI60 GPUs, even after proper
        // ncclCommFinalize + ncclCommDestroy. Our resources are already cleaned
        // up by this point — the OS reclaims the rest.
        _exit(0);
    }
    return result;
}
