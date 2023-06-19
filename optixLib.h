#pragma once


#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <iostream>

#define CheckCudaErrors(val) CheckCuda((val), #val, __FILE__, __LINE__)
inline void CheckCuda(cudaError_t result, char const* const func,
    const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
            << " (" << cudaGetErrorString(result) << ") "
            << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

#define CheckOptiXErrors(val) CheckOptix((val), #val, __FILE__, __LINE__)
inline void CheckOptix(OptixResult res, const char* call, const char* file, unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        std::cerr << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
        exit(98);
    }
}

#define CheckOptiXErrorsLog(val)                                                        \
    do {                                                                                \
        char    LOG[2048];                                                              \
        size_t  LOG_SIZE = sizeof(LOG);                                                 \
        CheckOptixLog((val), LOG, sizeof(LOG), LOG_SIZE, #val, __FILE__, __LINE__);   \
    } while (false)
inline void CheckOptixLog(OptixResult  res,
    const char* log,
    size_t       sizeof_log,
    size_t       sizeof_log_returned,
    const char* call,
    const char* file,
    unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        std::cerr << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
            << log << (sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "") << '\n';
    }
}


inline void InitCudaAndOptix() {
    std::clog << "Initializing CUDA..." << std::endl;

    cudaFree(0);

    int cudaDevicesNum = 0;
    cudaGetDeviceCount(&cudaDevicesNum);

    if (cudaDevicesNum == 0) {
        throw std::runtime_error("No CUDA capable devices found");
    }

    std::clog << "Successfully initialized CUDA !" << std::endl;

    std::clog << "Initializing Optix..." << std::endl;
    CheckOptiXErrors(optixInit());

    std::clog << "Successfully initialized Optix !" << std::endl;
}