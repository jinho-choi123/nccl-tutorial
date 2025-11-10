#ifndef UTILS_HPP
#define UTILS_HPP

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_COUNT 2

// MACROS FOR CUDA CHECKS
#define CUDACHECK(cmd)                                                                                                 \
  do {                                                                                                                 \
    cudaError_t e = cmd;                                                                                               \
    if (e != cudaSuccess) {                                                                                            \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));                            \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

// MACROS FOR NCCL CHECKS
#define NCCLCHECK(cmd)                                                                                                 \
  do {                                                                                                                 \
    ncclResult_t r = cmd;                                                                                              \
    if (r != ncclSuccess) {                                                                                            \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                            \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define MPICHECK(cmd)                                                                                                  \
  do {                                                                                                                 \
    int e = cmd;                                                                                                       \
    if (e != MPI_SUCCESS) {                                                                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                                                 \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#endif
