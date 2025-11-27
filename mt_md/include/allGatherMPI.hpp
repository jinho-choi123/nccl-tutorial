#ifndef ALL_GATHER_MPI_HPP
#define ALL_GATHER_MPI_HPP
#include "utilsMPI.hpp"

void allGatherMPI(ncclComm_t comm, cudaStream_t &stream, float *send_buffer, float *recv_buffer, int buffer_size,
                  int world_size, int world_rank);

#endif
