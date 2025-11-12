#ifndef ALL_REDUCE_HPP
#define ALL_REDUCE_HPP
#include "utils.hpp"

void allReduce(ncclComm_t *comms, cudaStream_t *streams, float **send_buffers, float **recv_buffers, int nDev,
               int *device_ids);

#endif
