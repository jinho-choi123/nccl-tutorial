#ifndef SEND_RECV_HPP
#define SEND_RECV_HPP
#include "utils.hpp"

void sendRecv(ncclComm_t *comms, cudaStream_t *streams, float **send_buffers, float **recv_buffers, int nDev,
              int *device_ids);

#endif
