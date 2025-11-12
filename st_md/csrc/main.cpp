#include "allReduce.hpp"
#include "sendRecv.hpp"

int main() {
  // Define the communicators for each GPU
  ncclComm_t comms[GPU_COUNT];

  // manage 2 devices
  int nDev = GPU_COUNT;
  int device_ids[GPU_COUNT] = {0, 1};

  // Define cuda stream pointer storage
  cudaStream_t *streams = (cudaStream_t *)malloc(nDev * sizeof(cudaStream_t));

  // Define buffer pointer storage
  float **send_buffers = (float **)malloc(nDev * sizeof(float *));
  float **recv_buffers = (float **)malloc(nDev * sizeof(float *));

  // Initialize CUDA devices
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaSetDevice(device_ids[i]));

    // Allocate memory for the send and receive buffers
    CUDACHECK(cudaMalloc(&send_buffers[i], BUFFER_SIZE * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv_buffers[i], BUFFER_SIZE * sizeof(float)));

    // Create a cuda stream
    CUDACHECK(cudaStreamCreate(&streams[i]));
  }

  // Initialize NCCL communicators
  NCCLCHECK(ncclCommInitAll(comms, nDev, device_ids));

  // #### Main Content of the program ####

  // allReduce
  allReduce(comms, streams, send_buffers, recv_buffers, nDev, device_ids);

  // sendRecv
  sendRecv(comms, streams, send_buffers, recv_buffers, nDev, device_ids);

  // #### End of Main Content of the program ####

  // Clean up
  for (int i = 0; i < nDev; i++) {

    NCCLCHECK(ncclCommDestroy(comms[i]));

    CUDACHECK(cudaSetDevice(device_ids[i]));
    CUDACHECK(cudaFree(send_buffers[i]));
    CUDACHECK(cudaFree(recv_buffers[i]));
    CUDACHECK(cudaStreamDestroy(streams[i]));
  }

  free(send_buffers);
  free(recv_buffers);
  free(streams);

  return 0;
}
