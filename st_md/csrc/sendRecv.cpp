#include "sendRecv.hpp"

void sendRecv(ncclComm_t *comms, cudaStream_t *streams, float **send_buffers, float **recv_buffers, int nDev,
              int *device_ids) {
  printf("Starting NCCL-SendRecv communication...\n");
  printf("GPU_COUNT: %d\n", nDev);

  // Malloc host_buffer for initialization
  float *host_buffer1 = (float *)malloc(BUFFER_SIZE * sizeof(float));
  for (int i = 0; i < BUFFER_SIZE; i++) {
    host_buffer1[i] = 4.0f;
  }
  printf("Each device will send a buffer of size %d with value 4.0f\n", BUFFER_SIZE);

  // Initialize the send and receive buffers
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaMemcpy(send_buffers[i], host_buffer1, BUFFER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(recv_buffers[i], 0, BUFFER_SIZE * sizeof(float)));
  }

  // Free the host buffer
  free(host_buffer1);

  // Start a NCCL group
  // This is used to group all the NCCL operations that need to be done together
  // Used when multiple devices are involved in a single thread execution
  NCCLCHECK(ncclGroupStart());

  // Insert sendRecv operations for each device's cuda stream
  for (int i = 0; i < nDev; i++) {
    NCCLCHECK(ncclSend(send_buffers[i], BUFFER_SIZE, ncclFloat, 1, comms[i], streams[i]));
    NCCLCHECK(ncclRecv(recv_buffers[i], BUFFER_SIZE, ncclFloat, 0, comms[i], streams[i]));
  }

  NCCLCHECK(ncclGroupEnd());

  // synchronize all the streams
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaSetDevice(device_ids[i]));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
  }

  // print the results
  float *host_buffer2 = (float *)malloc(BUFFER_SIZE * sizeof(float));
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaMemcpy(host_buffer2, recv_buffers[i], BUFFER_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  }
  printf("SendRecv results: \n");
  for (int i = 0; i < BUFFER_SIZE; i++) {
    printf("%f ", host_buffer2[i]);
  }
  printf("\n");
  free(host_buffer2);

  printf("NCCL-SendRecv communication completed successfully\n");
}
