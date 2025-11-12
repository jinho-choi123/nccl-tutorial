#include "sendRecvMPI.hpp"

void sendRecvMPI(ncclComm_t comm, cudaStream_t &stream, float *send_buffer, float *recv_buffer, int buffer_size,
                 int world_size, int world_rank) {

  printf("Starting MPI-SendRecv communication...\n");
  printf("World size: %d, World rank: %d\n", world_size, world_rank);

  // Malloc host_buffer for initialization
  float *host_buffer1 = (float *)malloc(buffer_size * sizeof(float));
  for (int i = 0; i < buffer_size; i++) {
    host_buffer1[i] = 4.0f;
  }
  printf("Each device will send a buffer of size %d with value 4.0f\n", buffer_size);

  // Initialize the send and receive buffers
  for (int i = 0; i < buffer_size; i++) {
    CUDACHECK(cudaMemcpy(&send_buffer[i], &host_buffer1[i], sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(&recv_buffer[i], 0, sizeof(float)));
  }

  // Free host_buffer
  free(host_buffer1);

  // Call ncclSendRecv
  if (world_rank == 0) {
    NCCLCHECK(ncclSend(send_buffer, buffer_size, ncclFloat, 1, comm, stream));
  } else {
    NCCLCHECK(ncclRecv(recv_buffer, buffer_size, ncclFloat, 0, comm, stream));
  }

  // Synchronize the stream
  CUDACHECK(cudaStreamSynchronize(stream));

  // Print the results
  float *host_buffer2 = (float *)malloc(buffer_size * sizeof(float));
  for (int i = 0; i < buffer_size; i++) {
    CUDACHECK(cudaMemcpy(&host_buffer2[i], &recv_buffer[i], sizeof(float), cudaMemcpyDeviceToHost));
  }
  printf("SendRecv results: \n");
  for (int i = 0; i < buffer_size; i++) {
    printf("%f ", host_buffer2[i]);
  }
  printf("\n");
  free(host_buffer2);
  printf("MPI-SendRecv communication completed successfully\n");
}
