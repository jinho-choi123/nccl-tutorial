#include "allReduceMPI.hpp"
#include <nccl.h>

void allReduceMPI(ncclComm_t comm, cudaStream_t &stream, float *send_buffer, float *recv_buffer, int buffer_size,
                  int world_size, int world_rank) {

  printf("Starting MPI-AllReduce communication...\n");
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
  
  // Make Start and End event to record the elapsed time
  cudaEvent_t start, end;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&end));

  // Record the start event
  CUDACHECK(cudaEventRecord(start, stream));

  // Call ncclAllReduce
  NCCLCHECK(ncclAllReduce(send_buffer, recv_buffer, buffer_size, ncclFloat, ncclSum, comm, stream));

  // Record the end event
  CUDACHECK(cudaEventRecord(end, stream));

  // Synchronize the stream
  CUDACHECK(cudaStreamSynchronize(stream));

  // Calculate the elapsed time
  float elapsed_time;
  CUDACHECK(cudaEventElapsedTime(&elapsed_time, start, end));
  printf("Elapsed time: %f ms\n", elapsed_time);

  // Print the results
  float *host_buffer2 = (float *)malloc(buffer_size * sizeof(float));
  for (int i = 0; i < buffer_size; i++) {
    CUDACHECK(cudaMemcpy(&host_buffer2[i], &recv_buffer[i], sizeof(float), cudaMemcpyDeviceToHost));
  }
  printf("All-reduce results: \n");
  for (int i = 0; i < buffer_size; i++) {
    printf("%f ", host_buffer2[i]);
  }
  printf("\n");

  // Free host_buffer
  free(host_buffer2);

  // Free the events
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(end));

  printf("MPI-AllReduce communication completed successfully\n");
}
