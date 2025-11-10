#include "sendRecv.hpp"

void sendRecv() {
  printf("Starting NCCL-SendRecv communication...\n");
  printf("GPU_COUNT: %d\n", GPU_COUNT);

  // Define the communicators for each GPU
  ncclComm_t comms[GPU_COUNT];

  // manage 2 devices
  int nDev = GPU_COUNT;
  int device_ids[GPU_COUNT] = {0, 1};
  int buffer_size = 32;

  // Define the device to send to and receive from
  int send_device = 0;
  int recv_device = 1;

  // Define buffer pointer storage
  float **send_buffers = (float **)malloc(nDev * sizeof(float *));
  float **recv_buffers = (float **)malloc(nDev * sizeof(float *));

  // Define cuda stream pointer storage
  cudaStream_t *streams = (cudaStream_t *)malloc(nDev * sizeof(cudaStream_t));

  // Malloc host_buffer for initialization
  float *host_buffer1 = (float *)malloc(buffer_size * sizeof(float));
  for (int i = 0; i < buffer_size; i++) {
    host_buffer1[i] = 4.0f;
  }
  printf("device 0 will send a buffer of size %d with value 4.0f to device 1\n", buffer_size);

  // Initialize CUDA devices
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaSetDevice(device_ids[i]));

    // Allocate memory for the send and receive buffers
    CUDACHECK(cudaMalloc(&send_buffers[i], buffer_size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv_buffers[i], buffer_size * sizeof(float)));

    // Initialize the send and receive buffers
    // Send buffer is initialized to 4.0f
    // Receive buffer is initialized to 0
    CUDACHECK(cudaMemcpy(send_buffers[i], host_buffer1, buffer_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(recv_buffers[i], 0, buffer_size * sizeof(float)));

    // Create a cuda stream
    CUDACHECK(cudaStreamCreate(&streams[i]));
  }

  // Free host_buffer
  free(host_buffer1);

  // Initialize NCCL communicators
  NCCLCHECK(ncclCommInitAll(comms, nDev, device_ids));

  // #### Main Content of the program ####

  // Start a NCCL group
  // This is used to group all the NCCL operations that need to be done together
  // Used when multiple devices are involved in a single thread execution
  NCCLCHECK(ncclGroupStart());

  // Insert send and recv operations for each device's cuda stream
  for (int i = 0; i < nDev; i++) {
    if (i == send_device) {
      // Send the buffer to device 1
      NCCLCHECK(ncclSend(send_buffers[i], buffer_size, ncclFloat, 1, comms[i], streams[i]));
    } else if (i == recv_device) {
      // Receive the buffer from device 0
      NCCLCHECK(ncclRecv(recv_buffers[i], buffer_size, ncclFloat, 0, comms[i], streams[i]));
    } else {
      // Do nothing
      continue;
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // synchronize all the streams
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaSetDevice(device_ids[i]));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
  }

  // print the results
  float *host_buffer2 = (float *)malloc(buffer_size * sizeof(float));
  printf("SendRecv results: \n");
  CUDACHECK(cudaSetDevice(device_ids[recv_device]));
  CUDACHECK(cudaMemcpy(host_buffer2, recv_buffers[recv_device], buffer_size * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Device %d: ", device_ids[recv_device]);
  // print the results
  for (int j = 0; j < buffer_size; j++) {
    printf("%f ", host_buffer2[j]);
  }
  printf("\n\n\n");

  // free the host buffer
  free(host_buffer2);

  printf("\n");

  // #### End of Main Content of the program ####

  // free device buffers
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaSetDevice(device_ids[i]));
    CUDACHECK(cudaFree(send_buffers[i]));
    CUDACHECK(cudaFree(recv_buffers[i]));
  }

  // finalize NCCL communicators
  for (int i = 0; i < nDev; i++) {
    NCCLCHECK(ncclCommDestroy(comms[i]));
  }

  // free the cuda streams
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaSetDevice(device_ids[i]));
    CUDACHECK(cudaStreamDestroy(streams[i]));
  }

  printf("NCCL-SendRecv communication completed successfully\n");
}
