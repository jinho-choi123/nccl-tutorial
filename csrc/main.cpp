#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>

#define GPU_COUNT 2

// MACROS FOR CUDA CHECKS
#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Failed: Cuda error %s:%d '%s'\n",             \
          __FILE__,__LINE__,cudaGetErrorString(e));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)
  
// MACROS FOR NCCL CHECKS
#define NCCLCHECK(cmd) do {                         \
ncclResult_t r = cmd;                             \
if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
}                                                 \
} while(0)

int main() {
    // Define the communicators for each GPU
    ncclComm_t comms[GPU_COUNT];

    // manage 2 devices
    int nDev = GPU_COUNT;
    int buffer_size = 32;

    // Define the ranks for each device 
    int dev_rank[GPU_COUNT];

    for (int i = 0; i < nDev; i++) {
        dev_rank[i] = i;
    }

    // Define buffer pointer storage
    float** send_buffers = (float**)malloc(nDev * sizeof(float*));
    float** recv_buffers = (float**)malloc(nDev * sizeof(float*));

    // Define cuda stream pointer storage
    cudaStream_t * streams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * nDev);

    // Malloc host_buffer for initialization
    float* host_buffer1 = (float*)malloc(buffer_size * sizeof(float));
    for (int i = 0; i < buffer_size; i++) {
        host_buffer1[i] = 4.0f;
    }

    // Initialize CUDA devices
    for (int i = 0; i < nDev; i++) {
        CUDACHECK(cudaSetDevice(dev_rank[i]));

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
    NCCLCHECK(ncclCommInitAll(comms, nDev, dev_rank));

    // #### Main Content of the program ####

    // Start a NCCL group
    // This is used to group all the NCCL operations that need to be done together
    // Used when multiple devices are involved in a single thread execution
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; i++) {
        // Insert all reduce operations for each device's cuda stream
        NCCLCHECK(ncclAllReduce((const void *) send_buffers[i], (void *) recv_buffers[i], buffer_size, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    // synchronize all the streams
    for (int i = 0; i < nDev; i++) {
        CUDACHECK(cudaSetDevice(dev_rank[i]));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    // print the results
    float* host_buffer2 = (float*)malloc(buffer_size * sizeof(float));
    for (int i = 0; i < nDev; i++) {
        CUDACHECK(cudaSetDevice(dev_rank[i]));
        CUDACHECK(cudaMemcpy(host_buffer2, recv_buffers[i], buffer_size * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Device %d: ", dev_rank[i]);
        // print the results
        for (int j = 0; j < buffer_size; j++) {
            printf("%f ", host_buffer2[j]);
        }

        printf("\n\n\n");
    }

    // free the host buffer
    free(host_buffer2);
    
    printf("\n");

    // #### End of Main Content of the program ####
    

    // free device buffers
    for (int i = 0; i < nDev; i++) {
        CUDACHECK(cudaSetDevice(dev_rank[i]));
        CUDACHECK(cudaFree(send_buffers[i]));
        CUDACHECK(cudaFree(recv_buffers[i]));
    }

    // finalize NCCL communicators
    for(int i = 0; i < nDev; i++) {
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }

    printf("NCCL communication completed successfully\n");

    return 0;
}
