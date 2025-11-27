#include "allGatherMPI.hpp"
#include <nccl.h>

void allGatherMPI(ncclComm_t comm, cudaStream_t &stream, float *send_buffer, float *recv_buffer, int buffer_size, int world_size, int world_rank) {
    printf("\nStarting MPI-AllGather communication...\n");
    printf("World size: %d, World rank: %d\n", world_size, world_rank);

    int send_buffer_size = buffer_size / world_size;

    //Malloc host_buffer for initialization
    float *host_buffer1 = (float *)malloc(send_buffer_size * sizeof(float));
    for (int i = 0; i < send_buffer_size; i++) {
        host_buffer1[i] = 4.0f;
    }

    printf("Each device will all-gather a buffer of size %d with value 4.0f\n", send_buffer_size);

    // Initialize the send and receive buffers
    for (int i = 0; i < send_buffer_size; i++) {
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

    // Call ncclAllGather
    NCCLCHECK(ncclAllGather(send_buffer, recv_buffer, send_buffer_size, ncclFloat, comm, stream));

    // Record the end event
    CUDACHECK(cudaEventRecord(end, stream));

    // Synchronize the stream
    CUDACHECK(cudaStreamSynchronize(stream));

    // Calculate the elapsed time
    float elapsed_time;
    CUDACHECK(cudaEventElapsedTime(&elapsed_time, start, end));
    printf("\n\nElapsed time: %f ms\n", elapsed_time);

    // Free the events
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(end));

    // Print the results
    float *host_buffer2 = (float *)malloc(buffer_size * sizeof(float));
    for (int i = 0; i < buffer_size; i++) {
        CUDACHECK(cudaMemcpy(&host_buffer2[i], &recv_buffer[i], sizeof(float), cudaMemcpyDeviceToHost));
    }
    // printf("All-gather results: \n");
    // for (int i = 0; i < buffer_size; i++) {
    //     printf("%f ", host_buffer2[i]);
    // }
    printf("\n");
    free(host_buffer2);
    printf("MPI-AllGather communication completed successfully\n");
}

