#include "allReduceMPI.hpp"
#include "sendRecvMPI.hpp"
#include "allGatherMPI.hpp"

int main(int argc, char **argv) {

  // Initialize MPI environment
  MPI_Init(&argc, &argv);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Setup NCCL related variables
  ncclUniqueId nccl_group_id;
  ncclComm_t comm;
  float *send_buffer, *recv_buffer;
  cudaStream_t stream;

  // generate nccl_group_id at root process(world_rank == 0)
  if (world_rank == 0) {
    ncclGetUniqueId(&nccl_group_id);
  }
  // broadcast nccl_group_id to all processes
  // so that all processes have the same nccl_group_id
  MPI_Bcast(&nccl_group_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

  // pick a GPU based on localRank, allocate device buffers
  // NOTE: Assuming we are using a single-node with multiple GPUS,
  // we can regard the world_rank as the local_rank
  int local_rank = world_rank;
  CUDACHECK(cudaSetDevice(local_rank));
  CUDACHECK(cudaMalloc(&send_buffer, BUFFER_SIZE * sizeof(float)));
  CUDACHECK(cudaMalloc(&recv_buffer, BUFFER_SIZE * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&stream));

  // initialize NCCL communicator
  NCCLCHECK(ncclCommInitRank(&comm, world_size, nccl_group_id, world_rank));

  // #### Main Content of the program ####

  // allReduceMPI
  // allReduceMPI(comm, stream, send_buffer, recv_buffer, BUFFER_SIZE, world_size, world_rank);

  // sendRecvMPI
  // sendRecvMPI(comm, stream, send_buffer, recv_buffer, BUFFER_SIZE, world_size, world_rank);

  // allGatherMPI
  allGatherMPI(comm, stream, send_buffer, recv_buffer, BUFFER_SIZE, world_size, world_rank);

  // #### End of Main Content of the program ####

  // clean up
  CUDACHECK(cudaFree(send_buffer));
  CUDACHECK(cudaFree(recv_buffer));
  NCCLCHECK(ncclCommDestroy(comm));
  CUDACHECK(cudaStreamDestroy(stream));

  // Finalize MPI environment
  MPI_Finalize();
  return 0;
}
