#include "mpi/mpi.h"

int main(int argc, char **argv) {

  // Initialize MPI environment
  MPI_Init(&argc, &argv);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the process
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print the rank and the name of the process
  printf("Hello from process %d of %d on %s\n", world_rank, world_size, processor_name);

  // Finalize MPI environment
  MPI_Finalize();

  return 0;
}
