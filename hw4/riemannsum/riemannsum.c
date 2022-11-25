#include <mpi.h>

#include "riemannsum.h"
#include "util.h"

double riemannsum(int num_intervals, int mpi_rank, int mpi_world_size,
                  int threads_per_process) {
  double pi = 0;
  double h = 1.0 / (double)num_intervals;

  // TODO: Parallelize the code using mpi_world_size processes (1 process per
  // node.
  // In total, (mpi_world_size * threads_per_process) threads will collaborate
  // to compute the Riemann sum.
  double sum = 0;
  int part = num_intervals / mpi_world_size;
  int remainder = num_intervals % mpi_world_size;

  if (mpi_rank == 0) {
    for (int i = 1; i < part; i++) {
      double x = h * ((double)i - 0.5);
      sum += h * f(x);
    }
    if(1){
      int start_index;
      if(mpi_world_size > num_intervals){
        start_index = mpi_world_size * part + 1;
      }else{
        start_index = mpi_world_size * part;
      }
      for (int i = start_index; i <= mpi_world_size * part + remainder; i++) {
        double x = h * ((double)i - 0.5);
        sum += h * f(x);
      }
    }
  }
  else{
    #pragma omp parallel for reduction(+:sum) 
    for(int i = mpi_rank * part; i < mpi_rank * part + part; i++){
      double x = h * ((double)i - 0.5);
      sum += h * f(x);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  // Rank 0 should return the estimated PI value
  // Other processes can return any value (don't care)
  return pi;
}