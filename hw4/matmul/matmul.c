#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int threads_per_process, int mpi_rank, int mpi_world_size) {

  // // TODO: FILL_IN_HERE
  int interval = M / mpi_world_size;
  int remainder = M % mpi_world_size;
  float r;

  if (mpi_rank == 0){
    MPI_Bcast(A,M*K,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(B,K*N,MPI_FLOAT,0,MPI_COMM_WORLD);

    // interval 처리
    for(int i = 1; i < mpi_world_size; i++){
      MPI_Recv(C + i * N * interval, interval * N, MPI_FLOAT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    for (int k=0; k<K; k++) {
      #pragma omp parallel for
      for (int i=0; i<interval; i++) {
          r = *(A + i*K + k);
          for (int j=0; j<N; j++){
            *(C + i*N + j) += r * *(B + j + k*N);
          }
      }
    }

    // residual 처리
    if(remainder != 0){
      for (int k=0; k<K; k++) {
        #pragma omp parallel for
        for (int i=interval * mpi_world_size; i<interval * mpi_world_size + remainder; i++) {
            r = *(A + i*K + k);
            for (int j=0; j<N; j++){
              *(C + i*N + j) += r * *(B + j + k*N);
            }
        }
      }

    }

    //
  }  
  else if(mpi_rank != 0)// interval * (N + 1)
  {
    MPI_Bcast(A,M*K,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(B,K*N,MPI_FLOAT,0,MPI_COMM_WORLD);
    //
    for (int k=0; k<K; k++){
      #pragma omp parallel for
      for (int i=mpi_rank * interval; i< mpi_rank * interval + interval; i++) {
          r = *(A + i*K + k);
          for (int j=0; j<N; j++){
             *(C + i*N + j) += r * *(B + j + k*N);
          }
      }
    }
    //
		MPI_Send(C + mpi_rank * N * interval, interval * N, MPI_FLOAT, 0, mpi_rank, MPI_COMM_WORLD);
  }
}
  