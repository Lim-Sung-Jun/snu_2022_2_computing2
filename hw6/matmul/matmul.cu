#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define MAX_NUM_GPU 4
#define TILE_WIDTH 16
int num_devices = 0;

__global__ void matmul_kernel(float *A, float *B, float *C, const int M, const int K, const int N)
{
    __shared__ float Asub[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bsub[TILE_WIDTH][TILE_WIDTH];
 
    int by = blockIdx.x, bx = blockIdx.y;
    int ty = threadIdx.x, tx = threadIdx.y;
 
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
 
    float Pvalue = 0;
    for (int ph = 0; ph < ceil(K / (float)TILE_WIDTH); ++ph) {
        Asub[ty][tx] = A[Row*K + ph*TILE_WIDTH + tx];
        Bsub[ty][tx] = B[(ph*TILE_WIDTH + ty)*K + Col];
 
        __syncthreads();
 
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += Asub[ty][k] * Bsub[k][tx];
        }
 
        __syncthreads();
    }
    C[Row*K + Col] = Pvalue;
}

static int mpi_rank, mpi_world_size;

// Array of device (GPU) pointers
static float *a_d[MAX_NUM_GPU];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  // MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  if(1){
  // 각 rank마다 start, end값을 가진다.
    int M_start = (M / mpi_world_size) * mpi_rank;
    int M_end = (M / mpi_world_size) * (mpi_rank + 1);
    // printf("mpi_rank[%d]: start:%d, end:%d\n",mpi_rank, M_start, M_end);
    // node마다 행렬을 나눠줬다.
    if(mpi_rank == 0){
      #pragma omp parallel for num_threads(mpi_world_size - 1)
      for(int i = 1; i < mpi_world_size; i++)
      {
        // 1,2,3 rank의 start, end값은 모르기때문에 이렇게 처리해준다.
        int M_start = i * (M / mpi_world_size);
        int M_end = (i + 1) * (M / mpi_world_size);
        MPI_Send(A + M_start * K, (M_end - M_start) * K, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
        MPI_Send(B, K*N, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
      }

    }else{
      MPI_Recv((void*)(A + M_start * K), (M_end - M_start) * K, MPI_FLOAT, 0, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Recv((void*)B, K*N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // MPI_Barrier(MPI_COMM_WORLD);

    // 여기부터는 Rank마다 matrixA의 원소가 달라질 것이다.
    // Upload A and B matrix to every GPU
    for (int i = 0; i < num_devices; i++) {
      // printf("mpi_rank[%d]/device[%d]: start_address:%d, amount:%d\n",mpi_rank, i, (Mbegin[i] + M_start), (Mend[i] - Mbegin[i]));
      CUDA_CALL(cudaMemcpy(a_d[i], A + (Mbegin[i] + M_start) * K,
                          (Mend[i] - Mbegin[i]) * K * sizeof(float),
                          cudaMemcpyHostToDevice));
      CUDA_CALL(
          cudaMemcpy(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Launch kernel on every GPU
    for (int i = 0; i < num_devices; i++) {
      dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
      dim3 gridDim((Mend[i] - Mbegin[i])/TILE_WIDTH, N/TILE_WIDTH, 1);

      CUDA_CALL(cudaSetDevice(i));
      matmul_kernel<<<gridDim, blockDim>>>(a_d[i], b_d[i], c_d[i], M, N, K); //(Mend[i] - Mbegin[i])
    }

    for (int i = 0; i < num_devices; i++) {
      CUDA_CALL(cudaDeviceSynchronize());
    }

    // Download C matrix from GPUs
    for (int i = 0; i < num_devices; i++) {
      CUDA_CALL(cudaMemcpy(C + (M_start + Mbegin[i]) * N, c_d[i],
                          (Mend[i] - Mbegin[i]) * N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    }

    int size = M / mpi_world_size;
    if(mpi_rank == 0){
      #pragma omp parallel for num_threads(mpi_world_size - 1)
      for(int i = 1; i < mpi_world_size; i++)
      {
        int M_start = i * (M / mpi_world_size);
        // int M_end = (i + 1) * (M / mpi_world_size);
        // MPI_Send(A + M_start * K, (M_end - M_start) * K, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
        // MPI_Send(B, K*N, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
        MPI_Recv(C + M_start * N, size * N, MPI_FLOAT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // printf("[mpi_rank:%d] recv address:%d\n",mpi_rank, M_start * N);
      }
    }else{
      MPI_Send(C + M_start * N, size * N, MPI_FLOAT, 0, mpi_rank, MPI_COMM_WORLD);
      // printf("[mpi_rank:%d] send address:%d\n",mpi_rank, M_start * N);
    }
    // MPI_Barrier(MPI_COMM_WORLD);
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaDeviceSynchronize());
  }
}

void matmul_initialize(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
  
  // Only root process do something
  int size = M / mpi_world_size;
  int residual = M % mpi_world_size;
  if (mpi_rank == 0) {
    CUDA_CALL(cudaGetDeviceCount(&num_devices));
    printf("size: %d\n",size);

    printf("Using %d devices\n", num_devices);
    for (int i = 0; i < num_devices; i++) {
      cudaDeviceProp prop;
      CUDA_CALL(cudaGetDeviceProperties(&prop, i));

      // Try printing more detailed information here
      printf("GPU %d: %s\n", i, prop.name);
    }

    if (num_devices <= 0) {
      printf("No CUDA device found. Aborting\n");
      exit(1);
    }

    // Setup problem size for each GPU
    for (int i = 0; i < num_devices; i++) {
      Mbegin[i] = (size / num_devices) * i;
      Mend[i] = Mbegin[i] + (size / num_devices);
      // printf("Mbegin[%d]: %d, Mend[%d]: %d\n",i,Mbegin[i],i,Mend[i]);
    }
    if(residual != 0){
      Mend[num_devices - 1] = M;
    }
    // Mend[num_devices - 1] = M;

    // Allocate device memory for each GPU
    for (int i = 0; i < num_devices; i++) {
      CUDA_CALL(cudaSetDevice(i));
      CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
      CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
      CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
    }
  }else{
    CUDA_CALL(cudaGetDeviceCount(&num_devices));

    for (int i = 0; i < num_devices; i++) {
      cudaDeviceProp prop;
      CUDA_CALL(cudaGetDeviceProperties(&prop, i));
    }

    // Setup problem size for each GPU
    for (int i = 0; i < num_devices; i++) {
      Mbegin[i] = (size / num_devices) * i;
      Mend[i] = Mbegin[i] + (size / num_devices);
      // printf("Mbegin[%d]: %d, Mend[%d]: %d\n",i,Mbegin[i],i,Mend[i]);
    }

    for (int i = 0; i < num_devices; i++) {
      CUDA_CALL(cudaSetDevice(i));
      CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
      CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
      CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
    }
    
  }
}

void matmul_finalize() {

  // Only root process do something
  if (mpi_rank == 0) {
    // Free all GPU memory
    for (int i = 0; i < num_devices; i++) {
      CUDA_CALL(cudaFree(a_d[i]));
      CUDA_CALL(cudaFree(b_d[i]));
      CUDA_CALL(cudaFree(c_d[i]));
    }
  }
}