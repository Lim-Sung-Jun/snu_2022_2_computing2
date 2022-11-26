#define _GNU_SOURCE
#include "matmul.h"
#include "util.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);              \
    exit(EXIT_FAILURE);                                                        \
  }

static cl_int err;
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel;
static cl_mem a_d, b_d, c_d;

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  int r = 0;
  int p_M = M;
  int p_N = N;
  int p_K = K;
  // float *p_a = A;
  // float *p_b = B;
  // float *p_c = C;
  // float *A = A;
  // float *B = B;

  if(M%32 != 0){
    r = 32 - (M-(32*(M/32)));
    M = M + r;
  }

  if(N%32 != 0){
    r = 32 - (N-(32*(N/32)));
    N = N + r;
  }

  if(K%32 != 0){
    r = 32 - (K-(32*(K/32)));
    K = K + r;
  }

  float *A_PAD;
  float *B_PAD;
  float *C_PAD;
  alloc_mat(&A_PAD, M, K);
  alloc_mat(&B_PAD, K, N);
  alloc_mat(&C_PAD, M, N);
  zero_mat(A_PAD, M, K);
  zero_mat(B_PAD, K, N);
  zero_mat(C_PAD, M, N);

  for(int i = 0; i < M; i++){
      for(int j = 0; j < K; j++){
        float value;
        if(i < p_M && j < p_K){
          value = A[i*p_K + j];
        }else{
          value = 0;
        }
        A_PAD[i*K + j] = value;
      }
  }
  for(int i = 0; i < K; i++){
      for(int j = 0; j < N; j++){
        float value;
        if(i < p_K && j < p_N){
          value = B[i*p_N + j];
        }else{
          value = 0;
        }
        B_PAD[i*N + j] = value;
      }
  }



  // Write to GPU; A (cpu) -> a_d (gpu), B (cpu) -> b_d (gpu)
  err = clEnqueueWriteBuffer(queue, a_d, CL_TRUE, 0, M * K * sizeof(float), A_PAD, // h_a, h_b? oo
                             0, NULL, NULL);
  CHECK_ERROR(err);
  err = clEnqueueWriteBuffer(queue, b_d, CL_TRUE, 0, K * N * sizeof(float), B_PAD,
                             0, NULL, NULL);
  CHECK_ERROR(err);
  
  err = clFinish(queue);
  CHECK_ERROR(err);

  // Setup kernel arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 3, sizeof(int), &M);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 4, sizeof(int), &N);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 5, sizeof(int), &K);
  CHECK_ERROR(err);

  // Setup global work size and local work size // when we set the global work size as below code line, then we can allocate each kernel to thread.
  size_t gws[2] = {(size_t)M, (size_t)N/8}, lws[2] = {32, 32/8}; // local size is too small i think
  for (int i = 0; i < 2; ++i) {
    // By OpenCL spec, global work size should be MULTIPLE of local work size
    // e.g., gws = 25, lws = 16, then (25 + 16 - 1) / 16 * 16 = 40 / 16 * 16 = 2
    // * 16 = 32
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  // Run kernel // kernel index space will be set in here
  // input padding kernel
  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, lws, 0, NULL, NULL); // Enqueues a command to execute a kernel on a device.
  // output padding kernel
  CHECK_ERROR(err);

  err = clFinish(queue);
  CHECK_ERROR(err);

  // Read from GPU; c_d (gpu) -> C (cpu)
  err = clEnqueueReadBuffer(queue, c_d, CL_TRUE, 0, M * N * sizeof(float), C_PAD, 0,
                            NULL, NULL);
  CHECK_ERROR(err);

  for(int i = 0; i < M; i++){
      for(int j = 0; j < N; j++){
        if(i < p_M && j < p_N){
          C[i*p_N + j] = C_PAD[i*N + j];
        }
      }
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  err = clFinish(queue);
  CHECK_ERROR(err);
}

static void print_platform_info(cl_platform_id platform) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz));
  buf = (char *)malloc(sz);
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, buf, NULL));
  printf("Detected OpenCL platform: %s\n", buf);
  free(buf);
}

static void print_device_info(cl_device_id device) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
  buf = (char *)malloc(sz);
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
  printf("Detected OpenCL device: %s\n", buf);
  free(buf);
}

// pass
static cl_program create_and_build_program_with_source(cl_context context,
                                                       cl_device_id device,
                                                       const char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);
  char *source_code = (char *)malloc(source_size + 1);
  size_t ntotal = 0;
  while (ntotal < source_size) {
    int nread = fread(source_code, sizeof(char), source_size, file);
    ntotal += nread;
  }
  source_code[source_size] = '\0';
  fclose(file);
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&source_code, &source_size, &err);
  CHECK_ERROR(err);
  free(source_code);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                      NULL, &log_size));
    char *log = (char *)malloc(log_size + 1);
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                      log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }
  CHECK_ERROR(err);
  return program;
}

void matmul_initialize(int M, int N, int K) {
  int r = 0;
  if(M%32 != 0){
    r = 32 - (M-(32*(M/32)));
    M = M + r;
  }

  if(N%32 != 0){
    r = 32 - (N-(32*(N/32)));
    N = N + r;
  }

  if(K%32 != 0){
    r = 32 - (K-(32*(K/32)));
    K = K + r;
  }

  // Get OpenCL platform
  err = clGetPlatformIDs(1, &platform, NULL); // obtain the list of platforms available, The number of cl_platform_id entries 
  CHECK_ERROR(err);
  print_platform_info(platform);

  // Get OpenCL device (only 1)
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); // Obtain the list of devices available on a platform, gpu type
  CHECK_ERROR(err);
  print_device_info(device);

  // Create OpenCL context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); // obtain context
  CHECK_ERROR(err);

  // Create OpenCL command queue
  queue = clCreateCommandQueue(context, device, 0, &err); // obtain command queue
  CHECK_ERROR(err);

  // Compile program from "kernel.cl"
  program = create_and_build_program_with_source(context, device, "kernel.cl"); // program with kernel
  // input padding kernel
  // output padding kernel

  // Extract kernel from compiled program
  kernel = clCreateKernel(program, "sgemm", &err);
  CHECK_ERROR(err);

  // Create GPU buffers
  a_d = clCreateBuffer(context, CL_MEM_READ_WRITE, M * K * sizeof(float), NULL,
                       &err);
  CHECK_ERROR(err);
  b_d = clCreateBuffer(context, CL_MEM_READ_WRITE, K * N * sizeof(float), NULL,
                       &err);
  CHECK_ERROR(err);
  c_d = clCreateBuffer(context, CL_MEM_READ_WRITE, M * N * sizeof(float), NULL,
                       &err);
  CHECK_ERROR(err);
}

void matmul_finalize() {}
