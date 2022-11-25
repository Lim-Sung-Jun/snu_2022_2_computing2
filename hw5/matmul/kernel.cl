// super super slow sgemm kernel by jinpyo
__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  int i = get_global_id(0); // row index of C
  int j = get_global_id(1); // column index of C
  if (i >= M || j >= N) return; // boundary check

  C[i * N + j] = 0;
  for (int k = 0; k < K; k++) {
    C[i * N + j] += A[i * K + k] * B[k * N + j];
  }
}
