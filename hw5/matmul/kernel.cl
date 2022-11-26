#define NUM_WORK_ITEM 32
#define VECTOR_WIDTH 8

__kernel void sgemm(__global float8 *A, __global float8 *B, __global float8 *C, int M, int N, int K){
  const int row = get_local_id(0);
  const int col = get_local_id(1);
  const int global_row = NUM_WORK_ITEM * get_group_id(0) + row;
  const int global_col = (NUM_WORK_ITEM/VECTOR_WIDTH) * get_group_id(1) + col;

  __local float8 tileA[NUM_WORK_ITEM][NUM_WORK_ITEM/VECTOR_WIDTH];
  __local float8 tileB[NUM_WORK_ITEM][NUM_WORK_ITEM/VECTOR_WIDTH];

  float8 intermediate_val = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

  const int num_tiles = K/NUM_WORK_ITEM;

  for (int t = 0; t < num_tiles; t++){
    const int t_row = NUM_WORK_ITEM * t + row;
    const int t_col = (NUM_WORK_ITEM/VECTOR_WIDTH) * t + col;
    tileA[row][col] = A[global_row * (K/VECTOR_WIDTH) + t_col];
    tileB[row][col] = B[t_row * (N/VECTOR_WIDTH) + global_col];

    barrier(CLK_LOCAL_MEM_FENCE);

    float8 vecA, vecB;
    float valA;

    for(int k = 0; k < NUM_WORK_ITEM/VECTOR_WIDTH; k++){
      vecA = tileA[row][k];
      for(int w = 0; w < VECTOR_WIDTH; w++){
        vecB = tileB[VECTOR_WIDTH*k + w][col];

        switch(w){
          case 0: valA = vecA.s0; break;
          case 1: valA = vecA.s1; break;
          case 2: valA = vecA.s2; break;
          case 3: valA = vecA.s3; break;
          case 4: valA = vecA.s4; break;
          case 5: valA = vecA.s5; break;
          case 6: valA = vecA.s6; break;
          case 7: valA = vecA.s7; break;
        }

        intermediate_val += vecB * valA;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
   C[global_row*(N/VECTOR_WIDTH) + global_col] = intermediate_val;
}