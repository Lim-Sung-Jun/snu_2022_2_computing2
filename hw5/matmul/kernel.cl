#define NUM_WORK_ITEM 32
#define VECTOR_WIDTH 8

__kernel void sgemm(__global float8 *A, __global float8 *B, __global float8 *C, int M, int N, int K){
  const int row = get_local_id(0);
  const int col = get_local_id(1);
  const int global_row = NUM_WORK_ITEM * get_group_id(0) + row;
  const int global_col = (NUM_WORK_ITEM/VECTOR_WIDTH) * get_group_id(1) + col;

  __local float8 tileA[NUM_WORK_ITEM][NUM_WORK_ITEM/VECTOR_WIDTH];
  __local float8 tileB[NUM_WORK_ITEM][NUM_WORK_ITEM/VECTOR_WIDTH];

  float8 inter_value = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

  const int num_tiles = K/NUM_WORK_ITEM;

  for (int t = 0; t < num_tiles; t++){
    const int t_row = NUM_WORK_ITEM * t + row;
    const int t_col = (NUM_WORK_ITEM/VECTOR_WIDTH) * t + col;
    tileA[row][col] = A[global_row * (K/VECTOR_WIDTH) + t_col];
    tileB[row][col] = B[t_row * (N/VECTOR_WIDTH) + global_col];

    barrier(CLK_LOCAL_MEM_FENCE);

    float8 vectorA, vectorB;
    float valA;

    for(int k = 0; k < NUM_WORK_ITEM/VECTOR_WIDTH; k++){
      vectorA = tileA[row][k];
      for(int w = 0; w < VECTOR_WIDTH; w++){
        vectorB = tileB[VECTOR_WIDTH*k + w][col];

        switch(w){
          case 0: valA = vectorA.s0; break;
          case 1: valA = vectorA.s1; break;
          case 2: valA = vectorA.s2; break;
          case 3: valA = vectorA.s3; break;
          case 4: valA = vectorA.s4; break;
          case 5: valA = vectorA.s5; break;
          case 6: valA = vectorA.s6; break;
          case 7: valA = vectorA.s7; break;
        }

        inter_value += vectorB * valA;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
   C[global_row*(N/VECTOR_WIDTH) + global_col] = inter_value;
}