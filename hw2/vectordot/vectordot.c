#include <immintrin.h>
#include <math.h>

float vectordot_naive(float *A, float *B, int N) {
  float c = 0.f;
  for (int i = 0; i < N; ++i) {
    c += A[i] * B[i];
  }
  return c;
}

float vectordot_fma(float *A, float *B, int N) {
  float c = 0.f;
  //TODO: FILL IN HERE
  __m256 vec_C = _mm256_setzero_ps();

  //code1
  size_t i = 0;
  size_t length = (size_t)N;
  printf("%d\n", N);
  for(i = 0; i < length - 7; i += 8) { //size_t for memory
      __m256 vec_A = _mm256_load_ps(A + i);
      __m256 vec_B = _mm256_load_ps(B + i);
      vec_C = _mm256_fmadd_ps(vec_A, vec_B, vec_C);
      if(length < 8) // i) 8보다 작은 경우
        break;
  }

  if(length % 8 != 0 && length > 8){ // ii) 8보다 큰데 selective loading이 필요한 경우
        __m256 vec_A = _mm256_load_ps(A + i);
        __m256 vec_B = _mm256_load_ps(B + i);
        vec_C = _mm256_fmadd_ps(vec_A, vec_B, vec_C);
  }

  // 합을 구하는 성능 좋은 방법 
  const __m128 vec_C128 = _mm_add_ps(_mm256_extractf128_ps(vec_C, 1), _mm256_castps256_ps128(vec_C));
  const __m128 vec_C64 = _mm_add_ps(vec_C128, _mm_movehl_ps(vec_C128, vec_C128));
  const __m128 vec_C32 = _mm_add_ss(vec_C64, _mm_shuffle_ps(vec_C64, vec_C64, 0x55));
  c =_mm_cvtss_f32(vec_C32);

  //code1: 합을 구하는 naive한 방법
  // float* vec_C_float = (float*)&vec_C;
  // for(int i = 0; i < 8; i++){
  //   c += vec_C_float[i];
  // }
  
  return c;
}
