#include <immintrin.h>
#include <math.h>

#include <stdio.h>

void prefix_sum_sequential(double *out, const double *in, int N) {
  out[0] = in[0];
  for (int i = 1; i < N; ++i) {
    out[i] = in[i] + out[i - 1];
  }
}

void prefix_sum_parallel(double *out, const double *in, int N) {

  //

  // global하게 설정해야 아래 코드에서 스레드가 offset값을 공유할 수 있다.
  double *offset;
  #pragma omp parallel
  {
    int thread_idx = omp_get_thread_num();
    int threads_num = omp_get_num_threads();
    offset = malloc(sizeof(double) * (threads_num + 1));
    offset[0] = 0;

    // partial sum
    double p_s = 0;
    #pragma omp for
    for(int i = 0; i < N; i++)
    {
        p_s += in[i];
        out[i] = p_s;
    }
    offset[thread_idx + 1] = p_s;

    // partial sum + offset
    #pragma omp barrier // 제일 중요!

    double temp = 0;
    for(int i = 0; i < (thread_idx + 1); i++)
    {
      temp += offset[i];
    }

    // 각 스레드 temp에 offset이 더해졌으므로 각 원소에 합한다. 아까랑 똑같은 배열이 각 스레드마다 할당될테니깐 그건 걱정없어도 되고 temp는 각 스레드마다 구해줘서 상관없다.
    #pragma omp for 
    for(int i = 0; i < N; i++)
    {
      out[i] += temp;
    }
  }
  //
}
