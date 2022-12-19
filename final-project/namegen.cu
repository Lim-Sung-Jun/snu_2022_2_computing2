#include "namegen.h"
#include "util.h"

#include <cassert>
#include <math.h>
#include <vector>

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
#define BATCHSIZE 64
int num_devices = 0;

// Array of device (GPU) pointers
static float *a_d_gpu[MAX_NUM_GPU];
static float *b_d_gpu[MAX_NUM_GPU];
static float *c_d_gpu[MAX_NUM_GPU];
static int Mbegin_gpu[MAX_NUM_GPU], Mend_gpu[MAX_NUM_GPU];

// Defined in main.cpp
extern int mpi_rank, mpi_size;

// You can modify the data structure as you want
struct Tensor {

  // constructor
  /* Alloc memory */
  Tensor(std::vector<int> shape_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    buf = (float *)malloc(n * sizeof(float)); //할당한 shape만큼 메모리에 공간을 준비한다.
  }

  /* Alloc memory and copy */
  Tensor(std::vector<int> shape_, float *buf_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    buf = (float *)malloc(n * sizeof(float));
    memcpy(buf, buf_, n * sizeof(float)); // 해당 공간을 buf로 옮긴다.
  }

  // destructor
  ~Tensor() {
    if (buf != nullptr)
      free(buf);
  }

  void set_SOS() {
    size_t n = num_elem();
    for (size_t i = 0; i < n; i++)
      buf[i] = SOS; // 데이터를 모두 0으로 바꾼다.
  }

  void set_zero() {
    size_t n = num_elem();
    for (size_t i = 0; i < n; i++)
      buf[i] = 0.0; // 데이터를 모두 0으로 바꾼다.
  }

  size_t num_elem() {
    size_t sz = 1;
    for (size_t i = 0; i < ndim; i++)
      sz *= shape[i];
    return sz; // 전체 갯수를 센다.
  }

  // Pointer to data
  float *buf = nullptr; // 데이터를 가리킨다.

  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
  size_t ndim = 0;
  size_t shape[4];
};

/* Network parameters */
Tensor *character_embedding;
Tensor *W_ir0, *W_iz0, *W_in0, *W_ir1, *W_iz1, *W_in1;
Tensor *W_hr0, *W_hz0, *W_hn0, *W_hr1, *W_hz1, *W_hn1;
Tensor *b_ir0, *b_iz0, *b_in0, *b_ir1, *b_iz1, *b_in1;
Tensor *b_hr0, *b_hz0, *b_hn0, *b_hr1, *b_hz1, *b_hn1;

Tensor *b_ir0_stack, *b_iz0_stack, *b_in0_stack, *b_ir1_stack, *b_iz1_stack, *b_in1_stack;
Tensor *b_hr0_stack, *b_hz0_stack, *b_hn0_stack, *b_hr1_stack, *b_hz1_stack, *b_hn1_stack, *b_fc_stack;
Tensor *W_fc, *b_fc;
Tensor *rfloats;

/* input, activations, output */
Tensor *input, *emb_out;
Tensor *hidden0, *hidden1;
Tensor *r0, *r1, *z0, *z1, *n0, *n1, *f, *char_prob;
Tensor *rtmp00, *rtmp01, *rtmp02, *rtmp03, *rtmp04;
Tensor *rtmp10, *rtmp11, *rtmp12, *rtmp13, *rtmp14;
Tensor *ztmp00, *ztmp01, *ztmp02, *ztmp03, *ztmp04;
Tensor *ztmp10, *ztmp11, *ztmp12, *ztmp13, *ztmp14;
Tensor *ntmp00, *ntmp01, *ntmp02, *ntmp03, *ntmp04, *ntmp05;
Tensor *ntmp10, *ntmp11, *ntmp12, *ntmp13, *ntmp14, *ntmp15;
Tensor *htmp00, *htmp01, *htmp02;
Tensor *htmp10, *htmp11, *htmp12;
Tensor *ftmp0;

//gpu
static Tensor *character_embedding_gpu[MAX_NUM_GPU];
static Tensor *W_ir0_gpu[MAX_NUM_GPU], *W_iz0_gpu[MAX_NUM_GPU], *W_in0_gpu[MAX_NUM_GPU], *W_ir1_gpu[MAX_NUM_GPU],*W_iz1_gpu[MAX_NUM_GPU],*W_in1_gpu[MAX_NUM_GPU],*W_fc_gpu[MAX_NUM_GPU];
static Tensor *W_hr0_gpu[MAX_NUM_GPU], *W_hz0_gpu[MAX_NUM_GPU], *W_hn0_gpu[MAX_NUM_GPU], *W_hr1_gpu[MAX_NUM_GPU],*W_hz1_gpu[MAX_NUM_GPU],*W_hn1_gpu[MAX_NUM_GPU];
static Tensor *b_ir0_stack_gpu[MAX_NUM_GPU], *b_iz0_stack_gpu[MAX_NUM_GPU], *b_in0_stack_gpu[MAX_NUM_GPU], *b_ir1_stack_gpu[MAX_NUM_GPU],*b_iz1_stack_gpu[MAX_NUM_GPU],*b_in1_stack_gpu[MAX_NUM_GPU];
static Tensor *b_hr0_stack_gpu[MAX_NUM_GPU], *b_hz0_stack_gpu[MAX_NUM_GPU], *b_hn0_stack_gpu[MAX_NUM_GPU], *b_hr1_stack_gpu[MAX_NUM_GPU],*b_hz1_stack_gpu[MAX_NUM_GPU],*b_hn1_stack_gpu[MAX_NUM_GPU],*b_fc_stack_gpu[MAX_NUM_GPU];
static Tensor *rfloats_gpu[MAX_NUM_GPU];


static Tensor *input_gpu[MAX_NUM_GPU], *emb_out_gpu[MAX_NUM_GPU];
static Tensor *hidden0_gpu[MAX_NUM_GPU], *hidden1_gpu[MAX_NUM_GPU];
static Tensor *r0_gpu[MAX_NUM_GPU], *r1_gpu[MAX_NUM_GPU], *z0_gpu[MAX_NUM_GPU], *z1_gpu[MAX_NUM_GPU], *n0_gpu[MAX_NUM_GPU], *n1_gpu[MAX_NUM_GPU], *f_gpu[MAX_NUM_GPU], *char_prob_gpu[MAX_NUM_GPU];
static Tensor *rtmp00_gpu[MAX_NUM_GPU], *rtmp01_gpu[MAX_NUM_GPU], *rtmp02_gpu[MAX_NUM_GPU], *rtmp03_gpu[MAX_NUM_GPU], *rtmp04_gpu[MAX_NUM_GPU];
static Tensor *rtmp10_gpu[MAX_NUM_GPU], *rtmp11_gpu[MAX_NUM_GPU], *rtmp12_gpu[MAX_NUM_GPU], *rtmp13_gpu[MAX_NUM_GPU], *rtmp14_gpu[MAX_NUM_GPU];
static Tensor *ztmp00_gpu[MAX_NUM_GPU], *ztmp01_gpu[MAX_NUM_GPU], *ztmp02_gpu[MAX_NUM_GPU], *ztmp03_gpu[MAX_NUM_GPU], *ztmp04_gpu[MAX_NUM_GPU];
static Tensor *ztmp10_gpu[MAX_NUM_GPU], *ztmp11_gpu[MAX_NUM_GPU], *ztmp12_gpu[MAX_NUM_GPU], *ztmp13_gpu[MAX_NUM_GPU], *ztmp14_gpu[MAX_NUM_GPU];
static Tensor *ntmp00_gpu[MAX_NUM_GPU], *ntmp01_gpu[MAX_NUM_GPU], *ntmp02_gpu[MAX_NUM_GPU], *ntmp03_gpu[MAX_NUM_GPU], *ntmp04_gpu[MAX_NUM_GPU], *ntmp05_gpu[MAX_NUM_GPU];
static Tensor *ntmp10_gpu[MAX_NUM_GPU], *ntmp11_gpu[MAX_NUM_GPU], *ntmp12_gpu[MAX_NUM_GPU], *ntmp13_gpu[MAX_NUM_GPU], *ntmp14_gpu[MAX_NUM_GPU], *ntmp15_gpu[MAX_NUM_GPU];
static Tensor *htmp00_gpu[MAX_NUM_GPU], *htmp01_gpu[MAX_NUM_GPU], *htmp02_gpu[MAX_NUM_GPU];
static Tensor *htmp10_gpu[MAX_NUM_GPU], *htmp11_gpu[MAX_NUM_GPU], *htmp12_gpu[MAX_NUM_GPU];
static Tensor *ftmp0_gpu[MAX_NUM_GPU];
static Tensor *output_gpu[MAX_NUM_GPU];
/* Operations */

/*
 * Embedding
 * input: [1] (scalar)
 * weight: [NUM_CHAR x EMBEDDING_DIM]
 * output: [EMBEDDING_DIM]
 */
void embedding(Tensor *input, Tensor *weight, Tensor *output) {
  size_t input_N = input->shape[1];
  size_t n = weight->shape[1]; // weight에 shape도 있고 buf가 있다.
  for(size_t i = 0; i < input_N; i++){
    for (size_t j = 0; j < n; j++) {
        int x = (int)input->buf[i];  // 현재 문자가 무엇인지 알아낸다.
        output->buf[j * input_N + i] = weight->buf[x * n + j];
    }
  }  
}

__global__ void embedding_kernel(Tensor *input, Tensor *weight, Tensor *output){
  size_t input_N = input->shape[1];
  size_t n = weight->shape[1];

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < input_N) {
    for (size_t j = 0; j < n; j++) {
      int x = (int)input->buf[i];
      output->buf[j * input_N + i] = weight->buf[x * n + j];
    }
  }
}
/*
 * Elementwise addition
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] + input2->buf[i];
  }
}

__global__ void elemwiseAddKernel(Tensor *input1, Tensor *input2, Tensor *output)
{

    // Calculate the index of the element to be processed
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the index is within the range of the input tensors

    output->buf[i] = input1->buf[i] + input2->buf[i];

}
/*
 * Elementwise (1-x)
 * input: [*]
 * output: [*] (same shape as input)
 */
void elemwise_oneminus(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = 1.0 - x;
  }
}
__global__ void elemwiseOneminusKernel(Tensor *input, Tensor *output)
{

    // Calculate the index of the element to be processed
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the index is within the range of the input tensor

    float x = input->buf[i];
    output->buf[i] = 1.0 - x;

}

/*
 * Elementwise multiplication
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
void elemwise_mul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] * input2->buf[i];
  }
}
__global__ void elemwiseMulKernel(Tensor *input1, Tensor *input2, Tensor *output)
{

    // Calculate the index of the element to be processed
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        output->buf[i] = input1->buf[i] * input2->buf[i];

}
/*
 * Elementwise tanh(x)
 * input: [*]
 * output: [*] (same shape as input)
 */
void elemwise_tanh(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = tanhf(x);
  }
}

__global__ void elemwiseTanhKernel(Tensor *input, Tensor *output)
{
    // Calculate the index of the element to be processed
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = input->buf[i];
    output->buf[i] = tanhf(x);
}
/*
 * Elementwise Sigmoid 1 / (1 + exp(-x))
 * input: [*]
 * output: [*] (same shape as input)
 */
void elemwise_sigmoid(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = 1.0 / (1.0 + expf(-x));
  }
}

__global__ void elemwiseSigmoidKernel(Tensor *input, Tensor *output)
{

    // Calculate the index of the element to be processed
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the index is within the range of the input tensor
    float x = input->buf[i];
    output->buf[i] = 1.0 / (1.0 + expf(-x));
}

/*
 * SGEMV
 * input1: [N x K]
 * input2: [K]
 * output: [N]
 */
void matvec(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t N_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  for (size_t i = 0; i < N_; i++) {
    float c = 0.0;
    for (size_t j = 0; j < K_; j++) {
      c += input1->buf[i * K_ + j] * input2->buf[j];
    }
    output->buf[i] = c;
  }
}

/*
 * SGEMM
 * input1: [M x K]
 * input2: [K x N]
 * output: [M x N]
 */
void matmul(Tensor *input1, Tensor *input2, Tensor *output) { // tiled 사용하기
  size_t M_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  size_t N_ = input2->shape[1];

  #pragma omp parallel for
  for (size_t i = 0; i < M_; i++) {
    for (size_t j = 0; j < N_; j++) {
      float c = 0.0;
      for (size_t k = 0; k < K_; k++) {
        c += input1->buf[i * K_ + k] * input2->buf[k * N_ + j];
      }
      output->buf[i * N_ + j] = c;
    }
  }
}

__global__ void matmulKernel(Tensor *input1, Tensor *input2, Tensor *output)
{
    size_t M_ = input1->shape[0];
    size_t K_ = input1->shape[1];
    size_t N_ = input2->shape[1];

    // Calculate the indices of the element to be processed
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the indices are within the range of the input matrices
    if (i < M_ && j < N_)
    {
        float c = 0.0;
        for (size_t k = 0; k < K_; k++)
        {
            c += input1->buf[i * K_ + k] * input2->buf[k * N_ + j];
        }
        output->buf[i * N_ + j] = c;
    }
}
// __global__ void matmul_kernel(float *A, float *B, float *C, const int M, const int K, const int N)
// {
//     __shared__ float Asub[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float Bsub[TILE_WIDTH][TILE_WIDTH];
 
//     int by = blockIdx.x, bx = blockIdx.y;
//     int ty = threadIdx.x, tx = threadIdx.y;
 
//     int Row = by * TILE_WIDTH + ty;
//     int Col = bx * TILE_WIDTH + tx;
 
//     float Pvalue = 0;
//     for (int ph = 0; ph < ceil(K / (float)TILE_WIDTH); ++ph) {
//         Asub[ty][tx] = A[Row*K + ph*TILE_WIDTH + tx];
//         Bsub[ty][tx] = B[(ph*TILE_WIDTH + ty)*K + Col];
 
//         __syncthreads();
 
//         for (int k = 0; k < TILE_WIDTH; k++) {
//             Pvalue += Asub[ty][k] * Bsub[k][tx];
//         }
 
//         __syncthreads();
//     }
//     C[Row*K + Col] = Pvalue;
// }

/*
 * Softmax
 * Normalize the input elements according to its exp value.
 * The result can be interpreted as a probability distribution.
 * input: [*]
 * output: [*], (same shape as input)
 */
void softmax(Tensor *input, Tensor *output, int N) { // softmax(f, char_prob, BATCHSIZE);
  for(int k = 0; k < N; k++){
    size_t n = input->num_elem()/N;
    float sum = 0.0;
    for (size_t i = 0; i < n; i++) {
      float x = input->buf[i*N+k];
      sum += expf(x);
    }
    for (size_t i = 0; i < n; i++) {
      float x = input->buf[i*N+k];
      output->buf[i*N+k] = expf(x) / sum;
    }
  }
}

__global__ void softmaxKernel(Tensor *input, Tensor *output, int N)
{
    // Calculate the index of the block to be processed
    int k = blockIdx.x;

    // Calculate the number of elements in each block
    size_t n = 256;

    // Calculate the sum of the exponentiated values of the elements in the block
    float sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        float x = input->buf[i*N+k];
        sum += expf(x);
    }

    // Calculate the probability of each element in the block
    for (size_t i = 0; i < n; i++) {
        float x = input->buf[i*N+k];
        output->buf[i*N+k] = expf(x) / sum;
    }
}
/*
 * Sample a random index according to the given probability distribution 
 * This function is called at most N*MAX_LEN times. Each call uses a
 * random float in [0,1] to sample an index from the given distribution.
 * input: [NUM_CHAR], probability distribution of the characters
 * rng_seq: [N*MAX_LEN],
 */
void random_select(Tensor *char_prob, Tensor *rng_seq, Tensor *input, char *output, int N, int l) { //char_prob, rfloats, n * MAX_LEN + l // 글씨의 최대 길이. ??
//char_prob, rfloats, input, output, N, l);n * MAX_LEN + l
  int selected_char = 0;
  for(int n = 0; n < BATCHSIZE; n++){ // 64 // 여기 batchsize는 전체 양을 node로 나눈 수
    int rng_offset = (n + (mpi_rank * BATCHSIZE)) * MAX_LEN + l; // 여기에 mpi를 고려한다.
    float r = rng_seq->buf[rng_offset]; //
    size_t n_elem = 256;
    float psum = 0.0;
    for (size_t i = 0; i < n_elem; i++) {
      
      psum += char_prob->buf[i * BATCHSIZE + n];
      
      if (psum > r) {
        selected_char = i;
        break;
      }
    }
    if(psum <= r){
      selected_char = n_elem - 1;
    }
    // printf("psum:%f, r: %f, selected_char: %d\n",psum, r,selected_char);
    // printf("mpirank:%d output index: %d\n",mpi_rank,(n + (mpi_rank * BATCHSIZE)));
    output[(n + (mpi_rank * BATCHSIZE)) * (MAX_LEN + 1) + l] = selected_char; //
    input->buf[n] = selected_char; // 다시 입력으로 넣는다.
  }
}

__global__ void randomSelectKernel(Tensor *char_prob, Tensor *rng_seq, Tensor *input, char *output, int N, int l)
{
    // Calculate the index of the block to be processed
    int n = blockIdx.x;

    // Calculate the offset into the rng_seq tensor
    int rng_offset = (n + (mpi_rank * BATCHSIZE)) * MAX_LEN + l;

    // Retrieve the random value from the rng_seq tensor
    float r = rng_seq->buf[rng_offset];

    // Calculate the number of elements in each block
    size_t n_elem = 256;
    int selected_char = 0;
    // Calculate the sum of the probabilities of the elements in the block
    float psum = 0.0;
    for (size_t i = 0; i < n_elem; i++) {
        psum += char_prob->buf[i * BATCHSIZE + n];
        if (psum > r) {
            selected_char = i;
            break;
        }
    }
    if (psum <= r) {
        selected_char = n_elem - 1;
    }

    // Store the selected element in the output character array and the input tensor
    output[(n + (mpi_rank * BATCHSIZE)) * (MAX_LEN + 1) + l] = selected_char;
    input->buf[n] = selected_char;
}

void stack_vector(int N, Tensor *b_ir0_stack){
  // Tensor* b_ir0_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
  int size = HIDDEN_DIM;
  if(b_ir0_stack->shape[0] == NUM_CHAR){
    size = NUM_CHAR;
  }
  for(int i = 0; i < N; i++){
    for(int j = 0; j < size; j++){
      b_ir0_stack->buf[j * N + i] = b_ir0->buf[j];
    }
  }
}

/*
 * Initialize the model.
 * Do input-independent job here.
 */
void namegen_initialize(int N, int rng_seed, char *parameter_fname) {

  /* Only the root process reads the parameter */ // 모든 파라메터를 가져온다.
  if (1) {
    size_t parameter_binary_size = 0;
    float *parameter =
        (float *)read_binary(parameter_fname, &parameter_binary_size);

    /* Network parameters */
    character_embedding =
        new Tensor({NUM_CHAR, EMBEDDING_DIM}, parameter + OFFSET0); // shape, buf의 시작위치인것같음.

    W_ir0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET1); // shape and buff / 1024, 512
    W_iz0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET2); // 1024, 512
    W_in0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET3); // 1024, 512
    W_ir1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET4); // 1024, 1024
    W_iz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET5); // 1024, 1024
    W_in1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET6); // 1024, 1024

    W_hr0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET7);// 1024, 1024
    W_hz0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET8);// 1024, 1024
    W_hn0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET9);// 1024, 1024
    W_hr1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET10);// 1024, 1024
    W_hz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET11);// 1024, 1024
    W_hn1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET12);// 1024, 1024

    b_ir0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET13);//1024
    b_iz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET14);//1024
    b_in0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET15);//1024
    b_ir1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET16);//1024
    b_iz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET17);//1024
    b_in1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET18);//1024

    b_hr0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET19);//1024
    b_hz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET20);//1024
    b_hn0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET21);//1024
    b_hr1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET22);//1024
    b_hz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET23);//1024
    b_hn1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET24);//1024

    b_ir0_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
    b_iz0_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
    b_in0_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
    b_ir1_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
    b_iz1_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
    b_in1_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
    stack_vector(BATCHSIZE, b_ir0_stack);
    stack_vector(BATCHSIZE, b_iz0_stack);
    stack_vector(BATCHSIZE, b_in0_stack);
    stack_vector(BATCHSIZE, b_ir1_stack);
    stack_vector(BATCHSIZE, b_iz1_stack);
    stack_vector(BATCHSIZE, b_in1_stack);

    b_hr0_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
    b_hz0_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
    b_hn0_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
    b_hr1_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
    b_hz1_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
    b_hn1_stack = new Tensor({HIDDEN_DIM, BATCHSIZE});
    stack_vector(BATCHSIZE, b_hr0_stack);
    stack_vector(BATCHSIZE, b_hz0_stack);
    stack_vector(BATCHSIZE, b_hn0_stack);
    stack_vector(BATCHSIZE, b_hr1_stack);
    stack_vector(BATCHSIZE, b_hz1_stack);
    stack_vector(BATCHSIZE, b_hn1_stack);


    W_fc = new Tensor({NUM_CHAR, HIDDEN_DIM}, parameter + OFFSET25);//256,1024
    b_fc = new Tensor({NUM_CHAR}, parameter + OFFSET26);//256
    b_fc_stack = new Tensor({NUM_CHAR, BATCHSIZE});
    stack_vector(BATCHSIZE, b_fc_stack);

    /* input, activations, output, etc. */
    input = new Tensor({1, BATCHSIZE});//1
    emb_out = new Tensor({EMBEDDING_DIM, BATCHSIZE});//512

    hidden0 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    hidden1 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024

    r0 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    r1 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    z0 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    z1 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    n0 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    n1 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    f = new Tensor({NUM_CHAR, BATCHSIZE});//256
//
    rtmp00 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    rtmp01 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    rtmp02 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    rtmp03 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    rtmp04 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    rtmp10 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    rtmp11 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    rtmp12 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    rtmp13 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    rtmp14 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
//
    ztmp00 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    ztmp01 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    ztmp02 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    ztmp03 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    ztmp04 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024
    ztmp10 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ztmp11 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ztmp12 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ztmp13 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ztmp14 = new Tensor({HIDDEN_DIM, BATCHSIZE});

    ntmp00 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ntmp01 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ntmp02 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ntmp03 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ntmp04 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ntmp05 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ntmp10 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ntmp11 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ntmp12 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ntmp13 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ntmp14 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    ntmp15 = new Tensor({HIDDEN_DIM, BATCHSIZE});
//
    htmp00 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    htmp01 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    htmp02 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    htmp10 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    htmp11 = new Tensor({HIDDEN_DIM, BATCHSIZE});
    htmp12 = new Tensor({HIDDEN_DIM, BATCHSIZE});//1024

    rfloats = new Tensor({N * MAX_LEN});// generate 글자 * 글자 수
    ftmp0 = new Tensor({NUM_CHAR, BATCHSIZE});//256
    char_prob = new Tensor({NUM_CHAR, BATCHSIZE});//256

    // int M_start = (N / mpi_size) * mpi_rank;
    // int M_end = (N / mpi_size) * (mpi_rank + 1);
    // if(mpi_rank == 0){
    //   #pragma omp parallel for num_threads(mpi_world_size - 1)
    //   for(int i = 1; i < mpi_size; i++)
    //   {
    //     // 1,2,3 rank의 start, end값은 모르기때문에 이렇게 처리해준다.
    //     int M_start = i * (N / mpi_size);
    //     int M_end = (i + 1) * (N / mpi_size);
    //     MPI_Send(A + M_start * K, (M_end - M_start) * K, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
    //     MPI_Send(B, K*N, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
    //   }

    // }else{
    //   MPI_Recv((void*)(A + M_start * K), (M_end - M_start) * K, MPI_FLOAT, 0, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    //   MPI_Recv((void*)B, K*N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }

    //cuda 준비


    //
    int size = N / mpi_size;
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
        Mbegin_gpu[i] = (size / num_devices) * i;
        Mend_gpu[i] = Mbegin_gpu[i] + (size / num_devices);
        // printf("Mbegin[%d]: %d, Mend[%d]: %d\n",i,Mbegin[i],i,Mend[i]);
      }
      // Mend[num_devices - 1] = M;

      // Allocate device memory for each GPU
      for (int i = 0; i < num_devices; i++) {
        CUDA_CALL(cudaSetDevice(i));
        ///여기
        CUDA_CALL(cudaMalloc(&character_embedding_gpu[i], character_embedding->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&W_ir0_gpu[i], W_ir0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_iz0_gpu[i], W_iz0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_in0_gpu[i], W_in0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_ir1_gpu[i], W_ir1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_iz1_gpu[i], W_iz1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_in1_gpu[i], W_in1->num_elem() * sizeof(float)));
        
        CUDA_CALL(cudaMalloc(&W_hr0_gpu[i], W_hr0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_hz0_gpu[i], W_hz0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_hn0_gpu[i], W_hn0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_hr1_gpu[i], W_hr1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_hz1_gpu[i], W_hz1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_hn1_gpu[i], W_hn1->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&b_ir0_stack_gpu[i], b_ir0_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_iz0_stack_gpu[i], b_iz0_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_in0_stack_gpu[i], b_in0_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_ir1_stack_gpu[i], b_ir1_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_iz1_stack_gpu[i], b_iz1_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_in1_stack_gpu[i], b_in1_stack->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&b_hr0_stack_gpu[i], b_hr0_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_hz0_stack_gpu[i], b_hz0_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_hn0_stack_gpu[i], b_hn0_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_hr1_stack_gpu[i], b_hr1_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_hz1_stack_gpu[i], b_hz1_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_hn1_stack_gpu[i], b_hn1_stack->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&W_fc_gpu[i], W_fc->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_fc_stack_gpu[i], b_fc_stack->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&input_gpu[i], input->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&emb_out_gpu[i], emb_out->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&hidden0_gpu[i], hidden0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&hidden1_gpu[i], hidden1->num_elem() * sizeof(float)));
//
        CUDA_CALL(cudaMalloc(&r0_gpu[i], r0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&r1_gpu[i], r1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&z0_gpu[i], z0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&z1_gpu[i], z1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&n0_gpu[i], n0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&n1_gpu[i], n1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&f_gpu[i], f->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&rtmp00_gpu[i], rtmp00->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp01_gpu[i], rtmp01->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp02_gpu[i], rtmp02->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp03_gpu[i], rtmp03->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp04_gpu[i], rtmp04->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp10_gpu[i], rtmp10->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp11_gpu[i], rtmp11->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp12_gpu[i], rtmp12->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp13_gpu[i], rtmp13->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp14_gpu[i], rtmp14->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&ztmp00_gpu[i], ztmp00->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp01_gpu[i], ztmp01->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp02_gpu[i], ztmp02->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp03_gpu[i], ztmp03->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp04_gpu[i], ztmp04->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp10_gpu[i], ztmp10->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp11_gpu[i], ztmp11->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp12_gpu[i], ztmp12->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp13_gpu[i], ztmp13->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp14_gpu[i], ztmp14->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&ntmp00_gpu[i], ntmp00->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp01_gpu[i], ntmp01->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp02_gpu[i], ntmp02->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp03_gpu[i], ntmp03->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp04_gpu[i], ntmp04->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp05_gpu[i], ntmp05->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp10_gpu[i], ntmp10->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp11_gpu[i], ntmp11->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp12_gpu[i], ntmp12->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp13_gpu[i], ntmp13->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp14_gpu[i], ntmp14->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp15_gpu[i], ntmp15->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&htmp00_gpu[i], htmp00->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&htmp01_gpu[i], htmp01->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&htmp02_gpu[i], htmp02->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&htmp10_gpu[i], htmp10->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&htmp11_gpu[i], htmp11->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&htmp12_gpu[i], htmp12->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&rfloats_gpu[i], rfloats->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ftmp0_gpu[i], ftmp0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&char_prob_gpu[i], char_prob->num_elem() * sizeof(float)));
        // CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * BATCHSIZE * sizeof(float))); // 이렇게 1024로 해도 되나?
        // CUDA_CALL(cudaMalloc(&b_d[i], BATCHSIZE * N * sizeof(float)));
        // CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
      }
    }else{
      CUDA_CALL(cudaGetDeviceCount(&num_devices));

      for (int i = 0; i < num_devices; i++) {
        cudaDeviceProp prop;
        CUDA_CALL(cudaGetDeviceProperties(&prop, i));
      }

      // Setup problem size for each GPU
      for (int i = 0; i < num_devices; i++) {
        Mbegin_gpu[i] = (size / num_devices) * i;
        Mend_gpu[i] = Mbegin_gpu[i] + (size / num_devices);
        // printf("Mbegin[%d]: %d, Mend[%d]: %d\n",i,Mbegin[i],i,Mend[i]);
      }

      for (int i = 0; i < num_devices; i++) {
        CUDA_CALL(cudaSetDevice(i));
        CUDA_CALL(cudaMalloc(&character_embedding_gpu[i], character_embedding->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_ir0_gpu[i], W_ir0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_ir0_gpu[i], W_ir0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_iz0_gpu[i], W_iz0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_in0_gpu[i], W_in0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_ir1_gpu[i], W_ir1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_iz1_gpu[i], W_iz1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_in1_gpu[i], W_in1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_hr0_gpu[i], W_hr0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_hz0_gpu[i], W_hz0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_hn0_gpu[i], W_hn0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_hr1_gpu[i], W_hr1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_hz1_gpu[i], W_hz1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&W_hn1_gpu[i], W_hn1->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&b_ir0_stack_gpu[i], b_ir0_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_iz0_stack_gpu[i], b_iz0_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_in0_stack_gpu[i], b_in0_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_ir1_stack_gpu[i], b_ir1_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_iz1_stack_gpu[i], b_iz1_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_in1_stack_gpu[i], b_in1_stack->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&b_hr0_stack_gpu[i], b_hr0_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_hz0_stack_gpu[i], b_hz0_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_hn0_stack_gpu[i], b_hn0_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_hr1_stack_gpu[i], b_hr1_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_hz1_stack_gpu[i], b_hz1_stack->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_hn1_stack_gpu[i], b_hn1_stack->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&W_fc_gpu[i], W_fc->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b_fc_stack_gpu[i], b_fc_stack->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&input_gpu[i], input->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&emb_out_gpu[i], emb_out->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&hidden0_gpu[i], hidden0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&hidden1_gpu[i], hidden1->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&r0_gpu[i], r0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&r1_gpu[i], r1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&z0_gpu[i], z0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&z1_gpu[i], z1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&n0_gpu[i], n0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&n1_gpu[i], n1->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&f_gpu[i], f->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&rtmp00_gpu[i], rtmp00->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp01_gpu[i], rtmp01->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp02_gpu[i], rtmp02->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp03_gpu[i], rtmp03->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp04_gpu[i], rtmp04->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp10_gpu[i], rtmp10->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp11_gpu[i], rtmp11->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp12_gpu[i], rtmp12->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp13_gpu[i], rtmp13->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&rtmp14_gpu[i], rtmp14->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&ztmp00_gpu[i], ztmp00->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp01_gpu[i], ztmp01->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp02_gpu[i], ztmp02->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp03_gpu[i], ztmp03->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp04_gpu[i], ztmp04->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp10_gpu[i], ztmp10->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp11_gpu[i], ztmp11->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp12_gpu[i], ztmp12->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp13_gpu[i], ztmp13->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ztmp14_gpu[i], ztmp14->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&ntmp00_gpu[i], ntmp00->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp01_gpu[i], ntmp01->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp02_gpu[i], ntmp02->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp03_gpu[i], ntmp03->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp04_gpu[i], ntmp04->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp05_gpu[i], ntmp05->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp10_gpu[i], ntmp10->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp11_gpu[i], ntmp11->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp12_gpu[i], ntmp12->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp13_gpu[i], ntmp13->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp14_gpu[i], ntmp14->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ntmp15_gpu[i], ntmp15->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&htmp00_gpu[i], htmp00->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&htmp01_gpu[i], htmp01->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&htmp02_gpu[i], htmp02->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&htmp10_gpu[i], htmp10->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&htmp11_gpu[i], htmp11->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&htmp12_gpu[i], htmp12->num_elem() * sizeof(float)));

        CUDA_CALL(cudaMalloc(&rfloats_gpu[i], rfloats->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&ftmp0_gpu[i], ftmp0->num_elem() * sizeof(float)));
        CUDA_CALL(cudaMalloc(&char_prob_gpu[i], char_prob->num_elem() * sizeof(float)));
        // CUDA_CALL(cudaMalloc(&a_d_gpu[i], (Mend_gpu[i] - Mbegin_gpu[i]) * BATCHSIZE * sizeof(float)));
        // CUDA_CALL(cudaMalloc(&b_d_gpu[i], BATCHSIZE * N * sizeof(float)));
        // CUDA_CALL(cudaMalloc(&c_d_gpu[i], (Mend_gpu[i] - Mbegin_gpu[i]) * N * sizeof(float)));

      }
    }
  }
}

/*
 * Generate names.
 * Any input-dependent computation/communication must be done here.
 * N: # of names to generate
 * random_floats: N*MAX_LEN sequence of random floats in [0,1].
 * output: 2D-array of size N x (MAX_LEN+1), allocaetd at main.cpp
 */
void namegen(int N, float *random_floats, char *output) { // 병렬화 대상

  // /* Only root process does the job, for now... */
  // if (mpi_rank != 0)
  //   return;
  int size = N / mpi_size;
  int M_start = (size) * mpi_rank;
  int M_end = (size) * (mpi_rank + 1);

// char_prob, rfloats, input, output, N, l)
  if (mpi_rank == 0){
    memcpy(rfloats->buf, random_floats, N * MAX_LEN * sizeof(float)); // buf = pointer to data, 메모리에 있는 random_floats를 buf로 옮긴다.
    memset(output, 0, N * (MAX_LEN + 1) * sizeof(char)); // 메모리 시작점부터 특정 범위까지 특정 값으로 지정할 수 있다.
    // input->set_SOS(); // start of sequence

    #pragma omp parallel for num_threads(mpi_size - 1)
    for(int i = 1; i < mpi_size; i++)
    {
      // int M_start = i * (N / mpi_size);
      // int M_end = (i + 1) * (N / mpi_size);
      MPI_Send(output + M_start* (MAX_LEN + 1), (M_end - M_start)* (MAX_LEN + 1), MPI_CHARACTER, i, 1, MPI_COMM_WORLD);
      MPI_Send(random_floats, N * MAX_LEN, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
    }

    /* Initialize input and hidden vector. */
    /* One hidden vector for each GRU layer */
    input->set_SOS();
    hidden0->set_zero(); // hidden0,1을 0으로 지정한다.
    hidden1->set_zero();
    for (int i = 0; i < num_devices; i++) {
      // printf("mpi_rank[%d]/device[%d]: start_address:%d, amount:%d\n",mpi_rank, i, (Mbegin[i] + M_start), (Mend[i] - Mbegin[i]));
      // CUDA_CALL(cudaMemcpy(a_d[i], A + (Mbegin[i] + M_start) * K,
      //                     (Mend[i] - Mbegin[i]) * K * sizeof(float),
      //                     cudaMemcpyHostToDevice));
      // CUDA_CALL(
      //     cudaMemcpy(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(character_embedding_gpu[i], character_embedding->buf, character_embedding->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CALL(cudaMemcpy(W_ir0_gpu[i], W_ir0->buf, W_ir0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_ir1_gpu[i], W_ir1->buf, W_ir1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_iz0_gpu[i], W_iz0->buf, W_iz0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_iz1_gpu[i], W_iz1->buf, W_iz1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_in0_gpu[i], W_in0->buf, W_in0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_in1_gpu[i], W_in1->buf, W_in1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CALL(cudaMemcpy(W_hr0_gpu[i], W_hr0->buf, W_hr0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_hr1_gpu[i], W_hr1->buf, W_hr1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_hz0_gpu[i], W_hz0->buf, W_hz0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_hz1_gpu[i], W_hz1->buf, W_hz1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_hn0_gpu[i], W_hn0->buf, W_hn0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_hn1_gpu[i], W_hn1->buf, W_hn1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(b_ir0_stack_gpu[i], b_ir0_stack->buf, b_ir0_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_iz0_stack_gpu[i], b_iz0_stack->buf, b_iz0_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_in0_stack_gpu[i], b_in0_stack->buf, b_in0_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_ir1_stack_gpu[i], b_ir1_stack->buf, b_ir1_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_iz1_stack_gpu[i], b_iz1_stack->buf, b_iz1_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_in1_stack_gpu[i], b_in1_stack->buf, b_in1_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CALL(cudaMemcpy(b_hr0_stack_gpu[i], b_hr0_stack->buf, b_hr0_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_hz0_stack_gpu[i], b_hz0_stack->buf, b_hz0_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_hn0_stack_gpu[i], b_hn0_stack->buf, b_hn0_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_hr1_stack_gpu[i], b_hr1_stack->buf, b_hr1_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_hz1_stack_gpu[i], b_hz1_stack->buf, b_hz1_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_hn1_stack_gpu[i], b_hn1_stack->buf, b_hn1_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(W_fc_gpu[i], W_fc->buf, W_fc->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_fc_stack_gpu[i], b_fc_stack->buf, b_fc_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(input_gpu[i], input->buf, input->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(emb_out_gpu[i], emb_out->buf, emb_out->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(hidden0_gpu[i], hidden0->buf, hidden0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(hidden1_gpu[i], hidden1->buf, hidden1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(r0_gpu[i], r0->buf, r0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(r1_gpu[i], r1->buf, r1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(z0_gpu[i], z0->buf, z0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(z1_gpu[i], z1->buf, z1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(n0_gpu[i], n0->buf, n0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(n1_gpu[i], n1->buf, n1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(f_gpu[i], f->buf, f->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(rtmp00_gpu[i], rtmp00->buf, rtmp00->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp01_gpu[i], rtmp01->buf, rtmp01->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp02_gpu[i], rtmp02->buf, rtmp02->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp03_gpu[i], rtmp03->buf, rtmp03->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp04_gpu[i], rtmp04->buf, rtmp04->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp10_gpu[i], rtmp10->buf, rtmp10->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp11_gpu[i], rtmp11->buf, rtmp11->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp12_gpu[i], rtmp12->buf, rtmp12->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp13_gpu[i], rtmp13->buf, rtmp13->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp14_gpu[i], rtmp14->buf, rtmp14->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      //
      CUDA_CALL(cudaMemcpy(ztmp00_gpu[i], ztmp00->buf, ztmp00->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp01_gpu[i], ztmp01->buf, ztmp01->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp02_gpu[i], ztmp02->buf, ztmp02->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp03_gpu[i], ztmp03->buf, ztmp03->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp04_gpu[i], ztmp04->buf, ztmp04->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp10_gpu[i], ztmp10->buf, ztmp10->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp11_gpu[i], ztmp11->buf, ztmp11->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp12_gpu[i], ztmp12->buf, ztmp12->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp13_gpu[i], ztmp13->buf, ztmp13->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp14_gpu[i], ztmp14->buf, ztmp14->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(ntmp00_gpu[i], ntmp00->buf, ntmp00->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp01_gpu[i], ntmp01->buf, ntmp01->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp02_gpu[i], ntmp02->buf, ntmp02->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp03_gpu[i], ntmp03->buf, ntmp03->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp04_gpu[i], ntmp04->buf, ntmp04->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp05_gpu[i], ntmp05->buf, ntmp05->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp10_gpu[i], ntmp10->buf, ntmp10->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp11_gpu[i], ntmp11->buf, ntmp11->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp12_gpu[i], ntmp12->buf, ntmp12->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp13_gpu[i], ntmp13->buf, ntmp13->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp14_gpu[i], ntmp14->buf, ntmp14->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp15_gpu[i], ntmp15->buf, ntmp15->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(htmp00_gpu[i], htmp00->buf, htmp00->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(htmp01_gpu[i], htmp01->buf, htmp01->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(htmp02_gpu[i], htmp02->buf, htmp02->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(htmp10_gpu[i], htmp10->buf, htmp10->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(htmp11_gpu[i], htmp11->buf, htmp11->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(htmp12_gpu[i], htmp12->buf, htmp12->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(rfloats_gpu[i], rfloats->buf, rfloats->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ftmp0_gpu[i], ftmp0->buf, ftmp0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(char_prob_gpu[i], char_prob->buf, char_prob->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
    }
    //   /* Initialize input and hidden vector. */
    // /* One hidden vector for each GRU layer */
    // input->set_SOS();
    // hidden0->set_zero(); // hidden0,1을 0으로 지정한다.
    // hidden1->set_zero();

    for (int l = 0; l < MAX_LEN; l++) { // 이름의 최대 길이는 10이다.
      /* Embedding */
      // embedding(input, character_embedding, emb_out); // 여기서 input은 buf[0]만을 가지고 있다. 만약 사이즈는 512다.
          // Launch kernel on every GPU
      for (int i = 0; i < num_devices; i++) {
        CUDA_CALL(cudaSetDevice(i));
        embedding_kernel<<<BATCHSIZE, 1>>>(input_gpu[i], character_embedding_gpu[i], emb_out_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      // double namegen_st = get_time();
      // // /* First layer r */ // GRU연산을 보면 r, z, n을 구하고 이를 통해서 h를 구한다.
      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(HIDDEN_DIM, BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_ir0_gpu[i], emb_out_gpu[i], rtmp00_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_hr0_gpu[i], hidden0_gpu[i], rtmp01_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(rtmp00_gpu[i], b_ir0_stack_gpu[i], rtmp02_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp02_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(rtmp02_gpu[i], rtmp01_gpu[i], rtmp03_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(rtmp03_gpu[i], b_hr0_stack_gpu[i], rtmp04_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseSigmoidKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(rtmp04_gpu[i], r0_gpu[i]); //(Mend[i] - Mbegin[i])
      }
    //   matmul(W_ir0, emb_out, rtmp00); // weight_input * embedding / NK * K -> N (matrix mector multiplication)
    //   matmul(W_hr0, hidden0, rtmp01); // weight_hidden * hidden / 
      // elemwise_add(rtmp00, b_ir0_stack, rtmp02); // weight_input * embedding + bias_input / 
    //   elemwise_add(rtmp02, rtmp01, rtmp03); //weight_input * embedding + weight_hidden * hidden + bias_input
    //   elemwise_add(rtmp03, b_hr0_stack, rtmp04); // weight_input * embedding + weight_hidden * hidden + bias_input + bias_hidden  
    //   elemwise_sigmoid(rtmp04, r0); // activation(weight_input * embedding + weight_hidden * hidden + bias_input + bias_hidden) -> r0

      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(HIDDEN_DIM, BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_iz0_gpu[i], emb_out_gpu[i], ztmp00_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_hz0_gpu[i], hidden0_gpu[i], ztmp01_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ztmp00_gpu[i], b_iz0_stack_gpu[i], ztmp02_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp02_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ztmp02_gpu[i], ztmp01_gpu[i], ztmp03_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ztmp03_gpu[i], b_hz0_stack_gpu[i], ztmp04_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseSigmoidKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ztmp04_gpu[i], z0_gpu[i]); //(Mend[i] - Mbegin[i])
      }
    //   // namegen_st = get_time();
    //   // /* First layer z */
    //   matmul(W_iz0, emb_out, ztmp00);
    //   matmul(W_hz0, hidden0, ztmp01);
    //   elemwise_add(ztmp00, b_iz0_stack, ztmp02);
    //   elemwise_add(ztmp02, ztmp01, ztmp03);
    //   elemwise_add(ztmp03, b_hz0_stack, ztmp04);
    //   elemwise_sigmoid(ztmp04, z0);
    //   // namegen_en = get_time();
    //   // if (mpi_rank == 0) {
    //   //   double elapsed_time = namegen_en - namegen_st;
    //   //   printf("Elapsed time for first layer z: %.6f seconds\n", elapsed_time);
    //   // }
      
    //   // namegen_st = get_time();
    //   // /* First layer n */
      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_in0_gpu[i], emb_out_gpu[i], ntmp00_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ntmp00_gpu[i], b_in0_stack_gpu[i], ntmp01_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_hn0_gpu[i], hidden0_gpu[i], ntmp02_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ntmp02_gpu[i], b_hn0_stack_gpu[i], ntmp03_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        // dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseMulKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(r0_gpu[i], ntmp03_gpu[i], ntmp04_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ntmp01_gpu[i], ntmp04_gpu[i], ntmp05_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseTanhKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ntmp05_gpu[i], n0_gpu[i]); //(Mend[i] - Mbegin[i])
      }
    //   matmul(W_in0, emb_out, ntmp00);
    //   elemwise_add(ntmp00, b_in0_stack, ntmp01);
    //   matmul(W_hn0, hidden0, ntmp02);
    //   elemwise_add(ntmp02, b_hn0_stack, ntmp03);
      // elemwise_mul(r0, ntmp03, ntmp04);
    //   elemwise_add(ntmp01, ntmp04, ntmp05);
      // elemwise_tanh(ntmp05, n0);
    //   // //shape
    //   // printf("\nshape of emb_out: <%zd, %zd>\n", n0->shape[0], n0->shape[1]);
    //   // //value
    //   // for(int i = 5; i < 10; i++){
    //   //   for(int j = 0; j < 2; j++){
    //   //     printf("%d번째 word %d번: %f\n",i, j+1, n0->buf[j * N + i]);
    //   //   }
    //   // }      
    //   // namegen_en = get_time();
    //   // if (mpi_rank == 0) {
    //   //   double elapsed_time = namegen_en - namegen_st;
    //   //   printf("Elapsed time for first layer n: %.6f seconds\n", elapsed_time);
    //   // }

    //   // namegen_st = get_time();
    //   // /* First layer h (hidden) */
      for (int i = 0; i < num_devices; i++) {
        // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        // dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseOneminusKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(z0_gpu[i], htmp00_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        // dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseMulKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(htmp00_gpu[i], n0_gpu[i], htmp01_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        // dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseMulKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(z0_gpu[i], hidden0_gpu[i], htmp02_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(htmp01_gpu[i], htmp02_gpu[i], hidden0_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      // elemwise_oneminus(z0, htmp00); // 1 - z0
    //   elemwise_mul(htmp00, n0, htmp01); // vector * vector 
    //   elemwise_mul(z0, hidden0, htmp02);
    //   elemwise_add(htmp01, htmp02, hidden0);
    //   // //shape
    //   // printf("\nshape of emb_out: <%zd, %zd>\n", hidden0->shape[0], hidden0->shape[1]);
    //   // //value
    //   // for(int i = 5; i < 10; i++){
    //   //   for(int j = 0; j < 2; j++){
    //   //     printf("%d번째 word %d번: %f\n",i, j+1, hidden0->buf[j * N + i]);
    //   //   }
    //   // }     
    //   // namegen_en = get_time();
    //   // if (mpi_rank == 0) {
    //   //   double elapsed_time = namegen_en - namegen_st;
    //   //   printf("Elapsed time for first layer h: %.6f seconds\n", elapsed_time);
    //   // }

    //   // namegen_st = get_time();
    //   // /* Second layer r */ // GRU2
      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(HIDDEN_DIM, BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_ir1_gpu[i], hidden0_gpu[i], rtmp10_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_hr1_gpu[i], hidden1_gpu[i], rtmp11_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(rtmp10_gpu[i], b_ir1_stack_gpu[i], rtmp12_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp02_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(rtmp12_gpu[i], rtmp11_gpu[i], rtmp13_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(rtmp13_gpu[i], b_hr1_stack_gpu[i], rtmp14_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseSigmoidKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(rtmp14_gpu[i], r1_gpu[i]); //(Mend[i] - Mbegin[i])
      }    
    //   matmul(W_ir1, hidden0, rtmp10);
    //   matmul(W_hr1, hidden1, rtmp11);
    //   elemwise_add(rtmp10, b_ir1_stack, rtmp12);
    //   elemwise_add(rtmp12, rtmp11, rtmp13);
    //   elemwise_add(rtmp13, b_hr1_stack, rtmp14);
    //   elemwise_sigmoid(rtmp14, r1);
    //   // //shape
    //   // printf("\nshape of emb_out: <%zd, %zd>\n", r1->shape[0], r1->shape[1]);
    //   // //value
    //   // for(int i = 5; i < 10; i++){
    //   //   for(int j = 0; j < 2; j++){
    //   //     printf("%d번째 word %d번: %f\n",i, j+1, r1->buf[j * N + i]);
    //   //   }
    //   // } 
    //   // namegen_en = get_time();
    //   // if (mpi_rank == 0) {
    //   //   double elapsed_time = namegen_en - namegen_st;
    //   //   printf("Elapsed time for second layer r: %.6f seconds\n", elapsed_time);
    //   // }

    //   // namegen_st = get_time();
    //   // /* Second layer z */
      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(HIDDEN_DIM, BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_iz1_gpu[i], hidden0_gpu[i], ztmp10_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_hz1_gpu[i], hidden1_gpu[i], ztmp11_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ztmp10_gpu[i], b_iz1_stack_gpu[i], ztmp12_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp02_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ztmp12_gpu[i], ztmp11_gpu[i], ztmp13_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ztmp13_gpu[i], b_hz1_stack_gpu[i], ztmp14_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseSigmoidKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ztmp14_gpu[i], z1_gpu[i]); //(Mend[i] - Mbegin[i])
      }
    //   matmul(W_iz1, hidden0, ztmp10);
    //   matmul(W_hz1, hidden1, ztmp11);
    //   elemwise_add(ztmp10, b_iz1_stack, ztmp12);
    //   elemwise_add(ztmp12, ztmp11, ztmp13);
    //   elemwise_add(ztmp13, b_hz1_stack, ztmp14);
    //   elemwise_sigmoid(ztmp14, z1);
    //   // namegen_en = get_time();
    //   // if (mpi_rank == 0) {
    //   //   double elapsed_time = namegen_en - namegen_st;
    //   //   printf("Elapsed time for second layer z: %.6f seconds\n", elapsed_time);
    //   // }

    //   // namegen_st = get_time();
    //   // /* Second layer n */
      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_in1_gpu[i], hidden0_gpu[i], ntmp10_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ntmp10_gpu[i], b_in1_stack_gpu[i], ntmp11_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_hn1_gpu[i], hidden1_gpu[i], ntmp12_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ntmp12_gpu[i], b_hn1_stack_gpu[i], ntmp13_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        // dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseMulKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(r1_gpu[i], ntmp13_gpu[i], ntmp14_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ntmp11_gpu[i], ntmp14_gpu[i], ntmp15_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseTanhKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(ntmp15_gpu[i], n1_gpu[i]); //(Mend[i] - Mbegin[i])
      }    
      // matmul(W_in1, hidden0, ntmp10);
    //   elemwise_add(ntmp10, b_in1_stack, ntmp11);
    //   matmul(W_hn1, hidden1, ntmp12);
    //   elemwise_add(ntmp12, b_hn1_stack, ntmp13);
    //   elemwise_mul(r1, ntmp13, ntmp14);
    //   elemwise_add(ntmp11, ntmp14, ntmp15);
    //   elemwise_tanh(ntmp15, n1);
    //   // //shape
    //   // printf("\nshape of emb_out: <%zd, %zd>\n", n1->shape[0], n1->shape[1]);
    //   // //value
    //   // for(int i = 5; i < 10; i++){
    //   //   for(int j = 0; j < 2; j++){
    //   //     printf("%d번째 word %d번: %f\n",i, j+1, n1->buf[j * N + i]);
    //   //   }
    //   // }
    //   // namegen_en = get_time();
    //   // if (mpi_rank == 0) {
    //   //   double elapsed_time = namegen_en - namegen_st;
    //   //   printf("Elapsed time for second layer n: %.6f seconds\n", elapsed_time);
    //   // }

    //   // namegen_st = get_time();
    //   // /* Second layer h (hidden) */
      for (int i = 0; i < num_devices; i++) {
        // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        // dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseOneminusKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(z1_gpu[i], htmp10_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        // dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseMulKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(htmp10_gpu[i], n1_gpu[i], htmp11_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        // dim3 gridDim(HIDDEN_DIM,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseMulKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(z1_gpu[i], hidden1_gpu[i], htmp12_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<HIDDEN_DIM*BATCHSIZE, 1>>>(htmp11_gpu[i], htmp12_gpu[i], hidden1_gpu[i]); //(Mend[i] - Mbegin[i])
      }    
    //   elemwise_oneminus(z1, htmp10);
    //   elemwise_mul(htmp10, n1, htmp11);
    //   elemwise_mul(z1, hidden1, htmp12);
    //   elemwise_add(htmp11, htmp12, hidden1);
    //   // //shape
    //   // printf("\nshape of emb_out: <%zd, %zd>\n", hidden1->shape[0], hidden1->shape[1]);
    //   // //value
    //   // for(int i = 5; i < 10; i++){
    //   //   for(int j = 0; j < 2; j++){
    //   //     printf("%d번째 word %d번: %f\n",i, j+1, hidden1->buf[j * N + i]);
    //   //   }
    //   // }
    //   // namegen_en = get_time();
    //   // if (mpi_rank == 0) {
    //   //   double elapsed_time = namegen_en - namegen_st;
    //   //   printf("Elapsed time for second layer h: %.6f seconds\n", elapsed_time);
    //   // }

    //   // namegen_st = get_time();
    //   // /* Fully connected layer */ // linear
      for (int i = 0; i < num_devices; i++) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(NUM_CHAR,BATCHSIZE);
        CUDA_CALL(cudaSetDevice(i));
        matmulKernel<<<gridDim, blockDim>>>(W_fc_gpu[i], hidden1_gpu[i], ftmp0_gpu[i]); //(Mend[i] - Mbegin[i])
      }
      for (int i = 0; i < num_devices; i++) {
        // size_t sn = rtmp00_gpu[i]->num_elem();
        // dim3 blockDim(make256);
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        elemwiseAddKernel<<<NUM_CHAR*BATCHSIZE, 1>>>(ftmp0_gpu[i], b_fc_stack_gpu[i], f_gpu[i]); //(Mend[i] - Mbegin[i])
      }    
    //   matmul(W_fc, hidden1, ftmp0);
    //   elemwise_add(ftmp0, b_fc_stack, f);

    //   // //shape
    //   // printf("\nshape of emb_out: <%zd, %zd>\n", f->shape[0], f->shape[1]);
    //   // //value
    //   // for(int i = 0; i < 1; i++){
    //   //   for(int j = 0; j < 1024; j++){
    //   //     printf("%d번째 word %d번: %f\n",i, j+1, hidden1->buf[j * N + i]);
    //   //   }
    //   // }    
    //   // namegen_en = get_time();
    //   // if (mpi_rank == 0) {
    //   //   double elapsed_time = namegen_en - namegen_st;
    //   //   printf("Elapsed time for fully connected layer: %.6f seconds\n", elapsed_time);
    //   // }

    // //   // namegen_st = get_time();
    // //   // /* Softmax */
      // softmax(f, char_prob, BATCHSIZE);
      for (int i = 0; i < num_devices; i++) {
        int numBlocks = N;
        int threadsPerBlock = 256;
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        softmaxKernel<<<numBlocks, threadsPerBlock>>>(f_gpu[i], char_prob_gpu[i], BATCHSIZE); //(Mend[i] - Mbegin[i])
      }    

    // //   // //shape
    // //   // printf("\nshape of emb_out: <%zd, %zd>\n", char_prob->shape[0], char_prob->shape[1]);
    // //   // //value
    // //   // for(int i = 5; i < 8; i++){
    // //   //   for(int j = 0; j < 10; j++){
    // //   //     printf("%d번째 word %d번: %f\n",i, j+1, char_prob->buf[j * N + i]);
    // //   //   }
    // //   // } 
    // // //   // namegen_en = get_time();
    // // //   // if (mpi_rank == 0) {
    // // //   //   double elapsed_time = namegen_en - namegen_st;
    // // //   //   printf("Elapsed time for softmax: %.6f seconds\n", elapsed_time);
    // // //   // }

    // //   // namegen_st = get_time();
    // //   // /* Random select */
      random_select(char_prob, rfloats, input, output, N, l); // 아 씨발
      for (int i = 0; i < num_devices; i++) {
        int numBlocks = N;
        int threadsPerBlock = 256;
        // dim3 gridDim((sn + blockDim.x - 1) / blockDim.x);
        CUDA_CALL(cudaSetDevice(i));
        randomSelectKernel<<<numBlocks, threadsPerBlock>>>(char_prob_gpu[i], rfloats_gpu[i], input_gpu[i], output, N, l); //(Mend[i] - Mbegin[i])
      }    
    //   // //shape
    //   // printf("\nshape of emb_out: <%zd, %zd>\n", char_prob->shape[0], char_prob->shape[1]);
    //   // //value
    //   // for(int i = 0; i < 1; i++){
    //   //   for(int j = 0; j < 256; j++){
    //   //     printf("%d번째 word %d번: %f\n",i, j+1, char_prob->buf[j * BATCHSIZE + i]);
    //   //   }
    //   // }
    //   // break;
    //   // namegen_en = get_time();
    //   // if (mpi_rank == 0) {
    //   //   double elapsed_time = namegen_en - namegen_st;
    //   //   printf("Elapsed time for random select: %.6f seconds\n", elapsed_time);
    //   // }
    }
    
    for(int i = 1; i < mpi_size; i++)
    {
      int M_start = i * (N / mpi_size);
      int M_end = (i + 1) * (N / mpi_size);
      // MPI_Recv(output + M_start * (MAX_LEN + 1),(M_end - M_start) * (MAX_LEN + 1), MPI_CHAR, i,i , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // printf("[mpi_rank:%d] recv address:%d\n",mpi_rank, M_start);
      // printf("[mpi_rank:%d]amount of receieved message: %d\n", mpi_rank,(M_end - M_start));
      MPI_Recv((void*)(output + M_start * (MAX_LEN + 1)),  (M_end - M_start) * (MAX_LEN + 1), MPI_CHAR, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  else if(mpi_rank != 0){
    char* output = (char *)malloc(N * (MAX_LEN + 1) * sizeof(char));
    float* random_floats_buff = (float *)malloc(N * (MAX_LEN) * sizeof(float));
    MPI_Recv((void*)(output + M_start * (MAX_LEN + 1)),  (M_end - M_start) * (MAX_LEN + 1), MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // MPI_Recv((void*)(random_floats_buff),  N * MAX_LEN, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // MPI_Recv((void*)(input->buf + M_start),  (M_end - M_start), MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv((void*)(random_floats_buff),  N * MAX_LEN, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    Tensor* rfloats = new Tensor({N * MAX_LEN});
    memcpy(rfloats->buf, random_floats_buff, N * MAX_LEN * sizeof(float));
    // memset(output, 0, N * (MAX_LEN + 1) * sizeof(char));
    /* Initialize input and hidden vector. */
    /* One hidden vector for each GRU layer */
    input->set_SOS();
    hidden0->set_zero(); // hidden0,1을 0으로 지정한다.
    hidden1->set_zero();

    for (int i = 0; i < num_devices; i++) {
      // printf("mpi_rank[%d]/device[%d]: start_address:%d, amount:%d\n",mpi_rank, i, (Mbegin[i] + M_start), (Mend[i] - Mbegin[i]));
      // CUDA_CALL(cudaMemcpy(a_d[i], A + (Mbegin[i] + M_start) * K,
      //                     (Mend[i] - Mbegin[i]) * K * sizeof(float),
      //                     cudaMemcpyHostToDevice));
      // CUDA_CALL(
      //     cudaMemcpy(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(character_embedding_gpu[i], character_embedding->buf, character_embedding->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CALL(cudaMemcpy(W_ir0_gpu[i], W_ir0->buf, W_ir0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_ir1_gpu[i], W_ir1->buf, W_ir1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_iz0_gpu[i], W_iz0->buf, W_iz0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_iz1_gpu[i], W_iz1->buf, W_iz1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_in0_gpu[i], W_in0->buf, W_in0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_in1_gpu[i], W_in1->buf, W_in1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CALL(cudaMemcpy(W_hr0_gpu[i], W_hr0->buf, W_hr0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_hr1_gpu[i], W_hr1->buf, W_hr1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_hz0_gpu[i], W_hz0->buf, W_hz0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_hz1_gpu[i], W_hz1->buf, W_hz1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_hn0_gpu[i], W_hn0->buf, W_hn0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(W_hn1_gpu[i], W_hn1->buf, W_hn1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(b_ir0_stack_gpu[i], b_ir0_stack->buf, b_ir0_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_iz0_stack_gpu[i], b_iz0_stack->buf, b_iz0_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_in0_stack_gpu[i], b_in0_stack->buf, b_in0_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_ir1_stack_gpu[i], b_ir1_stack->buf, b_ir1_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_iz1_stack_gpu[i], b_iz1_stack->buf, b_iz1_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_in1_stack_gpu[i], b_in1_stack->buf, b_in1_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CALL(cudaMemcpy(b_hr0_stack_gpu[i], b_hr0_stack->buf, b_hr0_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_hz0_stack_gpu[i], b_hz0_stack->buf, b_hz0_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_hn0_stack_gpu[i], b_hn0_stack->buf, b_hn0_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_hr1_stack_gpu[i], b_hr1_stack->buf, b_hr1_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_hz1_stack_gpu[i], b_hz1_stack->buf, b_hz1_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_hn1_stack_gpu[i], b_hn1_stack->buf, b_hn1_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(W_fc_gpu[i], W_fc->buf, W_fc->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(b_fc_stack_gpu[i], b_fc_stack->buf, b_fc_stack->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(input_gpu[i], input->buf, input->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(emb_out_gpu[i], emb_out->buf, emb_out->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(hidden0_gpu[i], hidden0->buf, hidden0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(hidden1_gpu[i], hidden1->buf, hidden1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(r0_gpu[i], r0->buf, r0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(r1_gpu[i], r1->buf, r1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(z0_gpu[i], z0->buf, z0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(z1_gpu[i], z1->buf, z1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(n0_gpu[i], n0->buf, n0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(n1_gpu[i], n1->buf, n1->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(f_gpu[i], f->buf, f->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(rtmp00_gpu[i], rtmp00->buf, rtmp00->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp01_gpu[i], rtmp01->buf, rtmp01->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp02_gpu[i], rtmp02->buf, rtmp02->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp03_gpu[i], rtmp03->buf, rtmp03->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp04_gpu[i], rtmp04->buf, rtmp04->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp10_gpu[i], rtmp10->buf, rtmp10->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp11_gpu[i], rtmp11->buf, rtmp11->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp12_gpu[i], rtmp12->buf, rtmp12->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp13_gpu[i], rtmp13->buf, rtmp13->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(rtmp14_gpu[i], rtmp14->buf, rtmp14->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      //
      CUDA_CALL(cudaMemcpy(ztmp00_gpu[i], ztmp00->buf, ztmp00->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp01_gpu[i], ztmp01->buf, ztmp01->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp02_gpu[i], ztmp02->buf, ztmp02->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp03_gpu[i], ztmp03->buf, ztmp03->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp04_gpu[i], ztmp04->buf, ztmp04->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp10_gpu[i], ztmp10->buf, ztmp10->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp11_gpu[i], ztmp11->buf, ztmp11->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp12_gpu[i], ztmp12->buf, ztmp12->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp13_gpu[i], ztmp13->buf, ztmp13->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ztmp14_gpu[i], ztmp14->buf, ztmp14->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(ntmp00_gpu[i], ntmp00->buf, ntmp00->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp01_gpu[i], ntmp01->buf, ntmp01->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp02_gpu[i], ntmp02->buf, ntmp02->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp03_gpu[i], ntmp03->buf, ntmp03->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp04_gpu[i], ntmp04->buf, ntmp04->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp05_gpu[i], ntmp05->buf, ntmp05->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp10_gpu[i], ntmp10->buf, ntmp10->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp11_gpu[i], ntmp11->buf, ntmp11->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp12_gpu[i], ntmp12->buf, ntmp12->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp13_gpu[i], ntmp13->buf, ntmp13->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp14_gpu[i], ntmp14->buf, ntmp14->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ntmp15_gpu[i], ntmp15->buf, ntmp15->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(htmp00_gpu[i], htmp00->buf, htmp00->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(htmp01_gpu[i], htmp01->buf, htmp01->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(htmp02_gpu[i], htmp02->buf, htmp02->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(htmp10_gpu[i], htmp10->buf, htmp10->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(htmp11_gpu[i], htmp11->buf, htmp11->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(htmp12_gpu[i], htmp12->buf, htmp12->num_elem() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy(rfloats_gpu[i], rfloats->buf, rfloats->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(ftmp0_gpu[i], ftmp0->buf, ftmp0->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(char_prob_gpu[i], char_prob->buf, char_prob->num_elem() * sizeof(float), cudaMemcpyHostToDevice));
    }
    for (int l = 0; l < MAX_LEN; l++) { // 이름의 최대 길이는 10이다.
      /* Embedding */
      embedding(input, character_embedding, emb_out); // 여기서 input은 buf[0]만을 가지고 있다. 만약 사이즈는 512다.

      // double namegen_st = get_time();
      // // /* First layer r */ // GRU연산을 보면 r, z, n을 구하고 이를 통해서 h를 구한다. 
      matmul(W_ir0, emb_out, rtmp00); // weight_input * embedding / NK * K -> N (matrix mector multiplication)
      matmul(W_hr0, hidden0, rtmp01); // weight_hidden * hidden / 
      elemwise_add(rtmp00, b_ir0_stack, rtmp02); // weight_input * embedding + bias_input / 
      elemwise_add(rtmp02, rtmp01, rtmp03); //weight_input * embedding + weight_hidden * hidden + bias_input
      elemwise_add(rtmp03, b_hr0_stack, rtmp04); // weight_input * embedding + weight_hidden * hidden + bias_input + bias_hidden  
      elemwise_sigmoid(rtmp04, r0); // activation(weight_input * embedding + weight_hidden * hidden + bias_input + bias_hidden) -> r0
      // double namegen_en = get_time();
      // if (mpi_rank == 0) {
      //   double elapsed_time = namegen_en - namegen_st;
      //   printf("Elapsed time for first layer r: %.6f seconds\n", elapsed_time);
      // }

      // namegen_st = get_time();
      // /* First layer z */
      matmul(W_iz0, emb_out, ztmp00);
      matmul(W_hz0, hidden0, ztmp01);
      elemwise_add(ztmp00, b_iz0_stack, ztmp02);
      elemwise_add(ztmp02, ztmp01, ztmp03);
      elemwise_add(ztmp03, b_hz0_stack, ztmp04);
      elemwise_sigmoid(ztmp04, z0);
      // namegen_en = get_time();
      // if (mpi_rank == 0) {
      //   double elapsed_time = namegen_en - namegen_st;
      //   printf("Elapsed time for first layer z: %.6f seconds\n", elapsed_time);
      // }
      
      // namegen_st = get_time();
      // /* First layer n */
      matmul(W_in0, emb_out, ntmp00);
      elemwise_add(ntmp00, b_in0_stack, ntmp01);
      matmul(W_hn0, hidden0, ntmp02);
      elemwise_add(ntmp02, b_hn0_stack, ntmp03);
      elemwise_mul(r0, ntmp03, ntmp04);
      elemwise_add(ntmp01, ntmp04, ntmp05);
      elemwise_tanh(ntmp05, n0);
      // namegen_en = get_time();
      // if (mpi_rank == 0) {
      //   double elapsed_time = namegen_en - namegen_st;
      //   printf("Elapsed time for first layer n: %.6f seconds\n", elapsed_time);
      // }

      // namegen_st = get_time();
      // /* First layer h (hidden) */
      elemwise_oneminus(z0, htmp00); // 1 - z0
      elemwise_mul(htmp00, n0, htmp01); // vector * vector 
      elemwise_mul(z0, hidden0, htmp02);
      elemwise_add(htmp01, htmp02, hidden0);
      // namegen_en = get_time();
      // if (mpi_rank == 0) {
      //   double elapsed_time = namegen_en - namegen_st;
      //   printf("Elapsed time for first layer h: %.6f seconds\n", elapsed_time);
      // }

      // namegen_st = get_time();
      // /* Second layer r */ // GRU2
      matmul(W_ir1, hidden0, rtmp10);
      matmul(W_hr1, hidden1, rtmp11);
      elemwise_add(rtmp10, b_ir1_stack, rtmp12);
      elemwise_add(rtmp12, rtmp11, rtmp13);
      elemwise_add(rtmp13, b_hr1_stack, rtmp14);
      elemwise_sigmoid(rtmp14, r1);
      // namegen_en = get_time();
      // if (mpi_rank == 0) {
      //   double elapsed_time = namegen_en - namegen_st;
      //   printf("Elapsed time for second layer r: %.6f seconds\n", elapsed_time);
      // }

      // namegen_st = get_time();
      // /* Second layer z */
      matmul(W_iz1, hidden0, ztmp10);
      matmul(W_hz1, hidden1, ztmp11);
      elemwise_add(ztmp10, b_iz1_stack, ztmp12);
      elemwise_add(ztmp12, ztmp11, ztmp13);
      elemwise_add(ztmp13, b_hz1_stack, ztmp14);
      elemwise_sigmoid(ztmp14, z1);
      // namegen_en = get_time();
      // if (mpi_rank == 0) {
      //   double elapsed_time = namegen_en - namegen_st;
      //   printf("Elapsed time for second layer z: %.6f seconds\n", elapsed_time);
      // }

      // namegen_st = get_time();
      // /* Second layer n */
      matmul(W_in1, hidden0, ntmp10);
      elemwise_add(ntmp10, b_in1_stack, ntmp11);
      matmul(W_hn1, hidden1, ntmp12);
      elemwise_add(ntmp12, b_hn1_stack, ntmp13);
      elemwise_mul(r1, ntmp13, ntmp14);
      elemwise_add(ntmp11, ntmp14, ntmp15);
      elemwise_tanh(ntmp15, n1);
      // //shape
      // printf("\nshape of emb_out: <%zd, %zd>\n", n1->shape[0], n1->shape[1]);
      // //value
      // for(int i = 5; i < 10; i++){
      //   for(int j = 0; j < 2; j++){
      //     printf("%d번째 word %d번: %f\n",i, j+1, n1->buf[j * N + i]);
      //   }
      // }
      // namegen_en = get_time();
      // if (mpi_rank == 0) {
      //   double elapsed_time = namegen_en - namegen_st;
      //   printf("Elapsed time for second layer n: %.6f seconds\n", elapsed_time);
      // }

      // namegen_st = get_time();
      // /* Second layer h (hidden) */
      elemwise_oneminus(z1, htmp10);
      elemwise_mul(htmp10, n1, htmp11);
      elemwise_mul(z1, hidden1, htmp12);
      elemwise_add(htmp11, htmp12, hidden1);
      // //shape
      // printf("\nshape of emb_out: <%zd, %zd>\n", hidden1->shape[0], hidden1->shape[1]);
      // //value
      // for(int i = 5; i < 10; i++){
      //   for(int j = 0; j < 2; j++){
      //     printf("%d번째 word %d번: %f\n",i, j+1, hidden1->buf[j * N + i]);
      //   }
      // }
      // namegen_en = get_time();
      // if (mpi_rank == 0) {
      //   double elapsed_time = namegen_en - namegen_st;
      //   printf("Elapsed time for second layer h: %.6f seconds\n", elapsed_time);
      // }

      // namegen_st = get_time();
      // /* Fully connected layer */ // linear
      matmul(W_fc, hidden1, ftmp0);
      elemwise_add(ftmp0, b_fc_stack, f);

      // //shape
      // printf("\nshape of emb_out: <%zd, %zd>\n", f->shape[0], f->shape[1]);
      // //value
      // for(int i = 0; i < 1; i++){
      //   for(int j = 0; j < 1024; j++){
      //     printf("%d번째 word %d번: %f\n",i, j+1, hidden1->buf[j * N + i]);
      //   }
      // }    
      // namegen_en = get_time();
      // if (mpi_rank == 0) {
      //   double elapsed_time = namegen_en - namegen_st;
      //   printf("Elapsed time for fully connected layer: %.6f seconds\n", elapsed_time);
      // }

    //   // namegen_st = get_time();
    //   // /* Softmax */
      softmax(f, char_prob, BATCHSIZE);
    //   // //shape
    //   // printf("\nshape of emb_out: <%zd, %zd>\n", char_prob->shape[0], char_prob->shape[1]);
    //   // //value
    //   // for(int i = 5; i < 8; i++){
    //   //   for(int j = 0; j < 10; j++){
    //   //     printf("%d번째 word %d번: %f\n",i, j+1, char_prob->buf[j * N + i]);
    //   //   }
    //   // } 
    // //   // namegen_en = get_time();
    // //   // if (mpi_rank == 0) {
    // //   //   double elapsed_time = namegen_en - namegen_st;
    // //   //   printf("Elapsed time for softmax: %.6f seconds\n", elapsed_time);
    // //   // }

    //   // namegen_st = get_time();
    //   // /* Random select */
      random_select(char_prob, rfloats, input, output, N, l); // 아 씨발
    //   // //shape
    //   // printf("\nshape of emb_out: <%zd, %zd>\n", char_prob->shape[0], char_prob->shape[1]);
    //   // //value
    //   // for(int i = 0; i < 1; i++){
    //   //   for(int j = 0; j < 256; j++){
    //   //     printf("%d번째 word %d번: %f\n",i, j+1, char_prob->buf[j * BATCHSIZE + i]);
    //   //   }
    //   // }
    //   // break;
    //   // namegen_en = get_time();
    //   // if (mpi_rank == 0) {
    //   //   double elapsed_time = namegen_en - namegen_st;
    //   //   printf("Elapsed time for random select: %.6f seconds\n", elapsed_time);
    //   // }
    }
    
    MPI_Send(output + M_start * (MAX_LEN + 1), (M_end - M_start) * (MAX_LEN + 1), MPI_CHAR, 0, mpi_rank, MPI_COMM_WORLD);
    // printf("[mpi_rank:%d]amount of sended message: %d\n", mpi_rank,(M_end - M_start));
    // printf("[mpi_rank:%d] send address:%d\n",mpi_rank, M_start);
  }
}

/*
 * Finalize the model.
 * Although it is not neccessary, we recommend to deallocate and destruct
 * everything you made in namegen_initalize() and namegen().
 */
void namegen_finalize() {
  if (1) {
    delete character_embedding;
    delete W_ir0;
    delete W_iz0;
    delete W_in0;
    delete W_ir1;
    delete W_iz1;
    delete W_in1;
    delete W_hr0;
    delete W_hz0;
    delete W_hn0;
    delete W_hr1;
    delete W_hz1;
    delete W_hn1;
    delete b_ir0;
    delete b_iz0;
    delete b_in0;
    delete b_ir1;
    delete b_iz1;
    delete b_in1;
    delete b_hr0;
    delete b_hz0;
    delete b_hn0;
    delete b_hr1;
    delete b_hz1;
    delete b_hn1;
    delete W_fc;
    delete b_fc;
    delete rfloats;

    delete input;
    delete emb_out;
    delete hidden0;
    delete hidden1;
    delete r0;
    delete r1;
    delete z0;
    delete z1;
    delete n0;
    delete n1;
    delete f;
    delete char_prob;
    delete rtmp00;
    delete rtmp01;
    delete rtmp02;
    delete rtmp03;
    delete rtmp04;
    delete rtmp10;
    delete rtmp11;
    delete rtmp12;
    delete rtmp13;
    delete rtmp14;
    delete ztmp00;
    delete ztmp01;
    delete ztmp02;
    delete ztmp03;
    delete ztmp04;
    delete ztmp10;
    delete ztmp11;
    delete ztmp12;
    delete ztmp13;
    delete ztmp14;
    delete ntmp00;
    delete ntmp01;
    delete ntmp02;
    delete ntmp03;
    delete ntmp04;
    delete ntmp05;
    delete ntmp10;
    delete ntmp11;
    delete ntmp12;
    delete ntmp13;
    delete ntmp14;
    delete ntmp15;
    delete htmp00;
    delete htmp01;
    delete htmp02;
    delete htmp10;
    delete htmp11;
    delete htmp12;
    delete ftmp0;
  }
}