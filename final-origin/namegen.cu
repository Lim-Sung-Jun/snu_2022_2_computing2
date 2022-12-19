#include "namegen.h"
#include "util.h"

#include <cassert>
#include <math.h>
#include <vector>

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

/* Operations */

/*
 * Embedding
 * input: [1] (scalar)
 * weight: [NUM_CHAR x EMBEDDING_DIM]
 * output: [EMBEDDING_DIM]
 */
void embedding(Tensor *input, Tensor *weight, Tensor *output) {
  size_t n = weight->shape[1]; // weight에 shape도 있고 buf가 있다.
  for (size_t i = 0; i < n; i++) {
    int x = (int)input->buf[0];  // 현재 문자가 무엇인지 알아낸다.
    output->buf[i] = weight->buf[x * n + i];
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
void matmul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t M_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  size_t N_ = input2->shape[1];
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

/*
 * Softmax
 * Normalize the input elements according to its exp value.
 * The result can be interpreted as a probability distribution.
 * input: [*]
 * output: [*], (same shape as input)
 */
void softmax(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  float sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    sum += expf(x);
  }
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = expf(x) / sum;
  }
}

/*
 * Sample a random index according to the given probability distribution 
 * This function is called at most N*MAX_LEN times. Each call uses a
 * random float in [0,1] to sample an index from the given distribution.
 * input: [NUM_CHAR], probability distribution of the characters
 * rng_seq: [N*MAX_LEN],
 */
int random_select(Tensor *input, Tensor *rng_seq, int rng_offset) { //char_prob, rfloats, n * MAX_LEN + l // 글씨의 최대 길이. ??
  float r = rng_seq->buf[rng_offset]; //
  size_t n = input->num_elem();
  float psum = 0.0;
  for (size_t i = 0; i < n; i++) {
    psum += input->buf[i];
    if (psum > r) {
      printf("psum:%f, r:%f, selected:%zd\n",psum, r, i);
      return i;
    }
  }
  printf("psum:%f, r:%f, selected:%zd\n",psum, r, n-1);
  return n - 1;
}

/*
 * Initialize the model.
 * Do input-independent job here.
 */
void namegen_initialize(int N, int rng_seed, char *parameter_fname) {

  /* Only the root process reads the parameter */ // 모든 파라메터를 가져온다.
  if (mpi_rank == 0) {
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

    W_fc = new Tensor({NUM_CHAR, HIDDEN_DIM}, parameter + OFFSET25);//256,1024
    b_fc = new Tensor({NUM_CHAR}, parameter + OFFSET26);//256

    /* input, activations, output, etc. */
    input = new Tensor({1});//1
    emb_out = new Tensor({EMBEDDING_DIM});//512

    hidden0 = new Tensor({HIDDEN_DIM});//1024
    hidden1 = new Tensor({HIDDEN_DIM});//1024

    r0 = new Tensor({HIDDEN_DIM});//1024
    r1 = new Tensor({HIDDEN_DIM});//1024
    z0 = new Tensor({HIDDEN_DIM});//1024
    z1 = new Tensor({HIDDEN_DIM});//1024
    n0 = new Tensor({HIDDEN_DIM});//1024
    n1 = new Tensor({HIDDEN_DIM});//1024
    f = new Tensor({NUM_CHAR});//256

    rtmp00 = new Tensor({HIDDEN_DIM});//1024
    rtmp01 = new Tensor({HIDDEN_DIM});//1024
    rtmp02 = new Tensor({HIDDEN_DIM});//1024
    rtmp03 = new Tensor({HIDDEN_DIM});//1024
    rtmp04 = new Tensor({HIDDEN_DIM});//1024
    rtmp10 = new Tensor({HIDDEN_DIM});//1024
    rtmp11 = new Tensor({HIDDEN_DIM});//1024
    rtmp12 = new Tensor({HIDDEN_DIM});//1024
    rtmp13 = new Tensor({HIDDEN_DIM});//1024
    rtmp14 = new Tensor({HIDDEN_DIM});//1024

    ztmp00 = new Tensor({HIDDEN_DIM});//1024
    ztmp01 = new Tensor({HIDDEN_DIM});//1024
    ztmp02 = new Tensor({HIDDEN_DIM});//1024
    ztmp03 = new Tensor({HIDDEN_DIM});//1024
    ztmp04 = new Tensor({HIDDEN_DIM});//1024
    ztmp10 = new Tensor({HIDDEN_DIM});
    ztmp11 = new Tensor({HIDDEN_DIM});
    ztmp12 = new Tensor({HIDDEN_DIM});
    ztmp13 = new Tensor({HIDDEN_DIM});
    ztmp14 = new Tensor({HIDDEN_DIM});

    ntmp00 = new Tensor({HIDDEN_DIM});
    ntmp01 = new Tensor({HIDDEN_DIM});
    ntmp02 = new Tensor({HIDDEN_DIM});
    ntmp03 = new Tensor({HIDDEN_DIM});
    ntmp04 = new Tensor({HIDDEN_DIM});
    ntmp05 = new Tensor({HIDDEN_DIM});
    ntmp10 = new Tensor({HIDDEN_DIM});
    ntmp11 = new Tensor({HIDDEN_DIM});
    ntmp12 = new Tensor({HIDDEN_DIM});
    ntmp13 = new Tensor({HIDDEN_DIM});
    ntmp14 = new Tensor({HIDDEN_DIM});
    ntmp15 = new Tensor({HIDDEN_DIM});

    htmp00 = new Tensor({HIDDEN_DIM});
    htmp01 = new Tensor({HIDDEN_DIM});
    htmp02 = new Tensor({HIDDEN_DIM});
    htmp10 = new Tensor({HIDDEN_DIM});
    htmp11 = new Tensor({HIDDEN_DIM});
    htmp12 = new Tensor({HIDDEN_DIM});//1024

    rfloats = new Tensor({N * MAX_LEN});// generate 글자 * 글자 수
    ftmp0 = new Tensor({NUM_CHAR});//256
    char_prob = new Tensor({NUM_CHAR});//256
  } else {
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

  /* Only root process does the job, for now... */
  if (mpi_rank != 0)
    return;

  memcpy(rfloats->buf, random_floats, N * MAX_LEN * sizeof(float)); // buf = pointer to data, 메모리에 있는 random_floats를 buf로 옮긴다.
  memset(output, 0, N * (MAX_LEN + 1) * sizeof(char)); // 메모리 시작점부터 특정 범위까지 특정 값으로 지정할 수 있다.

  /* Generate N names */
  for (int n = 0; n < N; n++) {
    /* Initialize input and hidden vector. */
    /* One hidden vector for each GRU layer */
    input->buf[0] = SOS; // start of sequence
    hidden0->set_zero(); // hidden0,1을 0으로 지정한다.
    hidden1->set_zero();

    for (int l = 0; l < MAX_LEN; l++) { // 이름의 최대 길이는 10이다.
      /* Embedding */
      embedding(input, character_embedding, emb_out); // 여기서 input은 buf[0]만을 가지고 있다. 만약 사이즈는 512다.
      
      /* First layer r */ // GRU연산을 보면 r, z, n을 구하고 이를 통해서 h를 구한다. 
      matvec(W_ir0, emb_out, rtmp00); // weight_input * embedding / NK * K -> N (matrix mector multiplication)
      matvec(W_hr0, hidden0, rtmp01); // weight_hidden * hidden / 
      elemwise_add(rtmp00, b_ir0, rtmp02); // weight_input * embedding + bias_input / 
      elemwise_add(rtmp02, rtmp01, rtmp03); //weight_input * embedding + weight_hidden * hidden + bias_input
      elemwise_add(rtmp03, b_hr0, rtmp04); // weight_input * embedding + weight_hidden * hidden + bias_input + bias_hidden
      elemwise_sigmoid(rtmp04, r0); // activation(weight_input * embedding + weight_hidden * hidden + bias_input + bias_hidden) -> r0
      /* First layer z */
      matvec(W_iz0, emb_out, ztmp00);
      matvec(W_hz0, hidden0, ztmp01);
      elemwise_add(ztmp00, b_iz0, ztmp02);
      elemwise_add(ztmp02, ztmp01, ztmp03);
      elemwise_add(ztmp03, b_hz0, ztmp04);
      elemwise_sigmoid(ztmp04, z0);

      /* First layer n */
      matvec(W_in0, emb_out, ntmp00);
      elemwise_add(ntmp00, b_in0, ntmp01);
      matvec(W_hn0, hidden0, ntmp02);
      elemwise_add(ntmp02, b_hn0, ntmp03);
      elemwise_mul(r0, ntmp03, ntmp04);
      elemwise_add(ntmp01, ntmp04, ntmp05);
      elemwise_tanh(ntmp05, n0);

      /* First layer h (hidden) */
      elemwise_oneminus(z0, htmp00); // 1 - z0
      elemwise_mul(htmp00, n0, htmp01); // vector * vector 
      elemwise_mul(z0, hidden0, htmp02);
      elemwise_add(htmp01, htmp02, hidden0);

      /* Second layer r */ // GRU2
      matvec(W_ir1, hidden0, rtmp10);
      matvec(W_hr1, hidden1, rtmp11);
      elemwise_add(rtmp10, b_ir1, rtmp12);
      elemwise_add(rtmp12, rtmp11, rtmp13);
      elemwise_add(rtmp13, b_hr1, rtmp14);
      elemwise_sigmoid(rtmp14, r1);

      /* Second layer z */
      matvec(W_iz1, hidden0, ztmp10);
      matvec(W_hz1, hidden1, ztmp11);
      elemwise_add(ztmp10, b_iz1, ztmp12);
      elemwise_add(ztmp12, ztmp11, ztmp13);
      elemwise_add(ztmp13, b_hz1, ztmp14);
      elemwise_sigmoid(ztmp14, z1);

      /* Second layer n */
      matvec(W_in1, hidden0, ntmp10);
      elemwise_add(ntmp10, b_in1, ntmp11);
      matvec(W_hn1, hidden1, ntmp12);
      elemwise_add(ntmp12, b_hn1, ntmp13);
      elemwise_mul(r1, ntmp13, ntmp14);
      elemwise_add(ntmp11, ntmp14, ntmp15);
      elemwise_tanh(ntmp15, n1);

      /* Second layer h (hidden) */
      elemwise_oneminus(z1, htmp10);
      elemwise_mul(htmp10, n1, htmp11);
      elemwise_mul(z1, hidden1, htmp12);
      elemwise_add(htmp11, htmp12, hidden1);
      /* Fully connected layer */ // linear
      matvec(W_fc, hidden1, ftmp0);
      elemwise_add(ftmp0, b_fc, f);
      
      // //shape
      // printf("\nshape of emb_out: <%zd, %zd>\n", char_prob->shape[0], char_prob->shape[1]);
      // //value
      // for(int j = 0; j < 1024; j++){
      //   printf("1번째 word %d번: %f\n", j+1, char_prob->buf[j]);
      // }
      // exit(-1);

      /* Softmax */
      softmax(f, char_prob);
      /* Random select */
      int selected_char = random_select(char_prob, rfloats, n * MAX_LEN + l);
      // printf("\nrng_offset: %d\n", n * MAX_LEN + l); //Tensor *input, Tensor *rng_seq, int rng_offset / word 256개에 대해서 확률을 갖는다.
      // // int length = sizeof(char_prob)/sizeof(size_t);
      // // printf("length: %d\n",NUM_CHAR); // 256
      // // for(int i = 0; i < NUM_CHAR; i++){
      // //   if(char_prob->buf[i] != 0.0){
      // //     printf("char_prob[%d]: %f\n", i, char_prob->buf[i]);
      // //   } 
      // // }

      output[n * (MAX_LEN + 1) + l] = selected_char;
      input->buf[0] = selected_char; // 다시 입력으로 넣는다.

      if (selected_char == EOS)
        break;
    }
  }
}

/*
 * Finalize the model.
 * Although it is not neccessary, we recommend to deallocate and destruct
 * everything you made in namegen_initalize() and namegen().
 */
void namegen_finalize() {
  if (mpi_rank == 0) {
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


