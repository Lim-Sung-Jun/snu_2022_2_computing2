#pragma once

#define MAX_LEN 10 // 주의

// Model parameters
#define PARAMETER_FILE_SIZE 45663232
#define NUM_CHAR 256
#define EMBEDDING_DIM 512
#define HIDDEN_DIM 1024

#define OFFSET0 0

#define OFFSET1 (OFFSET0 + NUM_CHAR * EMBEDDING_DIM)
#define OFFSET2 (OFFSET1 + HIDDEN_DIM * EMBEDDING_DIM)
#define OFFSET3 (OFFSET2 + HIDDEN_DIM * EMBEDDING_DIM)
#define OFFSET4 (OFFSET3 + HIDDEN_DIM * EMBEDDING_DIM)
#define OFFSET5 (OFFSET4 + HIDDEN_DIM * HIDDEN_DIM)
#define OFFSET6 (OFFSET5 + HIDDEN_DIM * HIDDEN_DIM)
#define OFFSET7 (OFFSET6 + HIDDEN_DIM * HIDDEN_DIM)
#define OFFSET8 (OFFSET7 + HIDDEN_DIM * HIDDEN_DIM)
#define OFFSET9 (OFFSET8 + HIDDEN_DIM * HIDDEN_DIM)
#define OFFSET10 (OFFSET9 + HIDDEN_DIM * HIDDEN_DIM)
#define OFFSET11 (OFFSET10 + HIDDEN_DIM * HIDDEN_DIM)
#define OFFSET12 (OFFSET11 + HIDDEN_DIM * HIDDEN_DIM)
#define OFFSET13 (OFFSET12 + HIDDEN_DIM * HIDDEN_DIM)
#define OFFSET14 (OFFSET13 + HIDDEN_DIM)
#define OFFSET15 (OFFSET14 + HIDDEN_DIM)
#define OFFSET16 (OFFSET15 + HIDDEN_DIM)
#define OFFSET17 (OFFSET16 + HIDDEN_DIM)
#define OFFSET18 (OFFSET17 + HIDDEN_DIM)
#define OFFSET19 (OFFSET18 + HIDDEN_DIM)
#define OFFSET20 (OFFSET19 + HIDDEN_DIM)
#define OFFSET21 (OFFSET20 + HIDDEN_DIM)
#define OFFSET22 (OFFSET21 + HIDDEN_DIM)
#define OFFSET23 (OFFSET22 + HIDDEN_DIM)
#define OFFSET24 (OFFSET23 + HIDDEN_DIM)
#define OFFSET25 (OFFSET24 + HIDDEN_DIM)
#define OFFSET26 (OFFSET25 + NUM_CHAR * HIDDEN_DIM)
#define OFFSET27 (OFFSET26 + NUM_CHAR)

#define EOS 0
#define SOS 1
#define PAD 2

void namegen_initialize(int N, int rng_seed, char *network_fname);
void namegen(int N, float *random_floats, char *output);
void namegen_finalize();