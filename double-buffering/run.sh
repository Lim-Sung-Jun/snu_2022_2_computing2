#!/bin/bash

# To run Nsight systems
srun --nodes=1 --exclusive --partition=shpc22 --gres=gpu:4 numactl --physcpubind 0-63 nsys profile --cudabacktrace=all ./main $@

# To run Nsight compute
#srun --nodes=1 --exclusive --partition=shpc22 --gres=gpu:4 numactl --physcpubind 0-63 ncu ./main $@
