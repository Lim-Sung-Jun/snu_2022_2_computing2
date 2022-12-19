#!/bin/bash

# : ${NODES:=1}
# salloc -N $NODES --partition shpc22 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 1 \
#     numactl --physcpubind 0-63 \
#     ./main $@


# : ${NODES:=4}
# salloc -N $NODES --partition shpc22 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 1 \
#     numactl --physcpubind 0-63 \
#     ./main model.bin output.txt 16 4155

# : ${NODES:=1}
# salloc -N $NODES --partition shpc22 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 1 \
#     numactl --physcpubind 0-63 \
#     ./main model.bin output.txt 16 4155

: ${NODES:=1}
salloc -N $NODES --partition shpc22 --exclusive --gres=gpu:4 \
    mpirun --bind-to none -mca btl ^openib -npernode 1 \
    numactl --physcpubind 0-63 \
    ./main model.bin output.txt 64 4155

# : ${NODES:=1}
# salloc -N $NODES --partition shpc22 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 1 \
#     numactl --physcpubind 0-63 \
#     ./main model.bin output.txt 128 4155

# : ${NODES:=1}
# salloc -N $NODES --partition shpc22 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 1 \
#     numactl --physcpubind 0-63 \
#     ./main model.bin output.txt 256 4155

# : ${NODES:=4}
# salloc -N $NODES --partition shpc22 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 1 \
#     numactl --physcpubind 0-63 \
#     ./main model.bin output.txt 512 4155

# : ${NODES:=8}
# salloc -N $NODES --partition shpc22 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 1 \
#     numactl --physcpubind 0-63 \
#     ./main model.bin output.txt 1024 4155

# : ${NODES:=1}
# salloc -N $NODES --partition shpc22 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 1 \
#     numactl --physcpubind 0-63 \
#     ./main model.bin output.txt 2048 4155

# : ${NODES:=1}
# salloc -N $NODES --partition shpc22 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 1 \
#     numactl --physcpubind 0-63 \
#     ./main model.bin output.txt 4096 4155

# : ${NODES:=4}
# salloc -N $NODES --partition shpc22 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 1 \
#     numactl --physcpubind 0-63 \
#     ./main model.bin output.txt 8192 4155

# : ${NODES:=1}
# salloc -N $NODES --partition shpc22 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 1 \
#     numactl --physcpubind 0-63 \
#     ./main model.bin output.txt 65536 4155