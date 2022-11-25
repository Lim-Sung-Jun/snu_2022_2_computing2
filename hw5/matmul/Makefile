TARGET=main
OBJECTS=util.o matmul.o

CPPFLAGS=-O3 -Wall -march=native -mavx2 -mfma -fopenmp -mno-avx512f -I/usr/local/cuda-11.7/targets/x86_64-linux/include
LDFLAGS=-L/usr/local/cuda-11.7/targets/x86_64-linux/lib
LDLIBS=-lm -pthread -lOpenCL

CC=gcc

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
