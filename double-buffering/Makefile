all: a.out

a.out: main.cu
	nvcc -O3 -o $@ $^

clean:
	rm -rf main
