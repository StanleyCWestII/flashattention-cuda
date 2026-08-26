NVCC  := nvcc
ARCH  := -arch=sm_89
FLAGS := -O2 $(ARCH)

all: test_flash bench_flash

test_flash: test_flash.cu flashattention.cu
	$(NVCC) $(FLAGS) -o test_flash test_flash.cu

bench_flash: bench_flash.cu flashattention.cu basekernel.cu
	$(NVCC) $(FLAGS) -o bench_flash bench_flash.cu

test: test_flash
	./test_flash

bench: bench_flash
	./bench_flash

clean:
	rm -f test_flash bench_flash

.PHONY: all test bench clean
