main: main.cu matmul.cuh
	nvcc -O3 -o main main.cu -Wno-deprecated-gpu-targets -Xcompiler -O3 -Xcompiler -fopenmp

clean:
	rm -f main