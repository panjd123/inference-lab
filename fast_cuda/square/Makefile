main: main.cu square.cuh
	nvcc -O3 -o main main.cu -Wno-deprecated-gpu-targets -arch=sm_89

clean:
	rm -f main