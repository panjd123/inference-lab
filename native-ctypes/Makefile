all: libsquare.so libsquare_all.so main

libsquare.so: square.cpp
	g++ -O3 -fPIC -shared square.cpp -o libsquare

libsquare_all.so: square.cpp square.cu
	nvcc -O3 -Xcompiler -O3 -Xcompiler -fPIC -Xcompiler -fopenmp -shared square.cpp square.cu -o libsquare_all.so -Wno-deprecated-gpu-targets

main: main.cpp square.h libsquare.so
	g++ -std=c++20 -O3 main.cpp -o main -L. -lsquare_all -fopenmp

clean:
	rm -f libsquare.so libsquare_all.so main dive_into