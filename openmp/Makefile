all: main main.s

main: main.cpp
	g++ -std=c++20 -Ofast main.cpp -o main -fopenmp -mavx2 -funroll-all-loops -funroll-loops

main.s: main.cpp
	g++ -S -std=c++20 -Ofast main.cpp -o main.s -fopenmp -mavx2 -funroll-all-loops -funroll-loops
	objdump -d main > main.asm

run: main
	taskset -c 0-15 ./main

loop: loop.cpp
	g++ -std=c++20 -Ofast loop.cpp -o loop -fopenmp -mavx2

clean:
	rm -rf main