#include <iostream>
#include "omp.h"
int main() {
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::cout << "Hello from thread " << thread_id << std::endl;
        float x = thread_id * 1.0f;
        float y = 3.14;
        while (1) {
            x = y + x / y;
            y = x + y / x;
        }
        std::cout << "Thread " << thread_id << " finished computation." << std::endl;
    }
}