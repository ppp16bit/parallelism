/**
* implementation of parallel Radix Sort in CUDA using the CUB library
*
* this code compares the sorting performance between:
* - radix Sort on the GPU (via CUB DeviceRadixSort)
* - std::sort on the CPU (standard C++ sorting algorithm)
*
* the program generates a vector of 1 million random integers and measures the time
* required to sort using both methods. After sorting,
* the correctness of the result is verified and performance metrics are presented
* including the speedup obtained by the GPU version.
*
*
* result  - GPU raidx: 0.000901536 s
*         - CPU std::sort: 0.220703 s
*         - speedup: 244.808×
*
* note: this code was tested on an NVIDIA GeForce GTX 1650 GPU.
* */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

bool isSorted(const std::vector<int>& v) {
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i-1] > v[i]) return false;
    return true;
}

int main() {
    const size_t N = 1 << 20;
    std::vector<int> h_data(N), h_cpu(N);
    srand(42);
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = rand() % 1000000;
        h_cpu[i]  = h_data[i];
    }

    int *d_keys_in, *d_keys_out;
    size_t temp_bytes = 0;
    void *d_temp = nullptr;

    checkCuda(cudaMalloc(&d_keys_in,  N * sizeof(int)));
    checkCuda(cudaMalloc(&d_keys_out, N * sizeof(int)));
    checkCuda(cudaMemcpy(d_keys_in, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
                                   d_keys_in, d_keys_out, N);
    checkCuda(cudaMalloc(&d_temp, temp_bytes));

    cudaEvent_t start, stop;
    float gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
                                   d_keys_in, d_keys_out, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    checkCuda(cudaMemcpy(h_data.data(), d_keys_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "GPU radix sort: " << gpuTime / 1000.0 << " s\n";

    if (!isSorted(h_data)) {
        std::cerr << "Erro: ordenação na GPU incorreta!\n";
        return 1;
    }

    float cpuTime = 0;
    cudaEventRecord(start, 0);
    std::sort(h_cpu.begin(), h_cpu.end());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpuTime, start, stop);

    std::cout << "CPU std::sort:  " << cpuTime / 1000.0 << " s\n";
    std::cout << "Speedup:        " << (cpuTime / gpuTime) << "×\n";

    std::cout << "Nota: execução realizada em uma NVIDIA GeForce GTX 1650 (humilde, porém valente!)\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_temp);
    
    return 0;
}
