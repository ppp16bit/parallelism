/* his code searches for a pattern (string) within a text,
* using two approaches: CPU and GPU (with CUDA).
* The search function on the CPU is implemented in a simple way, while the search on the GPU
* is done with a CUDA kernel to take advantage of the parallelism of the graphics card.
* the program compares the performance of both approaches and prints the time spent
* by each one, as well as the performance gain (speedup) when using the GPU. */

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// turing architecture for gtx 1650 mobile
const int BLOCK_SIZE = 256;
const int MAX_PATTERN = 32;

void gpu_warmup() {
    char* d_temp;
    cudaMalloc(&d_temp, 1);
    cudaFree(d_temp);
}

std::vector<int> cpu_search(const std::string& text, const std::string& pattern) {
    std::vector<int> matches;
    size_t pos = text.find(pattern);
    while(pos != std::string::npos) {
        matches.push_back(pos);
        pos = text.find(pattern, pos+1);
    }
    return matches;
}

__global__ void gpu_search_kernel(const char* text, int text_len, const char* pattern, 
                                int pattern_len, int* matches, int* count) {
    extern __shared__ char s_pattern[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (threadIdx.x < pattern_len) {
        s_pattern[threadIdx.x] = pattern[threadIdx.x];
    }
    __syncthreads();
    
    if (tid <= text_len - pattern_len) {
        bool found = true;
        for (int i = 0; i < pattern_len; ++i) {
            if (text[tid + i] != s_pattern[i]) {
                found = false;
                break;
            }
        }
        if (found) {
            int pos = atomicAdd(count, 1);
            matches[pos] = tid;
        }
    }
}

std::vector<int> gpu_search(const std::string& text, const std::string& pattern) {
    if (pattern.empty() || text.size() < pattern.size() || pattern.size() > MAX_PATTERN) {
        return cpu_search(text, pattern);
    }
    
    char *d_text, *d_pattern;
    int *d_matches, *d_count;
    cudaMalloc(&d_text, text.size());
    cudaMalloc(&d_pattern, pattern.size());
    cudaMalloc(&d_matches, text.size() * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    
    cudaMemcpy(d_text, text.data(), text.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, pattern.data(), pattern.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));
    
    int grid_size = (text.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpu_search_kernel<<<grid_size, BLOCK_SIZE, pattern.size()>>>(d_text, text.size(), 
                                                               d_pattern, pattern.size(),
                                                               d_matches, d_count);
    
    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::vector<int> matches(count);
    if (count > 0) {
        cudaMemcpy(matches.data(), d_matches, count * sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    cudaFree(d_text);
    cudaFree(d_pattern);
    cudaFree(d_matches);
    cudaFree(d_count);
    
    return matches;
}

void run_test(const std::string& text, const std::string& pattern) {
    static bool first_run = true;
    if (first_run) {
        gpu_warmup();
        first_run = false;
    }
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto cpu_result = cpu_search(text, pattern);
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - cpu_start).count();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    auto gpu_result = gpu_search(text, pattern);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    
    printf("%6zu | %2zu | %5ld | %6.0f | %4.1f | %s\n",
           text.size(), pattern.size(), cpu_time, gpu_time_ms*1000,
           cpu_time/(gpu_time_ms*1000),
           (cpu_result == gpu_result) ? "OK" : "FAIL");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(void) {
    printf("::: result :::\n");
    printf(" size | standard | CPU(us) | GPU(us) | Speedup | Valid\n");
    printf("-------------------------------------------------------\n");
    
    std::string base(50000, 'a');
    std::string pattern = "abc123";
    
    for (int i = 1; i <= 5; i++) {
        std::string text = base.substr(0, 10000*i) + pattern + base.substr(0, 10000*i);
        run_test(text, pattern);
    }
    
    return 0;
}