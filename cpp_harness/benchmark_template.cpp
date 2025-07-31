#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define CUDA_CHECK(err) {{ /* ... (CUDA_CHECK macro) ... */ }}
#define CUBLAS_CHECK(err) {{ /* ... (CUBLAS_CHECK macro) ... */ }}

// ==================================================================
// START: AI-GENERATED KERNEL CODE WILL BE INSERTED HERE
// ==================================================================
// {{GENERATED_KERNEL_CODE}}
// ==================================================================
// END: AI-GENERATED KERNEL CODE
// ==================================================================

void run_benchmark(int M, int N, int K) {{
    std::cout << "\\n--- Testing Size M=" << M << ", N=" << N << ", K=" << K << " ---" << std::endl;
    
    // The full run_benchmark function from our previous examples goes here
    {kernel_name}<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
}}

int main() {{
    run_benchmark(512, 512, 512);
    run_benchmark(1024, 1024, 1024);
    run_benchmark(2048, 2048, 2048);
    return 0;
}}
