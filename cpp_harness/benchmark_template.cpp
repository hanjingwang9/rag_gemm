#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
        std::cerr << " -> " << cudaGetErrorString(err_) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CUBLAS_CHECK(err) { \
    cublasStatus_t err_ = (err); \
    if (err_ != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__; \
        std::cerr << " -> " << err_ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// ==================================================================
// START: AI-GENERATED KERNEL CODE WILL BE INSERTED HERE
// ==================================================================
// {{GENERATED_KERNEL_CODE}}
// ==================================================================
// END: AI-GENERATED KERNEL CODE
// ==================================================================

void run_benchmark(int M, int N, int K) {{
    std::cout << "\\n--- Testing Size M=" << M << ", N=" << N << ", K=" << K << " ---" << std::endl;
    
    // Memory allocation
    float *h_A, *h_B, *h_C_handwritten, *h_C_cublas;
    CUDA_CHECK(cudaMallocHost(&h_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_C_handwritten, M * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_C_cublas, M * N * sizeof(float)));

    // Data initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 threads(256, 1);
    dim3 blocks(CEIL_DIV(N, 32), CEIL_DIV(M, 32));

    {kernel_name}<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0;
    int num_runs = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        {kernel_name}<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    double avg_ms = total_ms / num_runs;
    double tflops = (2.0 * M * N * K * 1e-12) / (avg_ms * 1e-3);
    std::cout << "Handwritten Kernel Time: " << avg_ms << " ms | Performance: " << tflops << " TFLOPS" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_C_handwritten, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float))); // Reset C for cuBLAS

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize()); // Warm-up

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    
    double avg_cublas_ms = total_ms / num_runs;
    double cublas_tflops = (2.0 * M * N * K * 1e-12) / (avg_cublas_ms * 1e-3);
    std::cout << "cuBLAS Kernel Time:    " << avg_cublas_ms << " ms | Performance: " << cublas_tflops << " TFLOPS" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    double error = 0.0;
    for (int i = 0; i < M * N; i++) {
        error += fabs(h_C_handwritten[i] - h_C_cublas[i]);
    }
    std::cout << "\nVerification Average Error per Element: " << error / (M * N) << std::endl;
    std::cout.precision(2);
    std::cout << "Performance vs cuBLAS: " << std::fixed << (tflops / cublas_tflops) * 100.0 << "%" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_A)); CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C_handwritten)); CUDA_CHECK(cudaFreeHost(h_C_cublas));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));

}}

int main() {{
    run_benchmark(512, 512, 512);
    run_benchmark(1024, 1024, 1024);
    run_benchmark(2048, 2048, 2048);
    return 0;
}}
