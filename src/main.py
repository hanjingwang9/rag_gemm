from langchain_community.document_loaders import WebBaseLoader
from . import rag_system
from . import benchmark

def main():
    loader = WebBaseLoader(
    web_paths=("https://docs.nvidia.com/cuda/cuda-c-programming-guide/",))
    full_doc = loader.load()
    vector_store = rag_system.store_docs(full_doc)
    
    graph1, llm1 = rag_system.create_rag_graph("gpt-4.1", vector_store)
    graph2, llm2 = rag_system.create_rag_graph("gpt-4o", vector_store)

    user_prompt = """You are a highly capable programmer. With specific code and not using any built-in libraries such as CUTLASS,
    please implement a high-performing GEMM kernel for FP32 in CUDA, with performance similar to cuBLAS. Your code should not be simplified
    in any way, instead fully exploiting the capabilities of GPUs to ensure maximal performance. Do not return anything other than the kernel itself,
    written strictly in C++ code. Your kernel should take in FP32 matrices A, B, C (C being the product) as well as integer dimensions M, N, K."""

    generated_code_1 = rag_system.generate_code(graph1, llm1, user_prompt)
    generated_code_2 = rag_system.generate_code(graph2, llm2, user_prompt)

    print("\n\n--- BENCHMARKING RAG-GENERATED GPT-4.1 KERNEL ---")
    rag_kernel_1 = benchmark.extract_cpp_code(generated_code_1["rag_answer"])
    if rag_kernel_1:
        rag_harness_1 = benchmark.create_benchmark_harness(rag_kernel_1)
        benchmark.compile_and_run(rag_harness_1, "rag_gemm_1.cu")
    else:
        print("Could not extract C++ code from the RAG response.")

    print("\n\n--- BENCHMARKING BASELINE-GENERATED GPT-4.1 KERNEL ---")
    baseline_kernel_1 = benchmark.extract_cpp_code(generated_code_1["baseline_answer"])
    if baseline_kernel_1:
        baseline_harness_1 = benchmark.create_benchmark_harness(baseline_kernel_1)
        benchmark.compile_and_run(baseline_harness_1, "baseline_gemm_1.cu")
    else:
        print("Could not extract C++ code from the Baseline response.")

    print("\n\n--- BENCHMARKING RAG-GENERATED GPT-4o KERNEL ---")
    rag_kernel_2 = benchmark.extract_cpp_code(generated_code_2["rag_answer"])
    if rag_kernel_2:
        rag_harness_2 = benchmark.create_benchmark_harness(rag_kernel_2)
        benchmark.compile_and_run(rag_harness_2, "rag_gemm_2.cu")
    else:
        print("Could not extract C++ code from the RAG response.")

    print("\n\n--- BENCHMARKING BASELINE-GENERATED GPT-4o KERNEL ---")
    baseline_kernel_2 = benchmark.extract_cpp_code(generated_code_2["baseline_answer"])
    if baseline_kernel_2:
        baseline_harness_2 = benchmark.create_benchmark_harness(baseline_kernel_2)
        benchmark.compile_and_run(baseline_harness_2, "baseline_gemm_2.cu")
    else:
        print("Could not extract C++ code from the Baseline response.")


if __name__ == "__main__":
    main()