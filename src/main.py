import argparse
import os
import pickle
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import AzureOpenAIEmbeddings
from . import rag_system
from . import benchmark
from . import config

def main():
    parser = argparse.ArgumentParser(description="Run GEMM kernel generation and benchmarking.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=["openai", "llama"],
        help="Specify the chat model family to use ('openai' or 'llama')."
    )
    args = parser.parse_args()

    vector_store_cache_path = "vector_store.pkl"

    if os.path.exists(vector_store_cache_path):
        with open(vector_store_cache_path, "rb") as f:
            vector_store = pickle.load(f)

            print("Re-initializing embedding client for loaded vector store...")
            embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=config.embedding_endpoint,
                api_key=config.api_key,
                azure_deployment=config.embedding_deployment,
                openai_api_version=config.embedding_api_version,
                )
            vector_store.embedding = embeddings
            print("Vector store loaded from cache.")
    else:
        print("No cached vector store found. Building a new one...")
        loader = WebBaseLoader(
        web_paths=("https://docs.nvidia.com/cuda/cuda-c-programming-guide/",))
        full_doc = loader.load()
        vector_store = rag_system.store_docs(full_doc)
        embedding_client = vector_store.embedding
        vector_store.embedding = None

        with open(vector_store_cache_path, "wb") as f:
            pickle.dump(vector_store, f)
        vector_store.embedding = embedding_client
        print("Vector store cached successfully.")
    
    if args.model == "openai":
        models_to_test = config.openai_deployments
    elif args.model == "llama":
        models_to_test = config.llama_deployments
    else:
        print(f"Model family '{args.model}' is not yet supported.")
        models_to_test = []

    user_prompt = """You are a highly capable programmer. With specific code and not using any built-in libraries such as CUTLASS,
    please implement a high-performing GEMM kernel for FP32 in CUDA, with performance similar to cuBLAS. Your code should not be simplified
    in any way, instead fully exploiting the capabilities of GPUs to ensure maximal performance. Do not return anything other than the kernel itself,
    written strictly in C++ code, and do not define any other data structures or constants (such as block sizes) outside the kernel. 
    Your kernel should take in FP32 blocks A, B, C (C being the product) as well as integer dimensions M, N, K."""

    for model_deployment in models_to_test:
        print(f"\n\n{'='*20} TESTING MODEL: {model_deployment} {'='*20}")
        llm = rag_system.get_llm(args.model, model_deployment)
        if not llm:
            print(f"Could not initialize model {model_deployment}. Skipping.")
            continue

        graph = rag_system.run_rag_gemm(args.model, llm, vector_store)
        generated_code = rag_system.generate_code(graph, llm, user_prompt)
        
        print(f"\n--- BENCHMARKING RAG-GENERATED {model_deployment} KERNEL ---")
        rag_kernel = benchmark.extract_cpp_code(generated_code["rag_answer"])
        if rag_kernel:
            rag_harness = benchmark.create_benchmark_harness(rag_kernel)
            benchmark.compile_and_run(rag_harness, f"rag_{model_deployment}.cu")
        else:
            print("Could not extract C++ code from the RAG response.")

        print(f"\n--- BENCHMARKING BASELINE {model_deployment} KERNEL ---")
        baseline_kernel = benchmark.extract_cpp_code(generated_code["baseline_answer"])
        if baseline_kernel:
            baseline_harness = benchmark.create_benchmark_harness(baseline_kernel)
            benchmark.compile_and_run(baseline_harness, f"baseline_{model_deployment}.cu")
        else:
            print("Could not extract C++ code from the Baseline response.")


if __name__ == "__main__":
    main()