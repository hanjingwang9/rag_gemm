# An Exploration of RAG-Supplemented LLM Code Generation for CUDA

**Author:** Hanjing Wang

**Date:** July 31, 2025

# Overview
Retrieval-augmented generation has been shown to improve the capabilities of LLMs in myriad of question-answering metrics (Guu et al. 2020, Borgeaud et al. 2022), as well in general code generation (Su et al. 2024). However, its effects on optimizing code generation in terms of system performance has been so far unclear. To explore its capabilities in creating high-performing CUDA code in particular, I created a simple RAG model which retrieves information from NVIDIA's CUDA manual. Then, I tasked an LLM trained under this model with creating a high-performing GEMM and compared results with its baseline (the LLM without an RAG application). 

The full code is available at [here](https://colab.research.google.com/drive/1Bcuc70JpNDcuzCypnodZl7-h1aemYdZh?usp=sharing).

# Methodology

I initialized two models: the first as the chat model we will actually eventually prompt, and the second as an embedding model which stores the embeddings of the document we loaded and creates a query-able vector store that can later be used to retrieve relevant information.

```Python
    llm = AzureChatOpenAI(
        azure_endpoint=llm_endpoint,
        api_key=api_key,
        azure_deployment=llm_deployment,
        api_version=llm_api_version,
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=embedding_endpoint,
        api_key=api_key,
        azure_deployment=embedding_deployment,
        openai_api_version=embedding_api_version,
    )
```
Then, we load the NVIDIA C++ CUDA Programming Guide as a document using `WebBaseLoader`, which we will later split into chunks and store as embeddings.

```Python
    loader = WebBaseLoader(
        web_paths=("https://docs.nvidia.com/cuda/cuda-c-programming-guide/",),
    )
    full_doc = loader.load()
```


However, I quickly realized that the entire guide is extremely lengthy (i.e. >100 chunks); furthermore, since we can only call on the embedding model once every 60 seconds to not exceed the rate limit, storing the entire was very time costly. To create a more efficient version of the embedding process, I used Beautiful Soup's text parser to extract out only the code chunks demonstrated in the programming guide. This ended up being only 19 chunks of text and could be stored a lot faster. 

```Python
    bs4_strainer = bs4.SoupStrainer(class_=("highlight-c++ notranslate"))
    loader = WebBaseLoader(
        web_paths=("https://docs.nvidia.com/cuda/cuda-c-programming-guide/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    reduced_doc = loader.load()
```

To complete the process, we split the document into smaller chunks, then add them into the embedding model's vector store one chunk at a time; this is to prevent going over the rate limits of the embedding model.

```Python
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    all_splits = text_splitter.split_documents(docs)

    vector_store = InMemoryVectorStore(embeddings)
    batch_size = 16

    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i+batch_size]
        _ = vector_store.add_documents(documents=batch)
        total_batches = (len(all_splits) + batch_size - 1) // batch_size
        if i + batch_size < len(all_splits):
            time.sleep(60)
```

Next, I created the retrieve-and-generate application using the chat model. To do so, I utilized LangChain's LangGraph library to define the states and steps of the app: analyze, retrieve, and generate. 

```Python
    class Search(TypedDict):
        """Search query."""
        query: Annotated[str, ..., "A well-formed search query to run against the vector store."]

    prompt = hub.pull("rlm/rag-prompt")

    class State(TypedDict):
        question: str
        query: Search
        context: List[Document]
        answer: str

    def analyze_query(state: State):
        print("-> Analyzing Query...")
        structured_llm = llm.with_structured_output(Search)
        query_object = structured_llm.invoke(state["question"])
        print(f"   Generated Query: {query_object['query']}")
        return {"query": query_object}

    def retrieve(state: State):
        print("-> Retrieving Documents...")
        query_str = state["query"]["query"]
        retrieved_docs = vector_store.similarity_search(query_str, k=4)
        print(f"   Retrieved {len(retrieved_docs)} documents.")
        return {"context": retrieved_docs}

    def generate(state: State):
        print("-> Generating Answer...")
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    graph_builder = StateGraph(State)
    graph_builder.add_node("analyze_query", analyze_query)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "analyze_query")
    graph_builder.add_edge("analyze_query", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph = graph_builder.compile()
    print("\n RAG graph compiled.")
```

Finally, I wrote a specific prompt to test the model on GEMM code generation. Initially, I asked only for a high-performing GEMM kernel, but the model would sometimes describe the process without code or refer me to cuBLAS/CUTLASS. Therefore, I ended up revising the prompt to contain clearer instructions and more specific benchmarks. My final prompt is: "You are a highly capable programmer. With specific code and not using any built-in libraries such as CUTLASS, please implement a high-performing GEMM kernel for FP32 in CUDA, with performance similar to cuBLAS. Your code should not be simplified in any way, instead fully exploiting the capabilities of GPUs to ensure maximal performance."

# Results & Comparison

Below are the GEMM code that each model provided.

### GPT-4.1 RAG (full doc)
```C++
#define BLOCK_SIZE 32

__global__ void GEMM(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float val = 0.0f;
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < M && t * BLOCK_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && t * BLOCK_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i)
            val += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = val;
}
```

### GPT-4.1 RAG (reduced doc)

```C++
#define BLOCK_SIZE 128
#define THREAD_SIZE 8

__device__ float GetElement(Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}
__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width  = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Block and thread indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // This block computes Csub
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    float accum[THREAD_SIZE][THREAD_SIZE] = {0};

    // Loop over tiles of A and B
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Shared memory
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Each thread loads a THREAD_SIZE x THREAD_SIZE block
        for(int i=0;i<THREAD_SIZE;i++)
            for(int j=0;j<THREAD_SIZE;j++) {
                int row = ty*THREAD_SIZE + i;
                int col = tx*THREAD_SIZE + j;
                As[row][col] = GetElement(GetSubMatrix(A, blockRow, m), row, col);
                Bs[row][col] = GetElement(GetSubMatrix(B, m, blockCol), row, col);
            }

        __syncthreads();

        // Compute accumulation
        for (int k = 0; k < BLOCK_SIZE; k++) {
            for(int i=0;i<THREAD_SIZE;i++)
            for(int j=0;j<THREAD_SIZE;j++) {
                int row = ty*THREAD_SIZE + i;
                int col = tx*THREAD_SIZE + j;
                accum[i][j] += As[row][k] * Bs[k][col];
            }
        }
        __syncthreads();
    }

    // Write result
    for(int i=0;i<THREAD_SIZE;i++)
    for(int j=0;j<THREAD_SIZE;j++) {
        int row = ty*THREAD_SIZE + i;
        int col = tx*THREAD_SIZE + j;
        SetElement(Csub, row, col, accum[i][j]);
    }
}
```

### GPT-4.1 baseline

```C++
#define TILE_M 128
#define TILE_N 128
#define TILE_K 8
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Each thread computes a 8x8 tile
#define BLOCK_ROW_TILES (TILE_M / 8)
#define BLOCK_COL_TILES (TILE_N / 8)
#define THREAD_TILE_M 8
#define THREAD_TILE_N 8

// Macros for error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void gemm_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Shared memory for A and B tiles
    __shared__ float Asub[2][TILE_M * TILE_K];
    __shared__ float Bsub[2][TILE_K * TILE_N];

    // Calculate the block-row and block-column indices
    int block_tile_i = blockIdx.y;
    int block_tile_j = blockIdx.x;

    // Thread index within the block (0..255)
    int thread_id = threadIdx.x;

    // Warp and lane (for vectorization)
    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;

    // Each block computes a 128x128 tile.
    // Each thread computes an 8x8 subtile in registers.
    float acc[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    // Calculate starting row and column for this thread's tile in output C
    int thread_row = (warp_id / (TILE_N / (WARP_SIZE * THREAD_TILE_N))) * THREAD_TILE_M + (lane_id / (THREAD_TILE_N));
    int thread_col = (lane_id % (THREAD_TILE_N)) * THREAD_TILE_N;

    int global_row = block_tile_i * TILE_M + thread_row;
    int global_col = block_tile_j * TILE_N + thread_col;

    // Loop over k-tile blocks of K dimension
    int tiled_K = (K + TILE_K - 1) / TILE_K;

    for (int tile_k = 0; tile_k < tiled_K; ++tile_k) {

        int load_idx = thread_id;
        // Each thread loads several floats from A and B into shared memory
        // Load A: (TILE_M x TILE_K)
        for (int i = load_idx; i < TILE_M * TILE_K; i += THREADS_PER_BLOCK)
        {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int a_idx = (block_tile_i * TILE_M + row) * K + tile_k * TILE_K + col;
            if ((block_tile_i * TILE_M + row) < M && (tile_k * TILE_K + col) < K)
                Asub[tile_k % 2][i] = A[a_idx];
            else
                Asub[tile_k % 2][i] = 0.0f;
        }

        // Load B: (TILE_K x TILE_N)
        for (int i = load_idx; i < TILE_K * TILE_N; i += THREADS_PER_BLOCK)
        {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int b_idx = (tile_k * TILE_K + row) * N + (block_tile_j * TILE_N + col);
            if ((tile_k * TILE_K + row) < K && (block_tile_j * TILE_N + col) < N)
                Bsub[tile_k % 2][i] = B[b_idx];
            else
                Bsub[tile_k % 2][i] = 0.0f;
        }

        // Barrier to make sure tile is loaded
        __syncthreads();

        // Compute on the tile
        for (int k = 0; k < TILE_K; ++k) {
#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    int row = thread_row + i;
                    int col = thread_col + j;
                    if (row < TILE_M && col < TILE_N) {
                        float a_float = Asub[tile_k % 2][(row) * TILE_K + k];
                        float b_float = Bsub[tile_k % 2][k * TILE_N + (col)];
                        acc[i][j] += a_float * b_float;
                    }
                }
            }
        }
        // Sync before next tile
        __syncthreads();
    }

    // Write the tile out to C (with alpha/beta)
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            int row = global_row + i;
            int col = global_col + j;
            if (row < M && col < N) {
                float c_val = C[row * N + col];
                C[row * N + col] = alpha * acc[i][j] + beta * c_val;
            }
        }
    }
}

// Host-side GEMM invocation
void my_gemm(
    const float *A_d, const float *B_d, float *C_d,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f)
{
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((N + TILE_N - 1) / TILE_N,
                  (M + TILE_M - 1) / TILE_M, 1);

    gemm_kernel<<<grid_dim, block_dim>>>(
        A_d, B_d, C_d,
        M, N, K,
        alpha, beta);

    CUDA_CHECK(cudaGetLastError());
}
```

### GPT-4o RAG (full doc)
GPT-4o does not give any code, instead providing this explanation: "The provided code implements a matrix multiplication kernel that uses shared memory to improve performance by blocking computations into sub-matrices and reducing global memory accesses. Each thread block calculates one sub-matrix of the result and leverages shared memory for faster data access within square sub-matrix computations. While it is optimized for shared memory utilization, achieving performance on par with cuBLAS would require further tuning, such as warp-level optimizations, usage of tensor cores (if available), and greater exploitation of the GPU's architectural specifics."

### GPT-4o RAG (reduced doc)
Again, GPT-4o does not provide code, instead giving a rationale. It says, "I don’t know. Writing a high-performing FP32 GEMM kernel in CUDA comparable to cuBLAS requires intricate optimization and detailed analysis of GPU architecture, none of which is provided in the context above."

### GPT-4o baseline
```C++
#include <cuda_runtime.h>
#include <iostream>

// Tile size for shared memory
#define BLOCK_SIZE 16

// CUDA Kernel for General Matrix-Matrix Multiplication (GEMM)
__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // Shared memory tiles for blocks
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];
    
    // Thread/block coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    // Accumulator for C value
    float Cvalue = 0.0f;

    // Iterate through tiles of A and B matrices
    for (int t = 0; t < (K - 1) / BLOCK_SIZE + 1; t++) {
        // Load a tile of A and B into shared memory
        if (row < M && t * BLOCK_SIZE + tx < K)
            Asub[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        else
            Asub[ty][tx] = 0.0f;

        if (t * BLOCK_SIZE + ty < K && col < N)
            Bsub[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        else
            Bsub[ty][tx] = 0.0f;

        // Ensure all threads in the block have loaded their data
        __syncthreads();

        // Compute on the tile
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += Asub[ty][k] * Bsub[k][tx];
        }

        // Wait for all threads to complete computation before loading next tile
        __syncthreads();
    }

    // Write the computed value to global memory
    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

// Host function to allocate memory and invoke the kernel
void gemm(float* A, float* B, float* C, int M, int N, int K) {
    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N - 1) / BLOCK_SIZE + 1, (M - 1) / BLOCK_SIZE + 1);

    // Launch the kernel
    gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Copy the result back from device to host
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Utility function to initialize matrices and benchmark the kernel
void run_gemm_example() {
    const int M = 512; // Rows in A and C
    const int K = 512; // Columns in A, Rows in B
    const int N = 512; // Columns in B and C

    // Allocate host memory
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];

    // Initialize matrices A and B
    for (int i = 0; i < M * K; i++) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Run GEMM
    gemm(A, B, C, M, N, K);

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;
}

int main() {
    run_gemm_example();
    return 0;
}
```


# Discussion
Structurally, I find that the RAG-based implementations are much more simple compared to the baseline implementations. They seem to be relatively straightforward and involve a minimal amount of steps, compared to the baseline implementations (which seems to generate multiple steps, including checking functions, and include substantially more code for each step). The baseline implementations also usually provide the execution code, while the RAG-based models only focus on an implementation of the GEMM kernel itself. This might be due to the CUDA guide document only providing examples of kernels, rather than the execution itself. However, I haven't yet tested the runtime performance of the implementations, so I'm still unsure if RAG-based implementations are better or worse performing than their baseline counterparts.

I also found that the RAG-based models are more "pessimistic" about their ability to generate a GEMM implementation that reaches the performance level of cuBLAS; in other words, they are more likely to deem the high-performance tasks too difficult and refuse to try altogether, compared to the baseline models which always responds with an implementation. I had to tweak my prompt accordingly (i.e. not mention specific benchmarking levels), but even then, GPT-4o does not seem to provide any code. I'm not sure if this is due to the structure of the retrieval-generation app I built, my prompt's wording, or some other factor.


# References
- Langchain RAG App Tutorial, https://python.langchain.com/docs/tutorials/rag/#orchestration
- NVIDIA CUDA documentation, https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-guidelines
- Guu et al., "REALM: Retrieval-Augmented Language Model Pre-Training", https://arxiv.org/abs/2002.08909
- Borgeaud et al., "Improving Language Models by Retrieving from Trillions of Tokens", https://proceedings.mlr.press/v162/borgeaud22a.html
- Su et al., "EVOR: Evolving Retrieval for Code Generation", https://aclanthology.org/2024.findings-emnlp.143.pdf
