# Overview
Retrieval-augmented generation has been shown to improve the capabilities of LLMs in myriad of question-answering metrics (Guu et al. 2020, Borgeaud et al. 2022), as well in general code generation (Su et al. 2024). However, its effects on optimizing code generation in terms of system performance has been so far unclear. To explore its capabilities in creating high-performing CUDA code in particular, I created a simple RAG model which retrieves information from NVIDIA's CUDA manual. Then, I tasked an LLM trained under this model with creating a high-performing GEMM and compared results with its baseline (the LLM without an RAG application). 

More information and testing results are available at `writeup.md`.


# Setup instructions

This project is hosted on NVIDIA GPU and tested with CUDA Toolkit 12.4. Support for `nvcc` is also required.

To use, clone the Github repository and install the necessary requirements via pip.

```
git clone https://github.com/hanjingwang9/rag_gemm.git
cd rag_gemm
pip install -r requirements.txt
```

You can run the entire pipeline from the command line, specifying which family of models you want to test.

### To run the entire pipeline on the OpenAI models (e.g., gpt-4.1, gpt-4o):

```
python -m src.main --model openai
```

### To run the entire pipeline on the LLama model (e.g., gpt-4.1, gpt-4o):

```
python -m src.main --model llama
```
The script will generate, compile, and benchmark the CUDA kernels produced by the specified models.


# References
- Langchain RAG App Tutorial, https://python.langchain.com/docs/tutorials/rag/#orchestration
- NVIDIA CUDA documentation, https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-guidelines
- Guu et al., "REALM: Retrieval-Augmented Language Model Pre-Training", https://arxiv.org/abs/2002.08909
- Borgeaud et al., "Improving Language Models by Retrieving from Trillions of Tokens", https://proceedings.mlr.press/v162/borgeaud22a.html
- Su et al., "EVOR: Evolving Retrieval for Code Generation", https://aclanthology.org/2024.findings-emnlp.143.pdf