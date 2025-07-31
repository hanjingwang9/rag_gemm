# Overview
Retrieval-augmented generation has been shown to improve the capabilities of LLMs in myriad of question-answering metrics (Guu et al. 2020, Borgeaud et al. 2022), as well in general code generation (Su et al. 2024). However, its effects on optimizing code generation in terms of system performance has been so far unclear. To explore its capabilities in creating high-performing CUDA code in particular, I created a simple RAG model which retrieves information from NVIDIA's CUDA manual. Then, I tasked an LLM trained under this model with creating a high-performing GEMM and compared results with its baseline (the LLM without an RAG application). 

## References
- Langchain RAG App Tutorial, https://python.langchain.com/docs/tutorials/rag/#orchestration
- NVIDIA CUDA documentation, https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-guidelines
- Guu et al., "REALM: Retrieval-Augmented Language Model Pre-Training", https://arxiv.org/abs/2002.08909
- Borgeaud et al., "Improving Language Models by Retrieving from Trillions of Tokens", https://proceedings.mlr.press/v162/borgeaud22a.html
- Su et al., "EVOR: Evolving Retrieval for Code Generation", https://aclanthology.org/2024.findings-emnlp.143.pdf