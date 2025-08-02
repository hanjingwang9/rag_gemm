import time
from typing import List

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated, TypedDict
from . import config

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

def store_docs(docs):
    """Stores document as vector embeddings using embedding model."""
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=config.embedding_endpoint,
        api_key=config.api_key,
        azure_deployment=config.embedding_deployment,
        openai_api_version=config.embedding_api_version,
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    all_splits = text_splitter.split_documents(docs)
    print(f"Split relevant sections into {len(all_splits)} chunks.")

    print(f"Successfully loaded and split the document into {len(all_splits)} chunks.")

    vector_store = InMemoryVectorStore(embeddings)
    batch_size = 16
    print(f"\nIndexing {len(all_splits)} document chunks in batches of {batch_size}...")

    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i+batch_size]

        _ = vector_store.add_documents(documents=batch)

        total_batches = (len(all_splits) + batch_size - 1) // batch_size
        print(f"  - Embedded batch {i//batch_size + 1}/{total_batches}")

        if i + batch_size < len(all_splits):
            print("  - Waiting for 60 seconds to avoid rate limit...")
            time.sleep(60)

    print("Indexing complete.")
    return vector_store

def run_rag_gemm(model, vector_store):
    """Builds RAG application using chat model."""
    llm = AzureChatOpenAI(
        azure_endpoint=config.llm_endpoint,
        api_key=config.api_key,
        azure_deployment=model,
        api_version=config.llm_api_version,
    )
    print("LLM model configured.")

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

    # Compile graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("analyze_query", analyze_query)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "analyze_query")
    graph_builder.add_edge("analyze_query", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph = graph_builder.compile()
    print("\n RAG graph compiled.")
    
    return llm, graph

def generate_code(graph, llm, prompt):
    """Generates RAG and baseline response given a specific model and prompt."""
    print("\n--- Invoking RAG system ---")
    rag_state = graph.invoke({"question": prompt})
    
    print("\n--- Invoking Baseline LLM ---")
    baseline_response = llm.invoke(prompt)
    
    return {
        "rag_answer": rag_state["answer"],
        "baseline_answer": baseline_response.content
    }