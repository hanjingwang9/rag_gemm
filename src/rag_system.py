import time
from typing import List

from langchain import hub
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

def get_llm(model_family, deployment_name):
    """Initializes LangChain LLM client based on the specified model family."""
    if model_family == "openai":
        print(f"Initializing AzureChatOpenAI for deployment: {deployment_name}")
        return AzureChatOpenAI(
            azure_endpoint=config.openai_endpoint,
            api_key=config.api_key,
            azure_deployment=deployment_name,
            api_version=config.openai_api_version,
        )
    elif model_family == "llama":
        print(f"Initializing AzureChatOpenAI for Llama deployment: {deployment_name}")
        return AzureChatOpenAI(
            azure_endpoint=config.llama_endpoint,
            api_key=config.api_key,
            azure_deployment=deployment_name,
            api_version=config.llama_api_version
        )
    else:
        print(f"Warning: Model family '{model_family}' not recognized. Using placeholder.")
        return None


def run_rag_gemm(model, llm, vector_store):
    """Builds RAG application using chat model."""
    print("RAG graph using configured LLM.")

    prompt = hub.pull("rlm/rag-prompt")

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str
        query: dict

    def analyze_query(state: State):
        print("-> Analyzing Query...")
        SearchSchema = TypedDict("Search", {"query": Annotated[str, "A well-formed search query."]})
        structured_llm = llm.with_structured_output(SearchSchema)
        query_object = structured_llm.invoke(state["question"])
        print(f"   Generated Query: {query_object['query']}")
        return {"query": query_object}

    def retrieve(state: State):
        print("-> Retrieving Documents...")
        query_str = state.get("query", {}).get("query", state["question"])
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

    if model == "openai":
        graph_builder.add_node("analyze_query", analyze_query)
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("generate", generate)
        graph_builder.add_edge(START, "analyze_query")
        graph_builder.add_edge("analyze_query", "retrieve")
        graph_builder.add_edge("retrieve", "generate")
    elif model == "llama":
        # Llama model is without queries
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("generate", generate)
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
    
    graph = graph_builder.compile()
    print("\n RAG graph compiled.")
    
    return graph

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