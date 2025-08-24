import pickle
from langchain_community.document_loaders import WebBaseLoader
from src import rag_system
from src import config

def create_and_save_store():
    """
    Performs the one-time, expensive process of loading the document,
    creating the vector store, and saving it to a local file.
    """
    vector_store_cache_path = "vector_store.pkl"
    
    print("Building a new vector store. This will take a long time...")
    
    # Load the document
    loader = WebBaseLoader(
        web_paths=("https://docs.nvidia.com/cuda/cuda-c-programming-guide/",)
    )
    full_doc = loader.load()
    
    # Create the vector store using your existing function
    vector_store = rag_system.store_docs(full_doc)
    
    # Temporarily remove the un-pickleable embedding client before saving
    embedding_client = vector_store.embedding
    vector_store.embedding = None # This makes the object serializable

    # Save the vector store to the cache file
    print(f"Saving new vector store to {vector_store_cache_path}...")
    with open(vector_store_cache_path, "wb") as f:
        pickle.dump(vector_store, f)
    
    print("Vector store created and saved successfully as vector_store.pkl")
    print("You can now commit this file to Git LFS.")

if __name__ == "__main__":
    create_and_save_store()