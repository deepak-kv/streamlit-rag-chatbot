from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import CHROMA_DB_PATH
import os
import shutil
import chromadb

# chromadb.api.client.SharedSystemClient.clear_system_cache()

def create_vector_store(chunks):
    """Creates a ChromaDB vector store from document chunks."""
        # Remove existing store if it exists
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)


    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    vector_store = Chroma.from_documents(chunks, embedding_model, persist_directory=CHROMA_DB_PATH)
    
    return vector_store

def get_vector_store():
    vector_store = Chroma(embedding_function=OllamaEmbeddings(model="nomic-embed-text"), persist_directory=CHROMA_DB_PATH)
    return vector_store


