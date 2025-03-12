from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from config import CHROMA_DB_PATH

def create_vector_store(chunks):
    """Creates a ChromaDB vector store from document chunks."""
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_DB_PATH)
    return vector_store

