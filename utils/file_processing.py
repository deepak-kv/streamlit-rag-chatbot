import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def save_uploaded_file(uploaded_file, temp_dir="temp_dir"):
    """Saves uploaded file locally and returns file path."""
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def download_pdf_from_url(pdf_url, temp_dir="temp_dir"):
    """Downloads a PDF from a URL and saves it locally."""
    os.makedirs(temp_dir, exist_ok=True)
    file_name = pdf_url.split("/")[-1]
    file_path = os.path.join(temp_dir, file_name)

    response = requests.get(pdf_url, stream=True)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    else:
        raise ValueError("Failed to download PDF. Check the URL.")

def process_pdf(file_path, chunk_size=1000, chunk_overlap=50):
    """Loads and splits PDF into chunks."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    # Assign chunk IDs for sorting
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks
