import os
import torch
import streamlit as st
from utils.file_processing import save_uploaded_file, download_pdf_from_url, process_pdf, process_html
from utils.vector_store import create_vector_store
from utils.chat_handler import retrieve_documents, generate_response, chat_memory
from config import GROQ_MODELS

torch.classes.__path__ = []


# Streamlit UI Config
st.set_page_config(page_title="Conversational RAG with PDF Upload & URL", layout="wide")
st.title("📚 Conversational RAG Chatbot")

# Sidebar: Model Selection
st.sidebar.header("🔄 Choose LLM Model")
model_choice = st.sidebar.selectbox("Select a Groq model:", GROQ_MODELS, index=0)
st.sidebar.success(f"✅ Using {model_choice}")

# Answer Type Selection (Short/Long)
# answer_type = st.radio("Select Answer Type:", ["Short", "Long"], horizontal=True)

# Session State Initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "current_webpage" not in st.session_state:
    st.session_state.current_webpage = None

# File Upload Section
st.subheader("📂 Upload a PDF or Provide a URL")
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
pdf_url = st.text_input("🔗 Or enter a PDF URL (e.g., arXiv link):")
html_url = st.text_input("🔗 enter web page link (e.g., wikipedia):")

if uploaded_file or pdf_url:
    new_pdf_source = uploaded_file.name if uploaded_file else pdf_url
    
    
    if st.session_state.current_pdf != new_pdf_source:
        st.session_state.messages = []
        chat_memory.clear()
        st.session_state.vector_store = None
        st.session_state.current_pdf = new_pdf_source

        if uploaded_file:
            st.success(f"📄 Uploaded: {uploaded_file.name}")
            file_path = save_uploaded_file(uploaded_file)
        elif pdf_url:
            st.success(f"📥 Downloading from URL: {pdf_url}")
            file_path = download_pdf_from_url(pdf_url)

        # Process PDF
        with st.spinner("📖 Processing document..."):
            chunks = process_pdf(file_path)
            st.session_state.vector_store = create_vector_store(chunks)
            

        st.success("✅ Document indexed in ChromaDB!")
        os.remove(file_path)  # Cleanup


if html_url:
    new_html_url   = html_url
    if st.session_state.current_webpage != new_html_url:
        st.session_state.messages = []  # Reset chat history
        chat_memory.clear()
        st.session_state.vector_store = None  # Clear previous vector store
        st.session_state.current_webpage = new_html_url  # Update with new webpage URL

        if html_url:
            st.success(f"📥 Downloading from web page: {html_url}")

        # Process new HTML file
        with st.spinner("📖 Processing document..."):
            chunks = process_html(html_url)
            st.session_state.vector_store = create_vector_store(chunks)

        st.success("✅ Document indexed in ChromaDB!")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
query = st.chat_input("💬 Ask a question:")

if query and st.session_state.vector_store:
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🔍 Searching for relevant context..."):
        retrieved_docs = retrieve_documents(st.session_state.vector_store, query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    if context:
        with st.spinner("🤖 Generating response..."):
            response_text = generate_response(model_choice,context, query)

        response_container = st.chat_message("assistant")
        with response_container:
            response_markdown = st.empty()
            response_markdown.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Store conversation in memory
        chat_memory.save_context({"input": query}, {"output": response_text})
