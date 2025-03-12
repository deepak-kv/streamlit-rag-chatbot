# 🦙 Streamlit RAG Chatbot with Groq & ChromaDB

A **Retrieval-Augmented Generation (RAG) Chatbot** built with **Streamlit, LangChain, ChromaDB, and Groq**.  
This chatbot can process **PDF documents** (from local files or URLs like arXiv), embed them using **Ollama embeddings**, store them in **ChromaDB**, and generate responses using **Groq’s Llama3 models**.

---

## 📂 Project Structure

```
streamlit-rag-chatbot/
│── app.py                     # Main Streamlit app
│── components/
│   ├── chat.py                # Chat interface and logic
│   ├── document_loader.py      # Load and split PDF documents
│   ├── vector_store.py         # Create and manage ChromaDB storage
│   ├── model_selector.py       # Dropdown for selecting different Groq models
│   ├── memory.py               # Manages conversation history (ConversationBufferMemory)
│── data/                       # Store uploaded PDFs (optional)
│── requirements.txt            # Required dependencies
│── .gitignore                  # Ignore unnecessary files
│── README.md                   # Documentation
```

---

## 🚀 Features
✅ **Upload PDFs** or provide a **URL (e.g., ArXiv link)** to load documents  
✅ **Chunk & Embed** text using **Ollama Embeddings**  
✅ **Store & Retrieve** documents using **ChromaDB**  
✅ **Query with Llama3** (via **Groq API**)  
✅ **Chat Memory**   
✅ **Choose Llama3 Model** (`llama3-8b`, `llama3-70b`, etc.)  
✅ **Streamlit UI** for an interactive chat experience  

---

## 🛋️ Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/streamlit-rag-chatbot.git
cd streamlit-rag-chatbot
```

### **2️⃣ Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Set Up Environment Variables**
Create a `.env` file in the root directory and add:
```plaintext
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```
This will launch the chatbot in your browser at **`http://localhost:8501`**.

---

## 🛠️ Usage Instructions
1️⃣ **Upload a PDF** or provide an **ArXiv URL**  
2️⃣ **The app will process the document, create embeddings, and store them in ChromaDB**  
3️⃣ **Ask questions based on the document's content**  
4️⃣ **Choose a different model if needed**  
5️⃣ **Continue chatting (memory stores last 50 messages)**  

---

## 🛠️ Technologies Used
- **Python**
- **Streamlit** → UI for chatbot
- **LangChain** → LLM framework
- **ChromaDB** → Vector storage
- **Ollama** → Embeddings
- **Groq API** → Llama3-based responses
- **PyPDF** → Extract text from PDFs
- **Requests** → Load PDFs from URLs
- **ConversationBufferMemory** → Maintains chat history

---

## 📌 To-Do (Future Enhancements)
- [ ] Add support for multiple document uploads  
- [ ] Improve UI (Dark mode, better formatting)  
- [ ] Support other embedding models (e.g., OpenAI, Cohere)  
- [ ] Enable multi-user chat memory  

---

## 🤝 Contributions
Feel free to **fork** this repo, create a **pull request**, or open an **issue** for feature suggestions!

---

