from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

# Initialize Conversation Memory (stores last 50 messages)
chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_len=50)

def retrieve_documents(vector_store, query, k=5):
    """Fetches relevant documents from vector store."""
    retrieved_docs = vector_store.similarity_search(query, k=k)
    return sorted(retrieved_docs, key=lambda doc: doc.metadata.get("chunk_id", float("inf")))

def generate_response(model_name, context, query):
    """Generates response from LLM using retrieved context."""
    llm = ChatGroq(model=model_name, api_key=GROQ_API_KEY) # temperature=0, 
    
    chat_history = chat_memory.load_memory_variables({}).get("chat_history", [])
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    prompt = f"""
    You are an AI assistant answering queries based on retrieved document context.

    Chat History (Last 50 Messages):
    {history_text}

    Relevant Context from the Document:
    {context}  

    Query: {query}

    Answer concisely using the above context.
    """
    
    response_text = ""
    for chunk in llm.stream(prompt):
        response_text += chunk.content
    return response_text

