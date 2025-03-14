from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_groq import ChatGroq
from groq import Groq
from config import GROQ_API_KEY
import re
from sentence_transformers import CrossEncoder

# Load reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# Initialize Conversation Memory (stores last 50 messages)
chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)#, max_len=50)

# llm = ChatGroq(model=model_name, api_key=GROQ_API_KEY)

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

def expand_query_groq(user_query):
    """Expands a user query using Groq's Mixtral-8x7B model."""
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",  # You can use "llama3-70b-8192" as well
        messages=[{"role": "user", "content": f"""
            If the following query is already complete and well-formed, return it as is.
            If the query is vague or incomplete, rephrase it to be more specific while preserving its original meaning.
            Do not add extra information or change the context.

            Query: '{user_query}'
            
            Output:
            """}]
    )
    
    return response.choices[0].message.content

# Example Usage
# original_query = "climate effects"
# expanded_query = expand_query_groq(original_query)
# print("Expanded Query:", expanded_query)


def retrieve_documents(vector_store, query, k=10):
    """Fetches relevant documents from vector store."""
    expanded_query = expand_query_groq(query)
    # print (f"original query : {query}")
    # print (f"expanded query : {expanded_query}")
    retrieved_docs = vector_store.similarity_search(expanded_query, k=k)
    query_doc_pairs = [(query, doc.page_content) for doc in retrieved_docs]
    # Compute relevance scores
    scores = reranker.predict(query_doc_pairs)
    # Sort retrieved documents by reranking scores (descending)
    reranked_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True, key=lambda x: x[0])]
    reranked_docs = reranked_docs[:5]


    # return reranked_docs#sorted(retrieved_docs, key=lambda doc: doc.metadata.get("chunk_id", float("inf")))
    return  sorted(reranked_docs, key=lambda doc: doc.metadata.get("chunk_id", float("inf")))

def generate_response(model_name, context, query):
    """Generates response from LLM using retrieved context."""
    llm = ChatGroq(model=model_name, api_key=GROQ_API_KEY) # temperature=0, max_tokens=max_tokens
    
    chat_history = chat_memory.load_memory_variables({}).get("chat_history", [])
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    prompt = f"""
    You are an AI assistant that only answers questions based on the retrieved documents. If the information is not found in the provided context, say "I don't know." Do not generate answers beyond the given data.



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

    # Remove the <think> section from the response
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    return response_text

