import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ChromaDB Directory
CHROMA_DB_PATH = "./chroma_db"

# Allowed Models
GROQ_MODELS = ["gemma2-9b-it","llama-3.3-70b-versatile","deepseek-r1-distill-llama-70b", "llama-guard-3-8b"]

