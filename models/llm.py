# models/llm.py

from langchain_groq import ChatGroq
from config import GROQ_API_KEY  # ✅ Corrected import

def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="mixtral-8x7b-32768"
    )
