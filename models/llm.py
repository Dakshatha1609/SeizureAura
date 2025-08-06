from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY

def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    try:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment")

        groq_model = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama3-8b-8192"
        )
        return groq_model

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")
