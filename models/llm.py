from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

def get_vectorstore_from_local():
    persist_directory = "vectorstore"  # or whatever folder you're using
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vectorstore directory '{persist_directory}' not found.")

    embedding = HuggingFaceEmbeddings()
    vectordb = FAISS.load_local(persist_directory, embedding, allow_dangerous_deserialization=True)
    return vectordb
