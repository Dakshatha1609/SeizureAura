from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import os


def get_vectorstore_from_local(path="data/epilepsy_knowledgebase.txt"):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

       
        db = FAISS.from_texts(texts, embeddings)
        print("✅ Vectorstore built successfully from local knowledge base.")


        return db
        

    except Exception as e:
        import traceback
        print(" Error during get_vectorstore_from_local():")
        print(traceback.format_exc())
        raise ConnectionError(" Failed to build vector store from local knowledge.")
