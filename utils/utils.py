from models.embeddings import get_vectorstore_from_local
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import RetrievalQA
from models.llm import get_chatgroq_model
import requests

def answer_from_local_knowledge(query: str):
    try:
        db = get_vectorstore_from_local()
        retriever = db.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=get_chatgroq_model(),
            retriever=retriever,
            return_source_documents=False
        )

        result = qa_chain.run(query)
        if not result or "I don't know" in result:
            raise ValueError("No answer found in local documents.")
        return result

    except Exception as e:
        return f"Error during RAG: {str(e)}"

def search_web(query: str):
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            related = data.get("RelatedTopics", [])
            if related:
                results = [topic.get("Text", "") for topic in related[:2]]
                return "\n\n".join(results)
            else:
                return "No relevant results found."
        else:
            return f"Web search failed with status: {response.status_code}"
    except Exception as e:
        return f"Web search error: {e}"
