from langchain_community.tools import DuckDuckGoSearchRun

def search_web(query: str):
    try:
        tool = DuckDuckGoSearchRun()
        result = tool.run(query)
        return result
    except Exception as e:
        return f" Web search error: {str(e)}"
