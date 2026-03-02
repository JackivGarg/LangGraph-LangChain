from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from operator import itemgetter

from src.llm import llm
from src.langchain.prompts import LANCHAIN_PROMPT
from src.langchain.history import get_session_history
from src.vectorstores import load_vector_store
from src.config import VALID_CATEGORIES
from src.embeddings import embedding


def search_all_vectorstores(query: str, k: int = 3):
    all_texts = []
    all_metadatas = []
    
    for category in VALID_CATEGORIES:
        try:
            vs = load_vector_store(category)
            docs = vs.similarity_search(query, k=k)
            for doc in docs:
                all_texts.append(doc.page_content)
                all_metadatas.append(doc.metadata)
        except Exception as e:
            continue
    
    if all_texts:
        return FAISS.from_texts(all_texts, embedding, metadatas=all_metadatas)
    return None


def langchain_mode(query: str, session_id: str):
    history = get_session_history(session_id)
    history_messages = history.messages
    
    history_str = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in history_messages
    )
    
    vs = search_all_vectorstores(query, k=3)
    
    if vs is None:
        yield "Sorry, I couldn't find any relevant information."
        return
    
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    
    parser = StrOutputParser()
    
    pro_chain = (
        {
            "context": itemgetter("user_input") | retriever,
            "user_input": itemgetter("user_input"),
            "history": itemgetter("history")
        }
        | LANCHAIN_PROMPT
        | llm
        | parser
    )
    
    full_response = ""
    for chunk in pro_chain.invoke({"user_input": query, "history": history_str}):
        full_response += chunk
        yield chunk
    
    history.add_user_message(query)
    history.add_ai_message(full_response)
