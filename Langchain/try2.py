# try2.py
import os
# ... all your imports ...
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
import common
import store_model_embeddings
import templates

# --- initialization: run once on import (cheap) ---
db_2 = FAISS.load_local(
    "vector_created_PRO", 
    store_model_embeddings.embedding,
    allow_dangerous_deserialization=True
)

# Build reusable objects (chain, retriever factory, etc.)
def build_pro_chain():
    parser = StrOutputParser()
    retriver = db_2.as_retriever(search_kwargs={"k":6})
    pro_prompt = PromptTemplate(
        template=templates.pro_template_str,
        input_variables=["context", "history", "user_input"]
    )

    from operator import itemgetter
    pro_chain = (
        {
            "context": itemgetter("user_input") | retriver,
            "user_input": itemgetter("user_input"),
            "history": itemgetter("history")
        }
        | pro_prompt
        | store_model_embeddings.model
        | parser
    )

    final_pro_chain = RunnableWithMessageHistory(
        pro_chain,
        common.get_session_history,
        input_messages_key="user_input",
        history_messages_key="history"
    )
    return final_pro_chain

# create the chain once
_FINAL_PRO_CHAIN = build_pro_chain()

# --- Export a simple function that handles ONE query and returns string ---
def pro_reply(user_query: str, session_id: str = "chat1") -> str:
    """
    Handle a single user query and return the model response as string.
    Safe to call from Streamlit / other modules.
    """
    if not user_query:
        return ""

    config = {"configurable": {"session_id": session_id}}
    response = _FINAL_PRO_CHAIN.invoke({"user_input": user_query}, config=config)
    return str(response)

# --- If user runs this file directly, run an interactive loop (dev only) ---
if __name__ == "__main__":
    print("Starting interactive pro bot. Type 'exit' to quit.")
    while True:
        q = input("YOU: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("AI: ", end="")
        common.typewriter(pro_reply(q), delay=0.01)
