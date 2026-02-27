import os
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import sys, time
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
HUGGINGFACE_KEY=os.getenv("HUGGINGFACE_KEY")
embedding=HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HUGGINGFACE_KEY,
    model=EMBEDDING_MODEL
)
def reload(txt_file,saveto):
    try:
        with open(txt_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print("Error: content.txt not found!")

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    vector=FAISS.from_texts(splitter.split_text(raw_text), embedding)
    vector.save_local(saveto)

store={}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Checks if session_id exists in store.
    If yes, returns existing history. If no, creates new.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def typewriter(text, delay=0.01):
    """Print text like a typewriter (char-by-char)."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()
