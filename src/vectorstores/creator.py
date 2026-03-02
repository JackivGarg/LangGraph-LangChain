import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool
from src.embeddings import embedding
from src.config import VALID_CATEGORIES


def create_vector_store(txt_file: str, save_dir: str):
    with open(txt_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    texts = splitter.split_text(raw_text)

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=[{"source": txt_file}] * len(texts)
    )

    vectorstore.save_local(save_dir)
    return vectorstore


def build_tools():
    tools = []

    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    faiss_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "faiss_stores")

    for category in VALID_CATEGORIES:
        txt_file = os.path.join(base_dir, f"{category}.txt")
        save_path = os.path.join(faiss_dir, f"faiss_store_{category}")

        vectorstore = create_vector_store(txt_file, save_path)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        tool = create_retriever_tool(
            retriever=retriever,
            name=f"{category}_retriever",
            description=f"Search information from {txt_file}"
        )

        tools.append(tool)

    return tools
