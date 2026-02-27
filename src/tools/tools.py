import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.tools import create_retriever_tool

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
# --------------------
# Embeddings
# --------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")

embedding = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HUGGINGFACE_KEY,
    model=EMBEDDING_MODEL
)

# --------------------
# Vector store creator
# --------------------
def create_vector_store(txt_file: str, save_dir: str):
    with open(txt_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
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

    txt_files = [
        "general.txt",
        "hostel.txt",
        "placements.txt",
        "policies.txt",
        "programs.txt",
        "admissions.txt"
    ]

    for txt_file in txt_files:
        name = os.path.splitext(os.path.basename(txt_file))[0]
        save_path = f"faiss_store_{name}"

        vectorstore = create_vector_store(txt_file, save_path)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        tool = create_retriever_tool(
            retriever=retriever,
            name=f"{name}_retriever",
            description=f"Search information from {txt_file}"
        )

        tools.append(tool)

    return tools


# --------------------
# Load vector store by category
# --------------------
VALID_CATEGORIES = ["admissions", "programs", "hostel", "placements", "policies", "general"]


def load_vector_store(category: str):
    """Load FAISS vector store for the given category."""
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category: {category}. Must be one of {VALID_CATEGORIES}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, f"faiss_store_{category}")

    # programs may be nested inside faiss_store_general
    if not os.path.exists(save_path) and category == "programs":
        alt_path = os.path.join(base_dir, "faiss_store_general", "faiss_store_programs")
        if os.path.exists(alt_path):
            save_path = alt_path

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Vector store not found: {save_path}")

    return FAISS.load_local(save_path, embedding, allow_dangerous_deserialization=True)
