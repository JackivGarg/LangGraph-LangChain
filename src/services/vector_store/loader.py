import os
from langchain_community.vectorstores import FAISS
from src.core.config import VALID_CATEGORIES
from src.core.embeddings import embedding


def load_vector_store(category: str):
    """Load FAISS vector store for the given category."""
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category: {category}. Must be one of {VALID_CATEGORIES}")

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    base_dir = os.path.join(root_dir, "faiss_stores")
    save_path = os.path.join(base_dir, f"faiss_store_{category}")

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Vector store not found: {save_path}")

    return FAISS.load_local(save_path, embedding, allow_dangerous_deserialization=True)
