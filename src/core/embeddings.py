import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from src.core.config import EMBEDDING_MODEL

HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")

embedding = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HUGGINGFACE_KEY,
    model=EMBEDDING_MODEL
)
