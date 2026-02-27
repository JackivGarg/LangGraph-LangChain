import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY") 
HUGGINGFACE_KEY=os.getenv("HUGGINGFACE_KEY")
model=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=GROQ_API_KEY
)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
embedding=HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HUGGINGFACE_KEY,
    model=EMBEDDING_MODEL
)
db = FAISS.load_local(
    "vector_created_2", 
    embedding,
    allow_dangerous_deserialization=True
)