import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
