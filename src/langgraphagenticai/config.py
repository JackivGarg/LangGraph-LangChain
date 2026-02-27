"""Configuration and constants for the university chatbot."""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

VECTOR_CATEGORIES = [
    "admissions", "programs", "hostel",
    "placements", "policies", "general"
]
