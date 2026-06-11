"""
Shared Tavily web search utility.
Uses the new langchain_tavily package (langchain-tavily>=0.1.0).
Falls back gracefully if the key/package is missing.
"""
import os
from pydantic import BaseModel, Field
from typing import Literal

# --- New package import (replaces deprecated TavilySearchResults) ---
try:
    from langchain_tavily import TavilySearch
    _tool = TavilySearch(max_results=5)
    _tavily_available = True
except Exception as e:
    _tavily_available = False
    _tool = None


def tavily_search(query: str) -> str:
    """
    Run a Tavily web search and return a clean, formatted context string.
    Returns an empty string if Tavily is unavailable or errors out.
    """
    if not _tavily_available or _tool is None:
        return ""
    try:
        raw = _tool.invoke({"query": query})

        # TavilySearch returns a dict with a "results" key (list of dicts)
        # Each result has: title, content, url
        if isinstance(raw, dict) and "results" in raw:
            results = raw["results"]
        elif isinstance(raw, list):
            results = raw
        else:
            return str(raw)  # Last resort: just stringify

        if not results:
            return ""

        lines = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            lines.append(f"[{title}]\n{content}\nSource: {url}")

        return "\n\n".join(lines)

    except Exception as e:
        return ""


# --- Grader model (shared) ---
class _Grade(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question, 'no' otherwise"
    )


def grade_docs(llm, document_grader_template, question: str, context: str) -> str:
    """
    Use the LLM + document_grader_template to score whether context is relevant.
    Returns 'yes' or 'no'. Defaults to 'yes' on any error to avoid unnecessary
    extra web searches.
    """
    if not context.strip():
        return "no"
    try:
        chain = document_grader_template | llm.with_structured_output(_Grade)
        result = chain.invoke({"question": question, "document": context})
        return result.binary_score
    except Exception:
        return "yes"
