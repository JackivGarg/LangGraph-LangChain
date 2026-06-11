import time
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

from src.core.llm import llm
from src.services.langchain_prompts import LANCHAIN_PROMPT
from src.utils.history import get_session_history
from src.services.vector_store.loader import load_vector_store
from src.core.config import VALID_CATEGORIES
from src.services.rewrite_service import rewrite_query
from src.agents.states.states import Route1
from src.agents.prompts import router_template, document_grader_template
from src.services.tavily_service import tavily_search as _tavily_search, grade_docs as _grade_docs


def route_to_category(query: str, history_str: str):
    try:
        router_chain = router_template | llm.with_structured_output(Route1)
        route_result = router_chain.invoke({
            "user_input": query,
            "history": history_str,
            "context": ""
        })
        category = route_result.category if route_result.action == "VECTOR_STORE" else "general"
        if category not in VALID_CATEGORIES or category is None:
            category = "general"
        return category, route_result.action
    except Exception:
        return "general", "VECTOR_STORE"


def search_specific_vectorstore(query: str, category: str, k: int = 3):
    try:
        vs = load_vector_store(category)
        docs = vs.similarity_search(query, k=k)
        return docs
    except Exception:
        return []


def langchain_mode(query: str, session_id: str):
    start_time = time.time()

    history = get_session_history(session_id)
    history_messages = history.messages
    history_str = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in history_messages
    )

    original_query = query
    interpreted_query = rewrite_query(query, history_str)

    # --- Step 1: Route ---
    category, action = route_to_category(interpreted_query, history_str)
    is_relevant = "yes"

    # --- Step 2: Retrieve or Search ---
    if action == "TAVILY_SEARCH":
        # Router decided this needs a web search directly
        context = _tavily_search(interpreted_query)
        is_relevant = "yes"
    else:
        # Try Vector Store first
        docs = search_specific_vectorstore(interpreted_query, category, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)

        # --- Step 3: Grade relevance ---
        is_relevant = _grade_docs(llm, document_grader_template, interpreted_query, context)

        # --- Step 4: Fallback to Tavily if not relevant ---
        if is_relevant == "no":
            tavily_context = _tavily_search(interpreted_query)
            if tavily_context:
                context = tavily_context
                action = "TAVILY_SEARCH (fallback)"

    # If context is still empty after all attempts, give a graceful message
    if not context.strip():
        full_response = "Sorry, I couldn't find any relevant information for your question."
        yield full_response
        history.add_user_message(query)
        history.add_ai_message(full_response)
        yield {"__stats__": {
            "original_query": original_query,
            "interpreted_query": interpreted_query,
            "action": action,
            "category": category,
            "is_relevant": is_relevant,
            "response_time": round(time.time() - start_time, 2),
            "word_count": len(full_response.split()),
            "char_count": len(full_response)
        }}
        return

    # --- Step 5: Generate ---
    parser = StrOutputParser()
    pro_chain = (
        {
            "context": lambda x: context,
            "user_input": itemgetter("user_input"),
            "history": itemgetter("history")
        }
        | LANCHAIN_PROMPT
        | llm
        | parser
    )

    full_response = ""
    for chunk in pro_chain.stream({"user_input": interpreted_query, "history": history_str}):
        full_response += chunk
        yield chunk

    history.add_user_message(query)
    history.add_ai_message(full_response)

    response_time = time.time() - start_time

    yield {"__stats__": {
        "original_query": original_query,
        "interpreted_query": interpreted_query,
        "action": action,
        "category": category,
        "is_relevant": is_relevant,
        "response_time": round(response_time, 2),
        "word_count": len(full_response.split()),
        "char_count": len(full_response)
    }}

