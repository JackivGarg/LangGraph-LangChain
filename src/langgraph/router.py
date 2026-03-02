from langchain_core.output_parsers import StrOutputParser

from src.llm import llm
from src.langgraph.prompts import router_template, generate_template
from src.langgraph.states.states import Route1
from src.langchain.history import get_session_history
from src.vectorstores import load_vector_store
from src.config import VALID_CATEGORIES


def langgraph_route_and_respond(query: str, session_id: str):
    history = get_session_history(session_id)
    history_messages = history.messages
    
    history_str = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in history_messages
    )
    
    router_chain = router_template | llm.with_structured_output(Route1)
    route_result = router_chain.invoke({
        "user_input": query,
        "history": history_str,
        "context": ""
    })
    
    category = route_result.category if route_result.action == "VECTOR_STORE" else "general"
    
    if category not in VALID_CATEGORIES:
        category = "general"
    
    try:
        vs = load_vector_store(category)
        docs = vs.similarity_search(query, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        context = f"Could not retrieve from {category}: {str(e)}"
    
    generate_chain = generate_template | llm | StrOutputParser()
    
    full_response = ""
    for chunk in generate_chain.invoke({
        "context": context,
        "history": history_str,
        "user_input": query
    }):
        full_response += chunk
        yield chunk
    
    history.add_user_message(query)
    history.add_ai_message(full_response)
