"""Utility functions for the university chatbot."""

'''
👉 turning a list of chat messages (in multiple possible formats) into a clean, readable conversation history string for an LLM.
'''
def format_history(messages: list) -> str:
    """Format messages as User/Assistant history string.
    Handles both dicts and LangChain BaseMessage (HumanMessage, AIMessage)."""
    if not messages:
        return ""
    parts = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
        else:
            role = getattr(m, "type", "user")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            content = getattr(m, "content", "") or ""
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {content}")
    return "\n".join(parts)


def node_summary(node_name: str, data: dict) -> str:
    """Return brief summary of what a node returned."""
    if node_name == "router":
        a, c = data.get("action", ""), data.get("category") or ""
        return f"action={a}" + (f", category={c}" if c else "")
    if node_name == "tavily":
        out = data.get("output", "")
        n = out.count("Source:") if out else 0
        return f"{n} results" if n else "done"
    if node_name == "vector_store":
        docs = data.get("documents", [])
        return f"{len(docs)} docs"
    if node_name == "grade":
        rel = data.get("relevent", "")
        return f"relevant={rel}"
    if node_name == "send_email":
        return "placeholder"
    if node_name == "generate":
        return "done"
    return "ok"
