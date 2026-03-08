from langchain_community.chat_message_histories import ChatMessageHistory

# Global store for session histories
# Note: In a production app, use a persistent store like Redis
_history_store = {}

def get_session_history(session_id: str):
    if session_id not in _history_store:
        _history_store[session_id] = ChatMessageHistory()
    return _history_store[session_id]
