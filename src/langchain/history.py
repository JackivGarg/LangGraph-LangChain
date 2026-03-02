import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory


def get_session_history(session_id: str):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    
    return st.session_state.chat_history[session_id]
