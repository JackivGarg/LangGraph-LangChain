import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

from src.langchain import langchain_mode
from src.langgraph import langgraph_route_and_respond

st.set_page_config(page_title="Benny AI - Unified Bot", page_icon="🤖", layout="wide")


def main():
    st.title("🤖 Benny AI - Unified Bot")
    
    mode = st.radio(
        "Select Mode:",
        ["LangChain", "LangGraph", "Comparison"],
        horizontal=True,
        key="mode_selector"
    )
    
    session_id = st.session_state.get("session_id", "default")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me anything about Bennett University..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if mode == "LangChain":
            with st.chat_message("assistant"):
                st.markdown("**LangChain Mode:**")
                response_container = st.empty()
                full_response = ""
                for chunk in langchain_mode(prompt, session_id):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": f"**LangChain Mode:**\n\n{full_response}"})
        
        elif mode == "LangGraph":
            with st.chat_message("assistant"):
                st.markdown("**LangGraph Mode:**")
                response_container = st.empty()
                full_response = ""
                for chunk in langgraph_route_and_respond(prompt, session_id):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": f"**LangGraph Mode:**\n\n{full_response}"})
        
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**LangChain Response:**")
                response_container1 = st.empty()
                full_response1 = ""
                for chunk in langchain_mode(prompt, session_id + "_lc"):
                    full_response1 += chunk
                    response_container1.markdown(full_response1 + "▌")
                response_container1.markdown(full_response1)
            
            with col2:
                st.markdown("**LangGraph Response:**")
                response_container2 = st.empty()
                full_response2 = ""
                for chunk in langgraph_route_and_respond(prompt, session_id + "_lg"):
                    full_response2 += chunk
                    response_container2.markdown(full_response2 + "▌")
                response_container2.markdown(full_response2)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"**Comparison Mode:**\n\n**LangChain:** {full_response1}\n\n---\n\n**LangGraph:** {full_response2}"
            })


if __name__ == "__main__":
    main()
