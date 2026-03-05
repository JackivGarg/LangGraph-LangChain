import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from src.langchain import langchain_mode
from src.agent import langgraph_route_and_respond
from src.agent.prompts import query_rewriter_template
from src.llm import llm

st.set_page_config(page_title="Benny AI - Unified Bot", page_icon="🤖", layout="wide")

if "human_review_toggle" not in st.session_state:
    st.session_state.human_review_toggle = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "pending_interpreted" not in st.session_state:
    st.session_state.pending_interpreted = None
if "pending_session_id" not in st.session_state:
    st.session_state.pending_session_id = None
if "human_review_waiting" not in st.session_state:
    st.session_state.human_review_waiting = False


def clear_human_review_state():
    st.session_state.pending_query = None
    st.session_state.pending_interpreted = None
    st.session_state.pending_interpreted_lc = None
    st.session_state.pending_session_id = None
    st.session_state.human_review_waiting = False
    if "comp_langchain_response" in st.session_state:
        del st.session_state.comp_langchain_response


def rewrite_query_for_display(query: str, session_id: str):
    from src.langchain.history import get_session_history
    from langchain_core.output_parsers import StrOutputParser
    
    history = get_session_history(session_id)
    history_messages = history.messages
    
    history_str = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in history_messages
    )
    
    rewriter_chain = query_rewriter_template | llm | StrOutputParser()
    interpreted = rewriter_chain.invoke({
        "user_input": query,
        "history": history_str
    })
    return query, interpreted.strip().split('\n')[0]


def display_query_box(original: str, interpreted: str):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Your Query**")
        st.info(original)
    with col2:
        st.markdown("**Rewritten Question**")
        st.success(interpreted)


def show_human_review_ui(mode: str):
    prompt = st.session_state.pending_query
    interpreted_q = st.session_state.pending_interpreted
    session_id = st.session_state.pending_session_id or "default"
    
    with st.chat_message("assistant"):
        st.markdown(f"**{mode} Mode**")
        display_query_box(prompt, interpreted_q)
        
        edited_q = st.text_area("Edit query if needed:", value=interpreted_q, height=80, key="edit_query")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            proceed_btn = st.button("Proceed", key="proceed_btn")
        with col2:
            cancel_btn = st.button("Cancel", key="cancel_btn")
        
        if cancel_btn:
            clear_human_review_state()
            st.rerun()
        
        if proceed_btn:
            actual_query = edited_q
            run_langgraph_respond(prompt, session_id, actual_query)
            clear_human_review_state()
            st.rerun()


def main():
    st.title("🤖 Benny AI - Unified Bot")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        mode = st.radio(
            "Select Mode:",
            ["LangChain", "LangGraph", "Comparison"],
            horizontal=True,
            key="mode_selector"
        )
    with col2:
        if mode == "LangGraph" or mode == "Comparison":
            st.session_state.human_review_toggle = st.toggle("Human Review", value=False)

    session_id = st.session_state.get("session_id", "default")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # FIRST: Check if waiting for human review (handles rerun after button click)
    if st.session_state.human_review_waiting:
        show_human_review_ui(mode)
        return
    
    # NEW SUBMISSION: User submits new query
    if prompt := st.chat_input("Ask me anything about Bennett University..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if mode == "LangChain":
            with st.chat_message("user"):
                st.markdown(prompt)
            run_langchain(prompt, session_id)
        
        elif mode == "LangGraph":
            original_q, interpreted_q = rewrite_query_for_display(prompt, session_id)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            if st.session_state.human_review_toggle:
                st.session_state.pending_query = prompt
                st.session_state.pending_interpreted = interpreted_q
                st.session_state.pending_session_id = session_id
                st.session_state.human_review_waiting = True
                show_human_review_ui(mode)
            else:
                run_langgraph_respond(prompt, session_id, interpreted_q)
        
        else:
            # Comparison mode
            if st.session_state.human_review_toggle:
                # Check if we have pending data from previous run
                if st.session_state.get("pending_query") == prompt:
                    # Reuse stored interpreted queries
                    interpreted_q_lg = st.session_state.pending_interpreted
                    original_q_lg, interpreted_q_lc = prompt, prompt
                else:
                    # First time - get interpreted queries
                    original_q_lc, interpreted_q_lc = rewrite_query_for_display(prompt, session_id + "_lc")
                    original_q_lg, interpreted_q_lg = rewrite_query_for_display(prompt, session_id + "_lg")
                    
                    # Store in session state
                    st.session_state.pending_query = prompt
                    st.session_state.pending_interpreted = interpreted_q_lg
                    st.session_state.pending_interpreted_lc = interpreted_q_lc
                    st.session_state.pending_session_id = session_id + "_lg"
                    st.session_state.human_review_waiting = True
                
                # Show comparison UI first
                st.markdown("---")
                st.markdown("### Query Analysis")
                
                col_q1, col_q2 = st.columns(2)
                with col_q1:
                    st.markdown("**LangChain Query**")
                    st.info(prompt)
                with col_q2:
                    st.markdown("**LangGraph Query**")
                    st.info(interpreted_q_lg)
                
                # Two columns: LEFT = LangChain, RIGHT = Human Review
                col1, col2 = st.columns(2)
                
                # LEFT: Run LangChain IMMEDIATELY
                with col1:
                    st.markdown("### LangChain Response")
                    # Check if we have a stored response
                    if st.session_state.get("comp_langchain_response"):
                        st.markdown(st.session_state.comp_langchain_response)
                        full_response1 = st.session_state.comp_langchain_response
                    else:
                        response_container1 = st.empty()
                        full_response1 = ""
                        for chunk in langchain_mode(prompt, session_id + "_lc"):
                            if isinstance(chunk, dict) and "__stats__" in chunk:
                                pass
                            else:
                                full_response1 += chunk
                                response_container1.markdown(full_response1 + "▌")
                        response_container1.markdown(full_response1)
                        st.session_state.comp_langchain_response = full_response1
                
                # RIGHT: Human Review UI
                with col2:
                    st.warning("Human Review Required")
                    edited_q = st.text_area("Edit query if needed:", value=interpreted_q_lg, height=80, key="comp_edit")
                    
                    col_p, col_c = st.columns([1, 1])
                    with col_p:
                        proceed_btn = st.button("Proceed", key="comp_proceed")
                    with col_c:
                        cancel_btn = st.button("Cancel", key="comp_cancel")
                    
                    if cancel_btn:
                        clear_human_review_state()
                        if "comp_langchain_response" in st.session_state:
                            del st.session_state.comp_langchain_response
                        st.rerun()
                    
                    if proceed_btn:
                        actual_query = edited_q
                        
                        # Create columns for both responses
                        col_left, col_right = st.columns(2)
                        
                        # LEFT: LangChain (from storage)
                        with col_left:
                            st.markdown("### LangChain Response")
                            st.markdown(st.session_state.comp_langchain_response)
                        
                        # RIGHT: LangGraph (fresh)
                        with col_right:
                            st.markdown("### LangGraph Response")
                            response_container2 = st.empty()
                            full_response2 = ""
                            for chunk in langgraph_route_and_respond(prompt, session_id + "_lg", use_human_review=True, edited_query=actual_query):
                                if isinstance(chunk, dict) and "__stats__" in chunk:
                                    pass
                                else:
                                    full_response2 += chunk
                                    response_container2.markdown(full_response2 + "▌")
                            response_container2.markdown(full_response2)
                        
                        # Store in messages
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"**Comparison Mode:**\n\n**LangChain:** {full_response1}\n\n---\n\n**LangGraph:** {full_response2}"
                        })
                        
                        clear_human_review_state()
                        if "comp_langchain_response" in st.session_state:
                            del st.session_state.comp_langchain_response
            else:
                run_comparison(prompt, session_id)


def run_langchain(prompt: str, session_id: str):
    with st.chat_message("assistant"):
        st.markdown("**LangChain Mode**")
        original_q, interpreted_q = rewrite_query_for_display(prompt, session_id)
        display_query_box(original_q, interpreted_q)
        
        response_container = st.empty()
        full_response = ""
        for chunk in langchain_mode(prompt, session_id):
            if isinstance(chunk, dict) and "__stats__" in chunk:
                pass
            else:
                full_response += chunk
                response_container.markdown(full_response + "▌")
        response_container.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": f"**LangChain Mode:**\n\n{full_response}"})


def run_langgraph_respond(prompt: str, session_id: str, interpreted_query: str):
    st.markdown("**LangGraph Response:**")
    response_container = st.empty()
    full_response = ""
    for chunk in langgraph_route_and_respond(prompt, session_id, use_human_review=True, edited_query=interpreted_query):
        if isinstance(chunk, dict) and "__stats__" in chunk:
            pass
        else:
            full_response += chunk
            response_container.markdown(full_response + "▌")
    response_container.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": f"**LangGraph Mode:**\n\n{full_response}"})


def run_comparison(prompt: str, base_session_id: str):
    session_id = base_session_id
    
    original_q_lc, interpreted_q_lc = rewrite_query_for_display(prompt, session_id + "_lc")
    original_q_lg, interpreted_q_lg = rewrite_query_for_display(prompt, session_id + "_lg")
    
    st.markdown("---")
    st.markdown("### Query Analysis")
    
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown("**LangChain Query**")
        display_query_box(original_q_lc, interpreted_q_lc)
    with col_q2:
        st.markdown("**LangGraph Query**")
        display_query_box(original_q_lg, interpreted_q_lg)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### LangChain Response")
        response_container1 = st.empty()
        full_response1 = ""
        for chunk in langchain_mode(prompt, session_id + "_lc"):
            if isinstance(chunk, dict) and "__stats__" in chunk:
                pass
            else:
                full_response1 += chunk
                response_container1.markdown(full_response1 + "▌")
        response_container1.markdown(full_response1)
    
    with col2:
        st.markdown("### LangGraph Response")
        response_container2 = st.empty()
        full_response2 = ""
        for chunk in langgraph_route_and_respond(prompt, session_id + "_lg", use_human_review=False, edited_query=interpreted_q_lg):
            if isinstance(chunk, dict) and "__stats__" in chunk:
                pass
            else:
                full_response2 += chunk
                response_container2.markdown(full_response2 + "▌")
        response_container2.markdown(full_response2)
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"**Comparison Mode:**\n\n**LangChain:** {full_response1}\n\n---\n\n**LangGraph:** {full_response2}"
    })


if __name__ == "__main__":
    main()
