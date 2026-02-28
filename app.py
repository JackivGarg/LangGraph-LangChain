import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from operator import itemgetter
from langgraphagenticai.prompts import router_template, generate_template
from langgraphagenticai.states.states import Route1
from src.tools.tools import load_vector_store, VALID_CATEGORIES

st.set_page_config(page_title="Benny AI - Unified Bot", page_icon="🤖", layout="wide")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")

from langchain_huggingface import HuggingFaceEndpointEmbeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HUGGINGFACE_KEY,
    model=EMBEDDING_MODEL
)

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

langchain_store = {}

def get_session_history(session_id: str):
    if session_id not in langchain_store:
        langchain_store[session_id] = ChatMessageHistory()
    return langchain_store[session_id]

LANCHAIN_PROMPT = PromptTemplate(
    template="""You are an advanced, analytical AI assistant designed for precise information retrieval and synthesis. 
    You are BENNY PLUS, A PRO MODEL OF BENNY BOT, you have better context that BENNY and you have better template and better tools used.
    **if not mentioned try to answer in brief no need to give entire info availiable , if user mentions 'elaborate' or 'detailed' or anything related to this only then give detailed answer**

    Your DEVELOPER IS JACKIV GARG.
    But if person ask you things other than Bennett University or anything other than acedemics reply like a normal bot would but very very short 
    You are provided with multiple retrieved context fragments that may contain the answer to the user's question.

    Your Objectives:
    1. **Analyze:** Carefully review all the provided context chunks below.
    2. **Synthesize:** Combine information from different chunks to form a comprehensive answer.
    3. **Reason:** Think step-by-step. If pieces of information conflict, prioritize the most specific or recent details.
    4. **Clarify:** If the context is insufficient, clearly state what is missing. Do not make up facts.

    Tone & Style Guidelines:
    - Professional, concise, and structured.
    - Use Bullet points for lists.
    - Language: Use the language used by user in that promt or if any specified in promt 
    

    ---
    **Context from Database:**
    {context}

    **Chat History:**
    {history}
    ---

    User Query: {user_input}

    Answer:
    """,
    input_variables=["context", "history", "user_input"]
)

@st.cache_resource
def get_combined_vectorstore():
    all_texts = []
    all_metadatas = []
    
    for category in VALID_CATEGORIES:
        try:
            vs = load_vector_store(category)
            docs = vs.similarity_search(" ", k=10)
            for doc in docs:
                all_texts.append(doc.page_content)
                all_metadatas.append(doc.metadata)
        except Exception as e:
            continue
    
    if all_texts:
        return FAISS.from_texts(all_texts, embedding, metadatas=all_metadatas)
    return None

db_2 = get_combined_vectorstore()

def build_pro_chain():
    parser = StrOutputParser()
    retriver = db_2.as_retriever(search_kwargs={"k": 3})

    pro_chain = (
        {
            "context": itemgetter("user_input") | retriver,
            "user_input": itemgetter("user_input"),
            "history": itemgetter("history")
        }
        | LANCHAIN_PROMPT
        | llm
        | parser
    )

    final_pro_chain = RunnableWithMessageHistory(
        pro_chain,
        get_session_history,
        input_messages_key="user_input",
        history_messages_key="history"
    )
    return final_pro_chain

_FINAL_PRO_CHAIN = build_pro_chain()

def langchain_mode(query: str, session_id: str):
    config = {"configurable": {"session_id": session_id}}
    response = _FINAL_PRO_CHAIN.invoke({"user_input": query}, config=config)
    
    for chunk in response:
        yield chunk

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
    
    response = generate_chain.invoke({
        "context": context,
        "history": history_str,
        "user_input": query
    })
    
    for chunk in response:
        yield chunk

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
