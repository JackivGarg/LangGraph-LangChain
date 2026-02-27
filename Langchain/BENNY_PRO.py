import os
import templates
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import sys, time
from langchain_community.vectorstores import FAISS
import common
import store_model_embeddings
load_dotenv()


print("DO YOU WANT TO LOAD THE DATA FROM content.txt FILE TO CREATE VECTOR DB? (yes/no): ")
choice = input().lower()
if choice == 'yes':
    common.reload("context_pro.txt","vector_created_PRO")
    print("Vector DB created and loaded successfully.")
else:
    print("Skipping data load. Using existing vector DB.")

db_2 = FAISS.load_local(
    "vector_created_PRO", 
    store_model_embeddings.embedding,
    allow_dangerous_deserialization=True
)
def pro_bot():
    parser = StrOutputParser()
    retriver=db_2.as_retriever(search_kwargs={"k":6})
    pro_prompt = PromptTemplate(
        template=templates.pro_template_str,
        input_variables=["context", "history", "user_input"]
    )
    from operator import itemgetter
    pro_chain = (
        {
            "context": itemgetter("user_input") | retriver,
            "user_input": itemgetter("user_input"),
            "history": itemgetter("history")
        }
        | pro_prompt
        | store_model_embeddings.model
        | parser
    )
    store = {}

    final_pro_chain = RunnableWithMessageHistory(
        pro_chain,
        common.get_session_history,
        input_messages_key="user_input",  
        history_messages_key="history"
    )
    while True:
        user_query = input("\nYOU (Pro): ")
        
        if user_query.lower() in ['exit', 'quit']: 
            
            session_id = "chat1" 
            if session_id in store:
                del store[session_id] 
                print("Session memory cleared.")
            print("Exiting the chatbot. Goodbye!")
            break
        
        config = {"configurable": {"session_id": "chat1"}}
        response = final_pro_chain.invoke(
            {"user_input": user_query}, 
            config=config
        )
        print("AI: ", end="")
        common.typewriter(str(response), delay=0.01)  
pro_bot()