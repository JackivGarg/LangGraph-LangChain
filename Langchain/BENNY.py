import os
import common
import templates
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import store_model_embeddings
load_dotenv()

db = FAISS.load_local(
    "vector_created_2", 
    store_model_embeddings.embedding,
    allow_dangerous_deserialization=True
)
retriver=db.as_retriever(search_kwargs={"k":3})
print("DO YOU WANT TO LOAD THE DATA FROM content.txt FILE TO CREATE VECTOR DB? (yes/no): ")
choice = input().lower()
if choice == 'yes':
    common.reload("content.txt","vector_created_2")
    print("Vector DB created and loaded successfully.")
else:
    print("Skipping data load. Using existing vector DB.")

def new_bot():
 
    parser = StrOutputParser()
    chain = RunnableSequence(templates.template, store_model_embeddings.model, parser)
    store={}

    with_message_history = RunnableWithMessageHistory(
        chain,  
        common.get_session_history,
        input_messages_key="user_input",  
        history_messages_key="history"   
    )

    config = {"configurable": {"session_id": "chat1"}}



    while(True):
        query=input("ENTER YOUR QUERY (GIVE 'exit','quit' to exit): ")
        if query.lower() in ['exit', 'quit']: 
            
            session_id = "chat1" 
            if session_id in store:
                del store[session_id] 
                print("Session memory cleared.")
            print("Exiting the chatbot. Goodbye!")
            break
        
        res = with_message_history.invoke(
            {
                "user_input": query,
                "context": retriver.invoke(query)
            },
            config=config
        )
        print("AI: ", end="")
        common.typewriter(str(res), delay=0.01)  

new_bot()