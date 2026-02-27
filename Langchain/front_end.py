import streamlit as st
import try2


prompt = st.chat_input("Say something")
if prompt:
    st.write(try2.pro_bot(prompt))