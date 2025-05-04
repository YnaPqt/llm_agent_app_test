import streamlit as st
from llm_engine import QwenChatbot  
from memory import load_memory, save_message

st.set_page_config(page_title="Qwen Chatbot")

st.title("🧠 Qwen3-0.6B Agent")

# Initialize chatbot in session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = QwenChatbot()

chatbot = st.session_state.chatbot

# Display chat history
memory = load_memory()
for m in memory:
    st.chat_message(m["role"]).write(m["message"])

# Chat input
if prompt := st.chat_input("Say something..."):
    save_message("user", prompt)
    response = chatbot.generate_response(prompt)  
    save_message("assistant", response)
    st.chat_message("assistant").write(response)
