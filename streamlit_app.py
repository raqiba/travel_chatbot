import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import streamlit as st
from main import create_agent

# Streamlit config
st.set_page_config(page_title="✈️ Travel Planner Chatbot")
st.title("🌍 Travel Assistant Chatbot")

# Initialize agent
if "agent" not in st.session_state:
    st.session_state.agent = create_agent()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box (chat style)
user_input = st.chat_input("Ask about flights, visas, or refunds...")

# If user sends a message
if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    with st.spinner("Thinking..."):
        response = st.session_state.agent.run(user_input)
    st.session_state.chat_history.append({"role": "bot", "text": response})

# Display all messages in chat format
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["text"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["text"])
