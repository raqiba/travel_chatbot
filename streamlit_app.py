import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
import streamlit as st
from main import create_agent

st.set_page_config(page_title="✈️ Travel Planner Chatbot")
st.title("🌍 Travel Assistant Chatbot")

agent = create_agent()

user_input = st.text_input("Ask about flights, visas, or refunds:", placeholder="e.g. Find me a flight to Tokyo...")

if user_input:
    with st.spinner("Thinking..."):
        response = agent.run(user_input)
    st.markdown(response)
