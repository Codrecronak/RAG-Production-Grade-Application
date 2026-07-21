import streamlit as st
from sidebar import display_sidebar
from chat_interface import display_chat_interface
import requests
import uuid

st.title("Langchain RAG Chatbot")

# Initialize a unique browser session ID
if "browser_session_id" not in st.session_state:
    browser_session_id = str(uuid.uuid4())
    st.session_state.browser_session_id = browser_session_id
    st.session_state.session_initialized = False

# On first load of this browser session, clear all documents from previous users
if not st.session_state.session_initialized:
    try:
        # Call backend to clear all documents
        requests.post("https://rag-production-grade-application.onrender.com/clear-all-docs")
        print(f"Cleared documents for new browser session: {st.session_state.browser_session_id}")
    except Exception as e:
        print(f"Error clearing documents on new session: {e}")
    
    st.session_state.session_initialized = True

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Display the sidebar
display_sidebar()

# Display the chat interface
display_chat_interface()
